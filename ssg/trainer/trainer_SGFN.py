import os
import copy
import torch
import time
import logging
import ssg
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
from ssg.utils.util_eva import EvalSceneGraphBatch, EvalUpperBound
import codeLib.utils.moving_average as moving_average
from codeLib.models import BaseTrainer
from tqdm import tqdm
import codeLib.utils.string_numpy as snp
from ssg.utils.util_data import match_class_info_from_two, merge_batch_mask2inst
from ssg import define
from ssg.utils.graph_vis import DrawSceneGraph
from ssg.trainer.eval_inst import EvalInst
import torch.nn.functional as F

logger_py = logging.getLogger(__name__)

class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, clip_model_name="ViT-B/32"):
        super().__init__()
        import clip
        self.model, _ = clip.load(clip_model_name, device="cpu")
        self.text_encoder = self.model.encode_text
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        import clip
        text_tokens = clip.tokenize(text).to(next(self.parameters()).device)
        return self.text_encoder(text_tokens)
    
class TripletProjector(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, output_dim=512):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(node_dim * 2 + edge_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim)
        )
    
    def forward(self, subj_feat, obj_feat, rel_feat):
        combined = torch.cat([subj_feat, rel_feat, obj_feat], dim=1)
        return self.proj(combined)

class Trainer_ALIGN(BaseTrainer, EvalInst):
    def __init__(self, cfg, model, node_cls_names: list, edge_cls_names: list,
                 device=None, **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model
        self.w_node_cls = kwargs.get('w_node_cls', None)
        self.w_edge_cls = kwargs.get('w_edge_cls', None)
        self.node_cls_names = node_cls_names
        self.edge_cls_names = edge_cls_names

        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)

        if self.w_node_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_edge_cls = self.w_edge_cls.to(self._device)

        self.eva_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            save_prediction=False,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=0, none_name=define.NAME_NONE)

        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(
                weight=self.w_edge_cls)
            
        self.clip_text_encoder = CLIPTextEncoder(cfg.clip_model_name)
        self.clip_text_encoder.to(self._device)
        
        node_dim = model.get_node_dim() if hasattr(model, 'get_node_dim') else 256
        edge_dim = model.get_edge_dim() if hasattr(model, 'get_edge_dim') else 256
        self.triplet_projector = TripletProjector(node_dim, edge_dim)
        self.triplet_projector.to(self._device)
        
        self.lambda_clip_obj = cfg.get('lambda_clip_obj', 1.0)
        self.lambda_clip_rel = cfg.get('lambda_clip_rel', 1.0)
        self.lambda_clip_pred = cfg.get('lambda_clip_pred', 1.0)
        
        self._text_embeddings_cache = {}

    def _get_text_embedding(self, text_template, fill_values):
        text = text_template.format(**fill_values)
        
        if text in self._text_embeddings_cache:
            return self._text_embeddings_cache[text]
        
        with torch.no_grad():
            embedding = self.clip_text_encoder([text])
            embedding = F.normalize(embedding, p=2, dim=1)
            self._text_embeddings_cache[text] = embedding
            
        return embedding

    def zero_metrics(self):
        self.eva_tool.reset()

    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=topk, save_prediction=True, none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        for data in tqdm(it_dataset, leave=False):
            eval_step_dict = self.eval_step(data, eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)

        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool

    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool=self.eva_tool)
        if 'loss' not in logs:
            return logs

        if not check_valid(logs):
            logs['loss'].backward()
            check_weights(self.model.state_dict())
            self.optimizer.step()
        else:
            logger_py.info('skip loss backward due to nan occurs')
        return logs

    def eval_step(self, data, eval_tool=None):
        self.model.eval()
        eval_dict = {}
        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        return eval_dict

    def compute_loss(self, data, eval_mode=False, it=None, eval_tool=None):
        logs = {}

        data = data.to(self._device)
        
        gt_node = data['node'].y
        gt_edge = data['node', 'to', 'node'].y
        
        node_cls, edge_cls = self.model(data)
        
        node_features = data['node'].x 
        edge_features = data['node', 'to', 'node'].x if hasattr(data['node', 'to', 'node'], 'x') else None  # 엣지 특징
        
        logs['loss'] = 0
        
        if self.cfg.training.lambda_mode == 'dynamic':
            batch_node = node_cls.shape[0]
            self.cfg.training.lambda_node = 1

            if edge_cls is not None:
                batch_edge = edge_cls.shape[0]
                self.cfg.training.lambda_edge = batch_edge / batch_node

        self.calc_node_loss(logs, node_cls, gt_node, self.w_node_cls)

        if edge_cls is not None:
            if edge_cls.shape[0] != gt_edge.shape[0]:
                if not eval_mode:
                    print(f"Warning: Size mismatch between edge_cls {edge_cls.shape} and gt_edge {gt_edge.shape}")
                    
                    min_edges = min(edge_cls.shape[0], gt_edge.shape[0])
                    edge_cls = edge_cls[:min_edges]
                    gt_edge = gt_edge[:min_edges]
                    
                    data['node', 'to', 'node'].y = gt_edge
                    
                    print(f"Adjusted edge_cls and gt_edge to size {min_edges}")
                else:
                    print(f"Error: Size mismatch in eval mode between edge_cls {edge_cls.shape} and gt_edge {gt_edge.shape}")
                    return logs
                    
            self.calc_edge_loss(logs, edge_cls, gt_edge, self.w_edge_cls)
            
            if not eval_mode and edge_features is not None:
                edge_index = data['node', 'to', 'node'].edge_index
                
                batch_size = min(128, edge_index.shape[1])
                if batch_size > 0:
                    sample_indices = torch.randperm(edge_index.shape[1])[:batch_size]
                    sampled_edges = edge_index[:, sample_indices]
                    
                    subject_indices = sampled_edges[0]
                    object_indices = sampled_edges[1]
                    subject_features = node_features[subject_indices]
                    object_features = node_features[object_indices]
                    relation_features = edge_features[sample_indices] if edge_features is not None else torch.zeros_like(subject_features)
                    
                    subject_cls_pred = torch.softmax(node_cls[subject_indices], dim=1)
                    object_cls_pred = torch.softmax(node_cls[object_indices], dim=1)
                    relation_cls_pred = torch.softmax(edge_cls[sample_indices], dim=1) if not self.cfg.model.multi_rel else torch.sigmoid(edge_cls[sample_indices])
                    
                    subject_cls_idx = subject_cls_pred.argmax(dim=1)
                    object_cls_idx = object_cls_pred.argmax(dim=1)
                    relation_cls_idx = relation_cls_pred.argmax(dim=1) if not self.cfg.model.multi_rel else relation_cls_pred > 0.5
                    
                    clip_obj_loss = 0
                    clip_pred_loss = 0
                    clip_rel_loss = 0
                    
                    for i in range(batch_size):
                        subject_name = self.node_cls_names[subject_cls_idx[i]]
                        object_name = self.node_cls_names[object_cls_idx[i]]
                        
                        if self.cfg.model.multi_rel:
                            rel_idx = relation_cls_pred[i].argmax().item()
                            relation_name = self.edge_cls_names[rel_idx]
                        else:
                            relation_name = self.edge_cls_names[relation_cls_idx[i]]
                        
                        subject_text_emb = self._get_text_embedding("a point cloud of a {obj}", {"obj": subject_name})
                        object_text_emb = self._get_text_embedding("a point cloud of a {obj}", {"obj": object_name})
                        
                        relation_text_emb = self._get_text_embedding("{pred}", {"pred": relation_name})
                        
                        triplet_text_emb = self._get_text_embedding(
                            "a point cloud of a {subj} {pred} a {obj}", 
                            {"subj": subject_name, "pred": relation_name, "obj": object_name}
                        )
                        
                        triplet_feature = self.triplet_projector(
                            subject_features[i].unsqueeze(0), 
                            object_features[i].unsqueeze(0), 
                            relation_features[i].unsqueeze(0)
                        )
                        triplet_feature = F.normalize(triplet_feature, p=2, dim=1)
                        
                        subj_feat_norm = F.normalize(subject_features[i].unsqueeze(0), p=2, dim=1)
                        obj_feat_norm = F.normalize(object_features[i].unsqueeze(0), p=2, dim=1)
                        rel_feat_norm = F.normalize(relation_features[i].unsqueeze(0), p=2, dim=1)
                        
                        clip_obj_loss += (1 - F.cosine_similarity(subj_feat_norm, subject_text_emb)).mean()
                        clip_obj_loss += (1 - F.cosine_similarity(obj_feat_norm, object_text_emb)).mean()
                        clip_pred_loss += (1 - F.cosine_similarity(rel_feat_norm, relation_text_emb)).mean()
                        clip_rel_loss += (1 - F.cosine_similarity(triplet_feature, triplet_text_emb)).mean()
                    
                    clip_obj_loss = clip_obj_loss / (2 * batch_size)
                    clip_pred_loss = clip_pred_loss / batch_size
                    clip_rel_loss = clip_rel_loss / batch_size
                    
                    logs['loss_clip_obj'] = clip_obj_loss
                    logs['loss_clip_pred'] = clip_pred_loss
                    logs['loss_clip_rel'] = clip_rel_loss
                    
                    logs['loss'] += self.lambda_clip_obj * clip_obj_loss
                    logs['loss'] += self.lambda_clip_pred * clip_pred_loss
                    logs['loss'] += self.lambda_clip_rel * clip_rel_loss

        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls,
            node_cls_gt=gt_node,
            edge_cls_pred=edge_cls,
            edge_cls_gt=gt_edge
        )
        for k, v in metrics.items():
            logs[k] = v

        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(), dim=1)
            data['node'].pd = node_cls.detach()

            if edge_cls is not None:
                if 'edge_index_original' not in data['node', 'to', 'node']:
                    data['node', 'to', 'node'].edge_index_original = data['node', 'to', 'node'].edge_index.clone()
                
                if edge_cls.shape[0] != data['node', 'to', 'node'].edge_index.shape[1]:
                    data['node', 'to', 'node'].edge_index = data['node', 'to', 'node'].edge_index_original[:, :edge_cls.shape[0]]
                    print(f"Adjusted edge_index from {data['node', 'to', 'node'].edge_index_original.shape} to {data['node', 'to', 'node'].edge_index.shape}")
                
                edge_cls = torch.sigmoid(edge_cls.detach()) if self.cfg.model.multi_rel else torch.softmax(edge_cls.detach(), dim=1)
                data['node', 'to', 'node'].pd = edge_cls.detach()
            
            eval_tool.add(data)

        return logs

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj

    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt, weights=None):
        if self.cfg.model.multi_rel:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel
        logs['loss_rel'] = loss_rel

    def visualize(self, eval_tool=None):
        if eval_tool is None:
            eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot=False
        )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }

    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()

        for k, v in obj_.items():
            output[k+'_node_cls'] = v
        for k, v in edge_.items():
            output[k+'_edge_cls'] = v
        return output

class Trainer_SGFN(BaseTrainer, EvalInst):
    def __init__(self, cfg, model, node_cls_names: list, edge_cls_names: list,
                 device=None,  **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model  # .to(self._device)
        # self.optimizer = optimizer
        self.w_node_cls = kwargs.get('w_node_cls', None)
        self.w_edge_cls = kwargs.get('w_edge_cls', None)
        self.node_cls_names = node_cls_names  # kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names  # kwargs['edge_cls_names']

        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)

        if self.w_node_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_edge_cls = self.w_edge_cls.to(self._device)

        self.eva_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            save_prediction=False,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=0, none_name=define.NAME_NONE)  # do not calculate topK in training mode
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(
                weight=self.w_edge_cls)

    def zero_metrics(self):
        self.eva_tool.reset()

    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraphBatch(
            self.node_cls_names, self.edge_cls_names,
            multi_rel_prediction=self.cfg.model.multi_rel,
            k=topk, save_prediction=True, none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        # time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset, leave=False):
            eval_step_dict = self.eval_step(data, eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            # break
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool

    def sample(self, dataloader):
        pass

    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool=self.eva_tool)
        if 'loss' not in logs:
            return logs

        if not check_valid(logs):
            logs['loss'].backward()
            check_weights(self.model.state_dict())
            self.optimizer.step()
        else:
            logger_py.info('skip loss backward due to nan occurs')
        return logs

    def eval_step(self, data, eval_tool=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict = {}
        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        # for (k, v) in eval_dict.items():
        #     eval_dict[k] = v.item()
        return eval_dict

    def compute_loss(self, data, eval_mode=False, it=None, eval_tool=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        logs = {}

        # Process data dictionary
        data = data.to(self._device)
        # data = self.process_data_dict(data)

        # Shortcuts
        # scan_id = data['scan_id']
        gt_node = data['node'].y
        gt_edge = data['node', 'to', 'node'].y

        # mask2instance = data['node'].idx2oid[0]
        # edge_indices_node_to_node = data['node','to','node'].edge_index

        # gt_cls = data['gt_cls']
        # gt_rel = data['gt_rel']
        # mask2instance = data['mask2instance']
        # node_edges_ori = data['node_edges']
        # data['node_edges'] = data['node_edges'].t().contiguous()

        # check input valid
        # if node_edges_ori.ndim==1:
        #     return {}

        # print('')
        # print('gt_rel.sum():',gt_rel.sum())

        ''' make forward pass through the network '''
        node_cls, edge_cls = self.model(data)

        ''' calculate loss '''
        logs['loss'] = 0

        if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
            batch_node = node_cls.shape[0]
            self.cfg.training.lambda_node = 1

            if edge_cls is not None:
                batch_edge = edge_cls.shape[0]
                self.cfg.training.lambda_edge = batch_edge / batch_node

        ''' 1. node class loss'''
        self.calc_node_loss(logs, node_cls, gt_node, self.w_node_cls)

        ''' 2. edge class loss '''
        if edge_cls is not None:
            if edge_cls.shape[0] != gt_edge.shape[0]:
                if not eval_mode:
                    print(f"Warning: Size mismatch between edge_cls {edge_cls.shape} and gt_edge {gt_edge.shape}")
                    
                    min_edges = min(edge_cls.shape[0], gt_edge.shape[0])
                    edge_cls = edge_cls[:min_edges]
                    gt_edge = gt_edge[:min_edges]
                    
                    data['node', 'to', 'node'].y = gt_edge
                    
                    print(f"Adjusted edge_cls and gt_edge to size {min_edges}")
                else:
                    print(f"Error: Size mismatch in eval mode between edge_cls {edge_cls.shape} and gt_edge {gt_edge.shape}")
                    return logs
                    
            self.calc_edge_loss(logs, edge_cls, gt_edge, self.w_edge_cls)

        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls,
            node_cls_gt=gt_node,
            edge_cls_pred=edge_cls,
            edge_cls_gt=gt_edge
        )
        for k, v in metrics.items():
            logs[k] = v

        ''' eval tool '''
        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(), dim=1)
            data['node'].pd = node_cls.detach()

            if edge_cls is not None:
                if 'edge_index_original' not in data['node', 'to', 'node']:
                    data['node', 'to', 'node'].edge_index_original = data['node', 'to', 'node'].edge_index.clone()
                
                if edge_cls.shape[0] != data['node', 'to', 'node'].edge_index.shape[1]:
                    data['node', 'to', 'node'].edge_index = data['node', 'to', 'node'].edge_index_original[:, :edge_cls.shape[0]]
                    print(f"Adjusted edge_index from {data['node', 'to', 'node'].edge_index_original.shape} to {data['node', 'to', 'node'].edge_index.shape}")
                
                edge_cls = torch.sigmoid(edge_cls.detach())
                data['node', 'to', 'node'].pd = edge_cls.detach()
            
            eval_tool.add(data)

        # if check_valid(logs):
        #     raise RuntimeWarning()
        #     print('has nan')
        return logs
        # return loss if eval_mode else loss['loss']

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        classification loss
        attribute loss
        affordance loss
        '''
        # loss_obj = F.nll_loss(node_cls_pred, node_cls_gt, weight = weights)
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj

    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt, weights=None):
        if self.cfg.model.multi_rel:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel
        logs['loss_rel'] = loss_rel

    def visualize(self, eval_tool=None):
        if eval_tool is None:
            eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot=False
        )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }

    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()

        for k, v in obj_.items():
            output[k+'_node_cls'] = v
        for k, v in edge_.items():
            output[k+'_edge_cls'] = v
        return output

    def evaluate_inst_incre(self, dataset_seg, dataset_inst, topk):
        is_eval_image = self.cfg.model.method in ['imp']
        ignore_missing = self.cfg.eval.ignore_missing

        '''add a none class for missing instances'''
        (scanid2idx_seg, _, node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
            seg_valid_node_cls_indices, inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices, inst_valid_edge_cls_indices) = \
            match_class_info_from_two(
                dataset_seg, dataset_inst, multi_rel=self.cfg.model.multi_rel)

        '''all'''
        eval_tool_all = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
                                            multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
                                            none_name=define.NAME_NONE, ignore_none=False)

        '''ignore none'''
        # eval_tool_ignore_none = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
        #                                 multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
        #                                 none_name=define.NAME_NONE,ignore_none=True)
        eval_tools = {'all': eval_tool_all,
                      #   'ignore_none': eval_tool_ignore_none
                      }

        # eval_upper_bound
        eval_UpperBound = EvalUpperBound(node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
                                         multi_rel=self.cfg.model.multi_rel, topK=topk, none_name=define.NAME_NONE)

        eval_list = defaultdict(moving_average.MA)

        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans, index)  # self.scans[idx]
            scanid2idx_seg[scan_id] = index

        scanid2idx_inst = dict()
        for index in range(len(dataset_inst)):
            scan_id = snp.unpack(dataset_inst.scans, index)  # self.scans[idx]
            scanid2idx_inst[scan_id] = index

        '''start eval'''
        self.model.eval()
        for index in tqdm(range(len(dataset_inst))):
            data_inst = dataset_inst.__getitem__(index)
            scan_id_inst = data_inst['scan_id']

            if scan_id_inst not in scanid2idx_seg:
                data_seq_seq = None
            else:
                index_seg = scanid2idx_seg[scan_id_inst]
                data_seq_seq = dataset_seg.__getitem__(index_seg)

            '''process seg'''
            eval_dict = {}
            with torch.no_grad():
                logs = {}
                data_inst = self.process_data_dict(data_inst)

                # use the latest timestamp to calculate the upperbound
                if data_seq_seq is not None:
                    key_int = sorted([int(t) for t in data_seq_seq])
                    latest_t = max(key_int)
                    data_lastest = self.process_data_dict(
                        data_seq_seq[str(latest_t)])

                else:
                    data_lastest = None
                eval_UpperBound(data_lastest, data_inst, is_eval_image)
                # continue

                # Shortcuts
                # scan_id = data_inst['scan_id']
                inst_oids = data_inst['node'].oid
                # inst_mask2instance = data_inst['node'].idx2oid[0]#data_inst['mask2instance']
                inst_gt_cls = data_inst['node'].y  # data_inst['gt_cls']
                # data_inst['seg_gt_rel']
                inst_gt_rel = data_inst['node', 'to', 'node'].y
                # data_inst['node_edges']
                inst_node_edges = data_inst['node', 'to', 'node'].edge_index
                gt_relationships = data_inst['relationships']

                if data_seq_seq is None:
                    '''
                    If no target scan in dataset_seg is found, set all prediction to none
                    '''
                    # Nodes
                    node_pred = torch.zeros_like(torch.nn.functional.one_hot(
                        inst_gt_cls, len(node_cls_names))).float()
                    node_pred[:, noneidx_node_cls] = 1.0

                    # Edges
                    if not self.cfg.model.multi_rel:
                        edge_pred = torch.zeros_like(torch.nn.functional.one_hot(
                            inst_gt_rel, len(edge_cls_names))).float()
                        edge_pred[:, noneidx_edge_cls] = 1.0
                    else:
                        edge_pred = torch.zeros_like(inst_gt_rel).float()

                    # log
                    data_inst['node'].pd = node_pred.detach()
                    data_inst['node', 'to', 'node'].pd = edge_pred.detach()
                    for eval_tool in eval_tools.values():
                        eval_tool.add(data_inst)
                    continue

                predictions_weights = dict()
                predictions_weights['node'] = dict()
                predictions_weights['node', 'to', 'node'] = dict()
                merged_node_cls = torch.zeros(
                    len(inst_oids), len(node_cls_names)).to(self.cfg.DEVICE)
                merged_node_cls_gt = (torch.ones(
                    len(inst_oids), dtype=torch.long) * noneidx_node_cls).to(self.cfg.DEVICE)

                # convert them to list
                assert inst_node_edges.shape[0] == 2
                inst_node_edges = inst_node_edges.tolist()
                inst_oids = inst_oids.tolist()

                '''merge batched dict to one single dict'''
                # mask2seg= merge_batch_mask2inst(mask2seg)
                # inst_mask2inst=merge_batch_mask2inst(inst_mask2instance)

                # build search list for GT edge pairs
                inst_gt_pairs = set()
                # This collects "from" and "to" instances pair as key  -> predicate label
                inst_gt_rel_dict = dict()
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = inst_node_edges[0][idx], inst_node_edges[1][idx]
                    src_oid, tgt_oid = inst_oids[src_idx], inst_oids[tgt_idx]
                    inst_gt_pairs.add((src_oid, tgt_oid))
                    inst_gt_rel_dict[(src_oid, tgt_oid)] = inst_gt_rel[idx]
                inst_gt_pairs = [pair for pair in inst_gt_pairs]

                '''merge predictions'''
                merged_edge_cls = torch.zeros(
                    len(inst_gt_rel), len(edge_cls_names)).to(self.cfg.DEVICE)
                if not self.cfg.model.multi_rel:
                    merged_edge_cls_gt = (torch.ones(
                        len(inst_gt_rel), dtype=torch.long) * noneidx_edge_cls).to(self.cfg.DEVICE)
                else:
                    merged_edge_cls_gt = inst_gt_rel.clone().float()

                for timestamp in key_int:
                    timestamp = key_int[-1]
                    timestamp = str(timestamp)
                    data_seg = self.process_data_dict(data_seq_seq[timestamp])

                    assert data_seg['scan_id'] == data_inst['scan_id']

                    if not is_eval_image:
                        # seg_gt_cls = data_seg['node'].y
                        seg_gt_rel = data_seg['node', 'to', 'node'].y
                        seg_oids = data_seg['node'].oid
                        seg_node_edges = data_seg['node',
                                                  'to', 'node'].edge_index
                    else:
                        # seg_gt_cls = data_seg['roi'].y
                        seg_gt_rel = data_seg['roi', 'to', 'roi'].y
                        # mask2seg = data_seg['roi'].idx2oid[0]
                        seg_oids = data_seg['roi'].oid
                        seg_node_edges = data_seg['roi',
                                                  'to', 'roi'].edge_index
                        # seg2inst = data_seg['roi'].get('idx2iid',None)

                    ''' make forward pass through the network '''
                    node_cls, edge_cls = self.model(data_seg)

                    # convert them to list
                    assert seg_node_edges.shape[0] == 2
                    seg_node_edges = seg_node_edges.tolist()
                    seg_oids = seg_oids.tolist()

                    '''merge prediction from seg to instance (in case of "same part")'''
                    # use list bcuz may have multiple predictions on the same object instance
                    seg_oid2idx = defaultdict(list)
                    for idx in range(len(seg_oids)):
                        seg_oid2idx[seg_oids[idx]].append(idx)

                    '''merge nodes'''
                    merged_idx2oid = dict()
                    merged_oid2idx = dict()

                    for idx in range(len(inst_oids)):
                        oid = inst_oids[idx]
                        # merge predictions
                        if not ignore_missing:
                            merged_oid2idx[oid] = idx
                            merged_idx2oid[idx] = oid
                            # use GT class
                            merged_node_cls_gt[idx] = inst_gt_cls[idx]
                            if oid in seg_oid2idx:
                                '''merge nodes'''
                                predictions = node_cls[seg_oid2idx[oid]
                                                       ]  # get all predictions on that instance
                                node_cls_pred = torch.softmax(predictions, dim=1).mean(
                                    dim=0)  # averaging the probability

                                # Weighted Sum
                                if idx not in predictions_weights['node']:
                                    predictions_weights['node'][idx] = 0
                                merged_node_cls[idx, inst_valid_node_cls_indices] = \
                                    (merged_node_cls[idx, inst_valid_node_cls_indices] * predictions_weights['node'][idx] +
                                        node_cls_pred[seg_valid_node_cls_indices]
                                     ) / (predictions_weights['node'][idx]+1)
                                predictions_weights['node'][idx] += 1

                            else:
                                assert noneidx_node_cls is not None
                                # Only do this in the last estimation
                                if int(timestamp) == key_int[-1]:
                                    merged_node_cls[idx,
                                                    noneidx_node_cls] = 1.0
                        else:
                            raise NotImplementedError()
                            if inst not in inst2masks:
                                continue
                            merged_mask2instance[counter] = inst
                            merged_instance2idx[inst] = counter
                            predictions = node_cls[inst2masks[inst]]
                            node_cls_pred = torch.softmax(
                                predictions, dim=1).mean(dim=0)
                            merged_node_cls[counter,
                                            inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]
                            merged_node_cls_gt[counter] = inst_gt_cls[mask_old]
                            counter += 1
                    if ignore_missing:
                        raise NotImplementedError()
                        merged_node_cls = merged_node_cls[:counter]
                        merged_node_cls_gt = merged_node_cls_gt[:counter]

                    '''merge batched dict to one single dict'''
                    # For segment level
                    # map edge predictions on the same pair of instances.
                    merged_edge_cls_dict = defaultdict(list)
                    for idx in range(len(seg_gt_rel)):
                        src_idx, tgt_idx = seg_node_edges[0][idx], seg_node_edges[1][idx]
                        src_oid, tgt_oid = seg_oids[src_idx], seg_oids[tgt_idx]
                        pair = (src_oid, tgt_oid)
                        if pair in inst_gt_pairs:
                            merged_edge_cls_dict[pair].append(edge_cls[idx])
                        else:
                            # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                            pass

                    '''merge predictions'''
                    merged_node_edges = list()  # new edge_indices
                    for idx, pair in enumerate(inst_gt_pairs):
                        inst_edge_cls = inst_gt_rel_dict[pair]
                        if ignore_missing:
                            if pair[0] not in merged_oid2idx:
                                continue
                            if pair[1] not in merged_oid2idx:
                                continue
                        # merge edge index to the new mask ids
                        src_idx = merged_oid2idx[pair[0]]
                        tgt_idx = merged_oid2idx[pair[1]]
                        merged_node_edges.append([src_idx, tgt_idx])

                        if pair in merged_edge_cls_dict:
                            edge_pds = torch.stack(merged_edge_cls_dict[pair])
                            edge_pds = edge_pds[:, seg_valid_edge_cls_indices]
                            # seg_valid_edge_cls_indices

                            # ignore same part
                            if not self.cfg.model.multi_rel:
                                edge_pds = torch.softmax(
                                    edge_pds, dim=1).mean(0)
                            else:
                                edge_pds = torch.sigmoid(edge_pds).mean(0)

                            # Weighted Sum
                            if idx not in predictions_weights['node', 'to', 'node']:
                                predictions_weights['node',
                                                    'to', 'node'][idx] = 0
                            merged_edge_cls[idx, inst_valid_edge_cls_indices] = \
                                (merged_edge_cls[idx, inst_valid_edge_cls_indices]*predictions_weights['node', 'to', 'node'][idx] +
                                    edge_pds) / (predictions_weights['node', 'to', 'node'][idx]+1)
                            predictions_weights['node', 'to', 'node'][idx] += 1
                            # merged_edge_cls[counter,inst_valid_edge_cls_indices] = edge_pds
                        elif not self.cfg.model.multi_rel:
                            # Only do this in the last estimation
                            if int(timestamp) == key_int[-1]:
                                merged_edge_cls[idx, noneidx_edge_cls] = 1.0

                        if not self.cfg.model.multi_rel:
                            merged_edge_cls_gt[idx] = inst_edge_cls

                    if ignore_missing:
                        raise NotImplementedError()
                        merged_edge_cls = merged_edge_cls[:counter]
                        merged_edge_cls_gt = merged_edge_cls_gt[:counter]
                    merged_node_edges = torch.tensor(
                        merged_node_edges, dtype=torch.long)
                    break
                merged_node_edges = merged_node_edges.t().contiguous()

            data_inst['node'].pd = merged_node_cls.detach()
            data_inst['node'].y = merged_node_cls_gt.detach()
            data_inst['node', 'to', 'node'].pd = merged_edge_cls.detach()
            data_inst['node', 'to', 'node'].y = merged_edge_cls_gt.detach()
            data_inst['node', 'to', 'node'].edge_index = merged_node_edges
            data_inst['node'].clsIdx = torch.from_numpy(
                np.array([k for k in merged_idx2oid.values()]))
            for eval_tool in eval_tools.values():
                eval_tool.add(data_inst)

        eval_dict = dict()
        eval_dict['visualization'] = dict()
        for eval_type, eval_tool in eval_tools.items():

            obj_, edge_ = eval_tool.get_mean_metrics()
            for k, v in obj_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_node_cls'] = v
            for k, v in edge_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_edge_cls'] = v

            for k, v in eval_list.items():
                eval_dict[eval_type+'_'+k] = v.avg

            vis = self.visualize(eval_tool=eval_tool)

            vis = {eval_type+'_'+k: v for k, v in vis.items()}

            eval_dict['visualization'].update(vis)

        return eval_dict, eval_tools, eval_UpperBound.eval_tool

    def visualize_inst_incre(self, dataset_seg, topk):
        ignore_missing = self.cfg.eval.ignore_missing
        '''add a none class for missing instances'''
        node_cls_names = copy.copy(self.node_cls_names)
        edge_cls_names = copy.copy(self.edge_cls_names)
        if define.NAME_NONE not in self.node_cls_names:
            node_cls_names.append(define.NAME_NONE)
        if define.NAME_NONE not in self.edge_cls_names:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        # if define.NAME_SAME_PART in edge_cls_names: edge_cls_names.remove(define.NAME_SAME_PART)

        noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)

        '''
        Find index mapping. Ignore NONE for nodes since it is used for mapping missing instance.
        Ignore SAME_PART for edges.
        '''
        seg_valid_node_cls_indices = []
        inst_valid_node_cls_indices = []
        for idx in range(len(self.node_cls_names)):
            name = self.node_cls_names[idx]
            if name == define.NAME_NONE:
                continue
            seg_valid_node_cls_indices.append(idx)
        for idx in range(len(node_cls_names)):
            name = node_cls_names[idx]
            if name == define.NAME_NONE:
                continue
            inst_valid_node_cls_indices.append(idx)

        seg_valid_edge_cls_indices = []
        inst_valid_edge_cls_indices = []
        for idx in range(len(self.edge_cls_names)):
            name = self.edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            seg_valid_edge_cls_indices.append(idx)
        for idx in range(len(edge_cls_names)):
            name = edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            inst_valid_edge_cls_indices.append(idx)

        eval_tool = EvalSceneGraphBatch(
            node_cls_names, edge_cls_names,
            multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
            none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        '''check'''
        # length
        # print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        print('ignore missing', ignore_missing)
        # classes

        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans, index)  # self.scans[idx]
            scanid2idx_seg[scan_id] = index

        '''start eval'''
        acc_time = 0
        timer_counter = 0
        self.model.eval()
        for index in tqdm(range(len(dataset_seg))):
            # for data_inst in seg_dataloader:
            # data = dataset_seg.__getitem__(index)
            # scan_id_inst = data['scan_id'][0]
            # if scan_id_inst not in scanid2idx_seg: continue
            scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
            index = scanid2idx_seg[scan_id]
            data_seg = dataset_seg.__getitem__(index)

            '''process seg'''
            eval_dict = {}
            with torch.no_grad():
                # logs = {}
                # data_inst = self.process_data_dict(data_inst)
                # Process data dictionary
                batch_data = data_seg
                if len(batch_data) == 0:
                    continue

                '''generate gt'''
                if isinstance(batch_data, list):
                    # the last one is the complete one
                    data_inst = self.process_data_dict(batch_data[-1])
                else:
                    data_inst = self.process_data_dict(batch_data)

                scan_id = data_inst['scan_id']
                graphDrawer = DrawSceneGraph(
                    scan_id, self.node_cls_names, self.edge_cls_names, debug=True)
                nodes_w = defaultdict(int)
                edges_w = defaultdict(int)
                nodes_pds_all = dict()
                edges_pds_all = dict()

                def fuse(old: dict, w_old: dict, new: dict):
                    for k, v in new.items():
                        if k in old:
                            old[k] = (old[k]*w_old[k]+new[k]) / (w_old[k]+1)
                            w_old[k] += 1
                        else:
                            old[k] = new[k]
                            w_old[k] = 1
                    return old, w_old

                def process(data):
                    data = self.process_data_dict(data)
                    # Shortcuts
                    scan_id = data['scan_id']
                    # gt_cls = data['gt_cls']
                    # gt_rel = data['gt_rel']
                    mask2seg = data['mask2instance']
                    node_edges_ori = data['node_edges']
                    data['node_edges'] = data['node_edges'].t().contiguous()
                    # seg2inst = data['seg2inst']

                    # check input valid
                    if node_edges_ori.ndim == 1:
                        return {}, {}, -1

                    ''' make forward pass through the network '''
                    tick = time.time()
                    node_cls, edge_cls = self.model(**data)
                    tock = time.time()

                    '''collect predictions on inst and edge pair'''
                    node_pds = dict()
                    edge_pds = dict()

                    '''merge prediction from seg to instance (in case of "same part")'''
                    # inst2masks = defaultdict(list)
                    mask2seg = merge_batch_mask2inst(mask2seg)
                    tmp_dict = defaultdict(list)
                    for mask, seg in mask2seg.items():
                        # inst = seg2inst[seg]
                        # inst2masks[inst].append(mask)

                        tmp_dict[seg].append(node_cls[mask])
                    for seg, l in tmp_dict.items():
                        if seg in node_pds:
                            raise RuntimeError()
                        pd = torch.stack(l, dim=0)
                        pd = torch.softmax(pd, dim=1).mean(dim=0)
                        node_pds[seg] = pd

                    tmp_dict = defaultdict(list)
                    for idx in range(len(node_edges_ori)):
                        src_idx, tgt_idx = data['node_edges'][0, idx].item(
                        ), data['node_edges'][1, idx].item()
                        seg_src, seg_tgt = mask2seg[src_idx], mask2seg[tgt_idx]
                        # inst_src,inst_tgt = seg2inst[seg_src],seg2inst[seg_tgt]
                        key = (seg_src, seg_tgt)

                        tmp_dict[key].append(edge_cls[idx])

                    for key, l in tmp_dict.items():
                        if key in edge_pds:
                            raise RuntimeError()
                        pd = torch.stack(l, dim=0)
                        pd = torch.softmax(pd, dim=1).mean(0)
                        edge_pds[key] = pd
                        # src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                        # inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))

                    return node_pds, edge_pds, tock-tick

                if isinstance(batch_data, list):
                    for idx, data in enumerate(batch_data):
                        fid = data['fid']
                        print(idx)
                        node_pds, edge_pds, pt = process(data)
                        if pt > 0:
                            acc_time += pt
                            timer_counter += 1

                            fuse(nodes_pds_all, nodes_w, node_pds)
                            fuse(edges_pds_all, edges_w, edge_pds)

                            inst_mask2instance = data_inst['mask2instance']
                            # data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                            gts = None
                            # gts = dict()
                            # gt_nodes = gts['nodes'] = dict()
                            # gt_edges = gts['edges'] = dict()
                            # for mid in range(data_inst['gt_cls'].shape[0]):
                            #     idx = inst_mask2instance[mid]
                            #     gt_nodes[idx] = data_inst['gt_cls'][mid]
                            # for idx in range(len(data_inst['gt_rel'])):
                            #     src_idx, tgt_idx = data_inst['node_edges'][idx,0].item(),data_inst['node_edges'][idx,1].item()
                            #     src_inst_idx, tgt_inst_idx = inst_mask2instance[src_idx], inst_mask2instance[tgt_idx]
                            #     gt_edges[(src_inst_idx, tgt_inst_idx)] = data_inst['gt_rel'][idx]

                            g = graphDrawer.draw({'nodes': nodes_pds_all, 'edges': edges_pds_all},
                                                 gts)

                            # merged_node_cls, merged_node_cls_gt,  merged_edge_cls, \
                            #     merged_edge_cls_gt, merged_mask2instance, merged_node_edges  = \
                            #            merge_pred_with_gt(data_inst,
                            #            node_cls_names,edge_cls_names,
                            #            nodes_pds_all, edges_pds_all,
                            #            inst_valid_node_cls_indices,seg_valid_node_cls_indices,
                            #            inst_valid_edge_cls_indices,seg_valid_edge_cls_indices,
                            #            noneidx_node_cls,noneidx_edge_cls,
                            #            ignore_missing,self.cfg.DEVICE)

                            # eval_tool.add([scan_id],
                            #                       merged_node_cls,
                            #                       merged_node_cls_gt,
                            #                       merged_edge_cls,
                            #                       merged_edge_cls_gt,
                            #                       [merged_mask2instance],
                            #                       merged_node_edges)

                            # pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                            # gts = process_gt(**eval_tool.predictions[scan_id]['gt'])

                            # g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN',
                            #         pd_only=False, gt_only=False)

                            g.render(os.path.join(
                                self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'), view=True)
                # else:
                #     data = batch_data
                #     fid = data['fid']
                #     node_pds, edge_pds, pt = process(data)
                #     if pt>0:
                #         acc_time += pt
                #         timer_counter+=1

                #         fuse(nodes_pds_all,nodes_w,node_pds)
                #         fuse(edges_pds_all,edges_w,edge_pds)

                #         eval_tool.add([scan_id],
                #                                       merged_node_cls,
                #                                       merged_node_cls_gt,
                #                                       merged_edge_cls,
                #                                       merged_edge_cls_gt,
                #                                       [merged_mask2instance],
                #                                       merged_node_edges)

                #         pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                #         gts = process_gt(**eval_tool.predictions[scan_id]['gt'])

                #         g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN',
                #                 pd_only=False, gt_only=False)

                #         g.render(os.path.join(self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'),view=True)
            # if index > 10: break
            break
        print('time:', acc_time, timer_counter, acc_time/timer_counter)

        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool