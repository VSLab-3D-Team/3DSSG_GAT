#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
#
import math
import importlib
import torch
from .network_util import build_mlp, Gen_Index, Aggre_Index, MLP
from .networks_base import BaseNetwork
import inspect
from collections import OrderedDict
import os
from codeLib.utils import onnx
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
import torch.nn as nn
from typing import Optional
from copy import deepcopy
from torch_scatter import scatter
from codeLib.common import filter_args_create
import ssg
import torch.nn.functional as F

def build_mlp(dims, do_bn=True, on_last=True):
    """Helper function to build an MLP with specified dimensions."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or on_last:
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(dims[i + 1]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

class MLP(torch.nn.Module):
    """Simple MLP class"""
    def __init__(self, dims):
        super().__init__()
        self.layers = build_mlp(dims)
    
    def forward(self, x):
        return self.layers(x)

def filter_args_create(cls, kwargs):
    """Helper to create a class instance with filtered arguments."""
    import inspect
    sig = inspect.signature(cls.__init__)
    filter_keys = [param.name for param in sig.parameters.values() if param.name != 'self']
    filtered_dict = {key: kwargs[key] for key in filter_keys if key in kwargs}
    return cls(**filtered_dict)

class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr='mean', with_bn=True):
        super().__init__(aggr=aggr)
        # print('============================')
        # print('aggr:',aggr)
        # print('============================')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge],
                             do_bn=with_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden, dim_hidden, dim_node], do_bn=with_bn)

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')

    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(
            edge_index, x=x, edge_feature=edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat([x_i, edge_feature, x_j], dim=1)
        x = self.nn1(x)  # .view(b,-1)
        new_x_i = x[:, :self.dim_hidden]
        new_e = x[:, self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:, (self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class TripletGCNModel(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            self.gconvs.append(TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature


class MessagePassing_IMP(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        # Attention layer
        self.subj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_feature, edge_index):
        node_msg, edge_msg = self.propagate(
            edge_index, x=x, edge_feature=edge_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j, edge_feature):
        '''Node'''
        message_pred_to_subj = self.subj_node_gate(
            torch.cat([x_i, edge_feature], dim=1)) * edge_feature  # n_rel x d
        message_pred_to_obj = self.obj_node_gate(
            torch.cat([x_j, edge_feature], dim=1)) * edge_feature  # n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)

        '''Edge'''
        message_subj_to_pred = self.subj_edge_gate(
            torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred = self.obj_edge_gate(
            torch.cat([x_j, edge_feature], 1)) * x_j  # nrel x d
        edge_message = (message_subj_to_pred+message_obj_to_pred)

        return [node_message, edge_message]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class MessagePassing_VGfM(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.subj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.geo_edge_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_feature, geo_feature, edge_index):
        node_msg, edge_msg = self.propagate(
            edge_index, x=x, edge_feature=edge_feature, geo_feature=geo_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j, edge_feature, geo_feature):
        message_pred_to_subj = self.subj_node_gate(
            torch.cat([x_i, edge_feature], dim=1)) * edge_feature  # n_rel x d
        message_pred_to_obj = self.obj_node_gate(
            torch.cat([x_j, edge_feature], dim=1)) * edge_feature  # n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)

        message_subj_to_pred = self.subj_edge_gate(
            torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred = self.obj_edge_gate(
            torch.cat([x_j, edge_feature], 1)) * x_j  # nrel x d
        message_geo = self.geo_edge_gate(
            torch.cat([geo_feature, edge_feature], 1)) * geo_feature
        edge_message = (message_subj_to_pred+message_obj_to_pred+message_geo)

        # x = torch.cat([x_i,edge_feature,x_j],dim=1)
        # x = self.nn1(x)#.view(b,-1)
        # new_x_i = x[:,:self.dim_hidden]
        # new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        # new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        # x = new_x_i+new_x_j
        return [node_message, edge_message]

    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim,
                       dim_size=dim_size, reduce=self.aggr)
        return x


class MessagePassing_Gate(MessagePassing):
    def __init__(self, dim_node, aggr='mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.temporal_gate = nn.Sequential(
            nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x_i = self.temporal_gate(torch.cat([x_i, x_j], dim=1)) * x_i
        return x_i


class TripletIMP(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr='mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node
        self.edge_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.msp_IMP = MessagePassing_IMP(dim_node=dim_node, aggr=aggr)
        self.reset_parameter()

    def reset_parameter(self):
        pass

    def forward(self, data):
        '''shortcut'''
        x = data['roi'].x
        edge_feature = data['edge2D'].x
        edge_index = data['roi', 'to', 'roi'].edge_index

        '''process'''
        x = self.node_gru(x)
        edge_feature = self.edge_gru(edge_feature)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msp_IMP(
                x=x, edge_feature=edge_feature, edge_index=edge_index)
            x = self.node_gru(node_msg, x)
            edge_feature = self.edge_gru(edge_msg, edge_feature)
        return x, edge_feature


class TripletVGfM(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr='mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node
        self.edge_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(
            input_size=self.dim_node, hidden_size=self.dim_node)

        self.msg_vgfm = MessagePassing_VGfM(dim_node=dim_node, aggr=aggr)
        self.msg_t_node = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)
        self.msg_t_edge = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)

        self.edge_encoder = ssg.models.edge_encoder.EdgeEncoder_VGfM()

        self.reset_parameter()

    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')

    def forward(self, data):
        '''shortcut'''
        x = data['roi'].x
        edge_feature = data['edge2D'].x
        edge_index = data['roi', 'to', 'roi'].edge_index
        geo_feature = data['roi'].desp
        temporal_node_graph = data['roi', 'temporal', 'roi'].edge_index
        temporal_edge_graph = data['edge2D', 'temporal', 'edge2D'].edge_index

        '''process'''
        x = self.node_gru(x)
        edge_feature = self.edge_gru(edge_feature)
        extended_geo_feature = self.edge_encoder(geo_feature, edge_index)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msg_vgfm(
                x=x, edge_feature=edge_feature, geo_feature=extended_geo_feature, edge_index=edge_index)
            if temporal_node_graph.shape[0] == 2:
                temporal_node_msg = self.msg_t_node(
                    x=x, edge_index=temporal_node_graph)
                node_msg += temporal_node_msg
            if temporal_edge_graph.shape[0] == 2:
                temporal_edge_msg = self.msg_t_edge(
                    x=edge_feature, edge_index=temporal_edge_graph)
                edge_msg += temporal_edge_msg
            x = self.node_gru(node_msg, x)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

        return x, edge_feature


class MSG_MV_DIRECT(MessagePassing):
    def __init__(self, aggr: str, use_res: bool = True):
        super().__init__(aggr=aggr,
                         flow='source_to_target')
        self.use_res = use_res

    def forward(self, node, images, edge_index):
        dummpy = (images, node)
        return self.propagate(edge_index, x=dummpy, node=node)

    def message(self, x_j):
        """

        Args:
            x_j (_type_): image_feature
        """
        return x_j

    def update(self, x, node):
        if self.use_res:
            x += node
        return x


class MSG_FAN(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                                 do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        '''update'''
        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index, x=x, edge_feature=edge_feature, x_ori=x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        '''
        x_i [N, D_N]
        x_j [N, D_N]
        '''
        num_node = x_i.size(0)

        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature, x_j], dim=1)
        triplet_feature = self.nn_edge(triplet_feature)

        '''FAN'''
        # proj
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        # est attention
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, triplet_feature, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x
    
class MSG_FAN_EDGE_UPDATE(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])
        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.combined_mlp = build_mlp([dim_node*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.cross_att1_q = build_mlp([dim_edge, dim_edge])
        self.cross_att1_k = build_mlp([dim_edge, dim_edge])
        self.cross_att1_v = build_mlp([dim_edge, dim_edge])
        
        self.cross_att2_q = build_mlp([dim_edge, dim_edge])
        self.cross_att2_k = build_mlp([dim_edge, dim_edge])
        self.cross_att2_v = build_mlp([dim_edge, dim_edge])
        
        self.edge_update_mlp = build_mlp([dim_edge*3, dim_edge*2, dim_edge], 
                                         do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index, x=x, edge_feature=edge_feature, x_ori=x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        num_node = x_i.size(0)
        
        combined = self.combined_mlp(torch.cat([x_i, x_j], dim=1))
        
        q1 = self.cross_att1_q(combined)
        k1 = self.cross_att1_k(edge_feature)
        v1 = self.cross_att1_v(edge_feature)
        
        att1_scores = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(q1.size(-1))
        att1_probs = torch.nn.functional.softmax(att1_scores, dim=-1)
        att1_probs = self.dropout(att1_probs)
        cross_att1_output = torch.matmul(att1_probs, v1)
        
        q2 = self.cross_att2_q(edge_feature)
        k2 = self.cross_att2_k(combined)
        v2 = self.cross_att2_v(combined)
        
        att2_scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(q2.size(-1))
        att2_probs = torch.nn.functional.softmax(att2_scores, dim=-1)
        att2_probs = self.dropout(att2_probs)
        cross_att2_output = torch.matmul(att2_probs, v2)
        
        updated_edge = self.edge_update_mlp(
            torch.cat([edge_feature, cross_att1_output, cross_att2_output], dim=1))
        
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, updated_edge, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x
    
class MSG_FAN_Masking(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 node_mask_prob: float = 0.3,
                 edge_mask_prob: float = 0.3,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        
        self.node_mask_prob = node_mask_prob
        self.edge_mask_prob = edge_mask_prob

        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])

        self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                                 do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        if edge_feature.shape[0] != edge_index.shape[1]:
            print(f"Warning: edge_feature shape {edge_feature.shape} doesn't match edge_index shape {edge_index.shape}")
            min_edges = min(edge_feature.shape[0], edge_index.shape[1])
            edge_feature = edge_feature[:min_edges]
            edge_index = edge_index[:, :min_edges]
        
        if self.training:
            return self.propagate_with_masking(edge_index, x=x, edge_feature=edge_feature, x_ori=x)
        else:
            return self.propagate_without_masking(edge_index, x=x, edge_feature=edge_feature, x_ori=x)
    
    def propagate_with_masking(self, edge_index, **kwargs):
        num_edges = edge_index.size(1)
        edge_mask = torch.rand(num_edges, device=edge_index.device) >= self.edge_mask_prob
        
        masked_edge_index = edge_index[:, edge_mask]
        
        x = kwargs.get('x')
        edge_feature = kwargs.get('edge_feature')
        x_ori = kwargs.get('x_ori')
        
        if edge_feature is not None:
            if edge_feature.shape[0] != num_edges:
                print(f"Warning: edge_feature shape[0] {edge_feature.shape[0]} != edge_index shape[1] {num_edges}")
                min_size = min(edge_feature.shape[0], num_edges)
                edge_feature = edge_feature[:min_size]
                edge_mask = edge_mask[:min_size]
            
            masked_edge_feature = edge_feature[edge_mask]
            kwargs['edge_feature'] = masked_edge_feature
        
        if edge_mask.sum() > 0:  # 남은 edge가 있는 경우
            src_nodes = masked_edge_index[0]
            dst_nodes = masked_edge_index[1]
            
            src_node_mask = torch.rand(src_nodes.size(0), device=edge_index.device) >= self.node_mask_prob
            dst_node_mask = torch.rand(dst_nodes.size(0), device=edge_index.device) >= self.node_mask_prob
            
            if x is not None:
                masked_x = x.clone()
                
                masked_src_nodes = src_nodes[~src_node_mask]
                masked_dst_nodes = dst_nodes[~dst_node_mask]
                
                masked_x[masked_src_nodes] = 0
                masked_x[masked_dst_nodes] = 0
                
                kwargs['x'] = masked_x
        
        result = super().propagate(masked_edge_index, **kwargs)
        
        return result[0], result[1], masked_edge_index, result[2]
    
    def propagate_without_masking(self, edge_index, **kwargs):
        result = super().propagate(edge_index, **kwargs)
        return result[0], result[1], result[2]

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        '''
        x_i [N, D_N]
        x_j [N, D_N]
        '''
        num_node = x_i.size(0)

        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature, x_j], dim=1)
        triplet_feature = self.nn_edge(triplet_feature)

        '''FAN'''
        # proj
        x_i = self.proj_q(x_i).view(
            num_node, self.dim_node_proj, self.num_head)  # [N,D,H]
        edge = self.proj_k(edge_feature).view(
            num_node, self.dim_edge_proj, self.num_head)  # [M,D,H]
        x_j = self.proj_v(x_j)
        # est attention
        att = self.nn_att(torch.cat([x_i, edge], dim=1))  # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j

        return [value, triplet_feature, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                            dim_size=dim_size, reduce=self.aggr)
        return inputs

    def update(self, x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori, x[0]], dim=1))
        return x
    
class BidirectionalEdgeLayer(MessagePassing):
    def __init__(self,
                 dim_node: int, dim_edge: int, dim_atten: int,
                 num_heads: int,
                 use_bn: bool,
                 aggr='max',
                 attn_dropout: float = 0.5,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.distance_mlp = build_mlp([3 + 1, dim_node//2, dim_node//4], do_bn=use_bn)
        
        self.mhsa = nn.MultiheadAttention(dim_node, num_heads, dropout=attn_dropout)
        
        self.edge_transform = build_mlp([dim_edge, dim_edge], do_bn=use_bn)
        
        self.update_node = build_mlp([dim_node + dim_edge, dim_node], do_bn=use_bn, on_last=False)
        
        self.update_edge = build_mlp([dim_node*2 + dim_edge*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(self, x, edge_feature, edge_index, node_pos=None):
        """
        x: 노드 특징 [num_nodes, dim_node]
        edge_feature: 에지 특징 [num_edges, dim_edge]
        edge_index: 에지 인덱스 [2, num_edges]
        node_pos: 노드의 3D 위치 [num_nodes, 3] (거리 기반 마스킹에 사용)
        """
        edge_dict = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_dict[(src, dst)] = i
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if (dst, src) in edge_dict:
                reverse_idx = edge_dict[(dst, src)]
                reverse_edge_feature[i] = edge_feature[reverse_idx]
        
        attn_mask = None
        if node_pos is not None:
            attn_mask = self._create_distance_mask(node_pos, edge_index)
        
        x_mhsa = x.unsqueeze(1)  # [num_nodes, 1, dim_node]
        x_updated, _ = self.mhsa(x_mhsa, x_mhsa, x_mhsa, attn_mask=attn_mask)
        x_updated = x_updated.squeeze(1)  # [num_nodes, dim_node]
        
        edge_updated, twinning_edge_attn = self.propagate(
            edge_index, 
            x=x, 
            edge_feature=edge_feature, 
            reverse_edge_feature=reverse_edge_feature
        )
        
        final_node_feature = self.update_node(torch.cat([x_updated, twinning_edge_attn], dim=1))
        
        return final_node_feature, edge_updated
    
    def _create_distance_mask(self, node_pos, edge_index):
        """
        노드 간 거리를 기반으로 attention 마스크 생성
        """
        num_nodes = node_pos.size(0)
        mask = torch.ones(num_nodes, num_nodes, device=node_pos.device) * float('-inf')
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            dist = torch.norm(node_pos[src] - node_pos[dst], p=2)
            
            weight = 1.0 / (dist + 1e-6)
            
            mask[src, dst] = weight
            mask[dst, src] = weight  # 양방향
        
        for i in range(num_nodes):
            mask[i, i] = 1.0
            
        return mask

    def message(self, x_i, x_j, edge_feature, reverse_edge_feature):
        """
        x_i: 소스 노드 특징 [num_edges, dim_node]
        x_j: 타겟 노드 특징 [num_edges, dim_node]
        edge_feature: 에지 특징 [num_edges, dim_edge]
        reverse_edge_feature: 역방향 에지 특징 [num_edges, dim_edge]
        """
        # e_{ij}^{l+1} = g_e([v_i^l, e_{ij}^l, e_{ji}^l, v_j^l])
        updated_edge = self.update_edge(
            torch.cat([x_i, edge_feature, reverse_edge_feature, x_j], dim=1)
        )
        
        transformed_edge = self.edge_transform(edge_feature)
        
        return updated_edge, transformed_edge

    def aggregate(self, inputs, index, dim_size=None):

        updated_edge, transformed_edge = inputs
        
        subj_edge_features = scatter(transformed_edge, index[0], dim=0, dim_size=dim_size, reduce=self.aggr)
        
        obj_edge_features = scatter(transformed_edge, index[1], dim=0, dim_size=dim_size, reduce=self.aggr)
        
        twinning_edge_attn = subj_edge_features + obj_edge_features
        
        return updated_edge, twinning_edge_attn

class JointGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.with_geo = kwargs['with_geo']
        self.num_layers = kwargs['num_layers']
        self.num_heads = kwargs['num_heads']
        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        drop_out_p = kwargs['drop_out']
        self.gconvs = torch.nn.ModuleList()

        # Get version
        args_jointgnn = kwargs['jointgnn']
        args_img_msg = kwargs[args_jointgnn['img_msg_method']]

        gnn_modules = importlib.import_module(
            'ssg.models.network_GNN').__dict__
        # jointGNNModel = gnn_modules['JointGNN_{}'.format(args_jointgnn['version'].lower())]
        img_model = gnn_modules[args_jointgnn['img_msg_method']]
        self.msg_img = filter_args_create(
            img_model, {**kwargs, **args_img_msg})

        # GRU
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        # gate
        if self.with_geo:
            self.geo_gate = nn.Sequential(
                nn.Linear(dim_node * 2, 1), nn.Sigmoid())

        self.drop_out = None
        if drop_out_p > 0:
            self.drop_out = torch.nn.Dropout(drop_out_p)

        # for _ in range(self.num_layers):
        #     self.gconvs.append(jointGNNModel(**kwargs))

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(
                MSG_FAN, {**kwargs, **kwargs['MSG_FAN']}))

    def forward(self, data):
        probs = list()
        node = data['node'].x
        if self.with_geo:
            geo_feature = data['geo_feature'].x
        # image = data['roi'].x
        edge = data['node', 'to', 'node'].x
        # spatial = data['node'].spatial if 'spatial' in data['node'] else None
        edge_index_node_2_node = data['node', 'to', 'node'].edge_index
        # edge_index_image_2_ndoe = data['roi','sees','node'].edge_index

        # TODO: use GRU?
        node = self.node_gru(node)
        edge = self.edge_gru(edge)
        for i in range(self.num_layers):
            gconv = self.gconvs[i]

            if self.with_geo:
                geo_msg = self.geo_gate(torch.cat(
                    (node, geo_feature), dim=1)) * torch.sigmoid(geo_feature)  # TODO:put the gate back
                # geo_msg = self.geo_gate(torch.cat((node,geo_feature),dim=1)) * geo_feature
                node += geo_msg

            # node, edge, prob = gconv(node,image,edge,edge_index_node_2_node,edge_index_image_2_ndoe)
            node_msg, edge_msg, prob = gconv(
                node, edge, edge_index_node_2_node)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_msg = torch.nn.functional.relu(node_msg)
                edge_msg = torch.nn.functional.relu(edge_msg)

                if self.drop_out:
                    node_msg = self.drop_out(node_msg)
                    edge_msg = self.drop_out(edge_msg)

            node = self.node_gru(node_msg, node)
            edge = self.edge_gru(edge_msg, edge)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node, edge, probs


class GraphEdgeAttenNetworkLayers(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs


class GraphEdgeAttenNetworkLayers_edge_update(torch.nn.Module):
    """ A sequence of scene graph convolution layers with modified edge update mechanism """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN_EDGE_UPDATE, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
                
        return node_feature, edge_feature, probs
    
class GraphEdgeAttenNetworkLayers_masking(torch.nn.Module):
    """ A sequence of scene graph convolution layers with node and edge masking """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        node_mask_prob = kwargs.get('node_mask_prob', 0.3)
        edge_mask_prob = kwargs.get('edge_mask_prob', 0.3)
        
        for _ in range(self.num_layers):
            self.gconvs.append(MSG_FAN_Masking(
                dim_node=kwargs['dim_node'],
                dim_edge=kwargs['dim_edge'],
                dim_atten=kwargs['dim_atten'],
                num_heads=kwargs['num_heads'],
                use_bn=kwargs['use_bn'],
                aggr=kwargs['aggr'],
                attn_dropout=kwargs.get('attn_dropout', 0.1),
                node_mask_prob=node_mask_prob,
                edge_mask_prob=edge_mask_prob,
                flow=kwargs.get('flow', 'target_to_source')
            ))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        original_edge_indices = edges_indices.clone()
        original_edge_feature = edge_feature.clone()
        
        if self.training:
            data['_original_edge_indices'] = original_edge_indices
            data['_original_edge_feature'] = original_edge_feature
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            
            if self.training:
                # 학습
                node_feature, edge_feature, edges_indices, prob = gconv(
                    node_feature, edge_feature, edges_indices)
            else:
                # 추론
                node_feature, edge_feature, prob = gconv.propagate_without_masking(
                    edges_indices, x=node_feature, edge_feature=edge_feature, x_ori=node_feature)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        
        return node_feature, edge_feature, probs
    
class BidirectionalEdgeGraphNetwork(torch.nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']
        self.dim_node = kwargs['dim_node']
        self.dim_edge = kwargs['dim_edge']
        self.dim_atten = kwargs['dim_atten']
        self.num_heads = kwargs['num_heads']
        self.use_bn = kwargs.get('use_bn', True)
        self.aggr = kwargs.get('aggr', 'max')
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        
        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        
        for _ in range(self.num_layers):
            self.gconvs.append(BidirectionalEdgeLayer(
                dim_node=self.dim_node,
                dim_edge=self.dim_edge,
                dim_atten=self.dim_atten,
                num_heads=self.num_heads,
                use_bn=self.use_bn,
                aggr=self.aggr,
                attn_dropout=self.attn_dropout
            ))
    
    def forward(self, data):

        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        node_pos = data['node'].pos if 'pos' in data['node'] else None
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            
            node_feature, edge_feature = gconv(
                node_feature, edge_feature, edges_indices, node_pos
            )
            
            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = F.relu(node_feature)
                edge_feature = F.relu(edge_feature)
                
                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)
        
        return node_feature, edge_feature, None  # None은 probs 자리


class FAN_GRU(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index

        # Init GRU
        node_feature = self.node_gru(node_feature)
        edge_feature = self.edge_gru(edge_feature)

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_msg, edge_msg, prob = gconv(
                node_feature, edge_feature, edges_indices)

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_msg = torch.nn.functional.relu(node_msg)
                edge_msg = torch.nn.functional.relu(edge_msg)

                if self.drop_out:
                    node_msg = self.drop_out(node_msg)
                    edge_msg = self.drop_out(edge_msg)

            node_feature = self.node_gru(node_msg, node_feature)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs


class FAN_GRU_2(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        dim_node = kwargs['dim_node']
        dim_edge = kwargs['dim_edge']
        self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.edge_gru = nn.GRUCell(input_size=dim_edge, hidden_size=dim_edge)

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index

        # Init GRU
        node_feature = self.node_gru(node_feature)
        edge_feature = self.edge_gru(edge_feature)

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_msg, edge_msg, prob = gconv(
                node_feature, edge_feature, edges_indices)

            # if i < (self.num_layers-1) or self.num_layers==1:
            #     node_msg = torch.nn.functional.relu(node_msg)
            #     edge_msg = torch.nn.functional.relu(edge_msg)

            #     if self.drop_out:
            #         node_msg = self.drop_out(node_msg)
            #         edge_msg = self.drop_out(edge_msg)

            node_feature = self.node_gru(node_msg, node_feature)
            edge_feature = self.edge_gru(edge_msg, edge_feature)

            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs
