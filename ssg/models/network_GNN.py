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

from typing import Optional, List, Dict
import clip

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
        x = kwargs.get('x')
        edge_feature = kwargs.get('edge_feature')
        x_ori = kwargs.get('x_ori')
        
        num_nodes = x.size(0)
        node_mask = torch.rand(num_nodes, device=edge_index.device) >= self.node_mask_prob
        
        masked_x = x.clone()
        masked_x[~node_mask] = 0 
        kwargs['x'] = masked_x
        
        result = super().propagate(edge_index, **kwargs)
        
        return result[0], result[1], edge_index, result[2]
    
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
                 use_bn: bool = False,
                 aggr='max',
                 attn_dropout: float = 0.3,
                 flow: str = 'target_to_source',
                 use_distance_mask: bool = False):
        super().__init__(aggr=aggr, flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.use_distance_mask = use_distance_mask

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_v = build_mlp([dim_node, dim_atten])
        
        self.proj_k = build_mlp([dim_edge, dim_edge])
        
        self.distance_mlp = build_mlp([4, 32, 1], do_bn=use_bn, on_last=False)
        
        self.nn_edge_update = build_mlp([dim_node*2+dim_edge*2, dim_node+dim_edge*2, dim_edge],
                                       do_bn=use_bn, on_last=False)
        
        self.edge_attention_mlp = build_mlp([dim_edge*2, dim_edge], do_bn=use_bn, on_last=False)
        
        self.nn_node_update = build_mlp([dim_node+dim_edge, dim_node+dim_edge, dim_node],
                                       do_bn=use_bn, on_last=False)
        
        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj, 
                          self.dim_node_proj+self.dim_edge_proj,
                          self.dim_edge_proj])
        
        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_feature, edge_index, node_positions=None):
        row, col = edge_index
        
        edge_id_mapping = {}
        for idx, (i, j) in enumerate(zip(row, col)):
            edge_id_mapping[(i.item(), j.item())] = idx
        
        reverse_edge_feature = torch.zeros_like(edge_feature)
        
        for idx, (i, j) in enumerate(zip(row, col)):
            if (j.item(), i.item()) in edge_id_mapping:
                reverse_idx = edge_id_mapping[(j.item(), i.item())]
                reverse_edge_feature[idx] = edge_feature[reverse_idx]
        
        distance_mask = None
        if self.use_distance_mask and node_positions is not None:
            distance_features = []
            for i, j in zip(row, col):
                i, j = i.item(), j.item()
                pos_i, pos_j = node_positions[i], node_positions[j]
                diff = pos_i - pos_j
                dist = torch.norm(diff, p=2)
                distance_features.append(torch.cat([diff, dist.unsqueeze(0)], dim=0))
            
            distance_features = torch.stack(distance_features)
            
            distance_mask = self.distance_mlp(distance_features)
            distance_mask = self.sigmoid(distance_mask).squeeze(-1)
        
        outgoing_edges = {}
        incoming_edges = {}
        
        for idx, (i, j) in enumerate(zip(row, col)):
            i, j = i.item(), j.item()
            if i not in outgoing_edges:
                outgoing_edges[i] = []
            outgoing_edges[i].append((idx, j))
            
            if j not in incoming_edges:
                incoming_edges[j] = []
            incoming_edges[j].append((idx, i))
        
        updated_node, updated_edge, prob = self.propagate(
            edge_index, 
            x=x, 
            edge_feature=edge_feature,
            reverse_edge_feature=reverse_edge_feature,
            distance_mask=distance_mask,
            x_ori=x
        )
        
        twin_edge_attention = torch.zeros((x.size(0), self.dim_edge*2), device=x.device)
        
        for node_id in range(x.size(0)):
            outgoing_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in outgoing_edges:
                for edge_idx, _ in outgoing_edges[node_id]:
                    outgoing_feature += updated_edge[edge_idx]
                if len(outgoing_edges[node_id]) > 0:
                    outgoing_feature /= len(outgoing_edges[node_id])
            
            incoming_feature = torch.zeros(self.dim_edge, device=x.device)
            if node_id in incoming_edges:
                for edge_idx, _ in incoming_edges[node_id]:
                    incoming_feature += updated_edge[edge_idx]
                if len(incoming_edges[node_id]) > 0:
                    incoming_feature /= len(incoming_edges[node_id])
            
            twin_edge_attention[node_id] = torch.cat([outgoing_feature, incoming_feature], dim=0)
        
        edge_attention = self.edge_attention_mlp(twin_edge_attention)
        edge_attention = self.sigmoid(edge_attention)
        
        node_feature_nonlinear = torch.nn.functional.relu(updated_node)  # f(v_i^l)
        final_node = node_feature_nonlinear * edge_attention  # ⊙ β(A_ε)
        
        return final_node, updated_edge, prob

    def message(self, x_i: Tensor, x_j: Tensor, 
                edge_feature: Tensor, reverse_edge_feature: Tensor,
                distance_mask: Optional[Tensor] = None) -> Tensor:
        '''
        x_i: 소스 노드 특징 [N, D_N]
        x_j: 타겟 노드 특징 [N, D_N]
        edge_feature: 정방향 에지 특징 [N, D_E]
        reverse_edge_feature: 역방향 에지 특징 [N, D_E]
        distance_mask: 거리 기반 마스킹 가중치 [N] (선택적)
        '''
        num_edge = x_i.size(0)
        
        updated_edge = self.nn_edge_update(
            torch.cat([x_i, edge_feature, reverse_edge_feature, x_j], dim=1)
        )
        
        x_i_proj = self.proj_q(x_i).view(
            num_edge, self.dim_node_proj, self.num_head)  # [N, D, H]
        edge_proj = self.proj_k(edge_feature).view(
            num_edge, self.dim_edge_proj, self.num_head)  # [N, D, H]
        x_j_val = self.proj_v(x_j)
        
        att = self.nn_att(torch.cat([x_i_proj, edge_proj], dim=1))  # [N, D, H]
        
        if distance_mask is not None:
            distance_mask = distance_mask.view(-1, 1, 1)
            att = att * distance_mask
        
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        
        weighted_value = prob.reshape_as(x_j_val) * x_j_val
        
        return [weighted_value, updated_edge, prob]

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        weighted_value, updated_edge, prob = inputs
        weighted_value = scatter(weighted_value, index, dim=self.node_dim,
                                dim_size=dim_size, reduce=self.aggr)
        return weighted_value, updated_edge, prob

    def update(self, inputs, x_ori):
        weighted_value, updated_edge, prob = inputs
        
        updated_node = self.nn_node_update(
            torch.cat([x_ori, weighted_value], dim=1)
        )
        
        return updated_node, updated_edge, prob
    
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
    """ A sequence of scene graph convolution layers with node masking only """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        node_mask_prob = kwargs.get('node_mask_prob', 0.3)
        
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
                flow=kwargs.get('flow', 'target_to_source')
            ))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            
            if self.training:
                # Training with node masking
                node_feature, edge_feature, edges_indices, prob = gconv(
                    node_feature, edge_feature, edges_indices)
            else:
                # Inference without masking
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
        self.use_distance_mask = kwargs.get('use_distance_mask', True)

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])

        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(BidirectionalEdgeLayer, kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x
        edges_indices = data['node', 'to', 'node'].edge_index
        
        node_positions = None
        if self.use_distance_mask and 'desp' in data['node']:
            node_positions = data['node'].desp[:, :3]
        
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(
                node_feature, edge_feature, edges_indices, node_positions
            )

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

""" Here is SuYeon's code zone """
class MSG_MMAN(MessagePassing):
    def __init__(self,
                 dim_node: int, 
                 dim_edge: int, 
                 dim_atten: int,
                 dim_clip: int = 512,
                 num_heads: int = 8,
                 use_bn: bool = True,
                 aggr: str = 'max',
                 attn_dropout: float = 0.5,
                 node_class_names: List[str] = None,
                 edge_class_names: List[str] = None,
                 use_text_attention: bool = True,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        
        self.node_class_names = node_class_names
        self.edge_class_names = edge_class_names
        self.use_text_attention = use_text_attention
        
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.dim_clip = dim_clip
        
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att_3d = MLP([self.dim_node_proj+self.dim_edge_proj, 
                           self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])
        
        self.nn_att_text = MLP([self.dim_node_proj+self.dim_edge_proj+self.dim_edge_proj, 
                               self.dim_node_proj+self.dim_edge_proj+self.dim_edge_proj,
                               self.dim_edge_proj])
        
        self.clip_encoder = CLIPTextEncoder()

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])
        self.proj_text = build_mlp([dim_clip, dim_edge], do_bn=use_bn)

        self.nn_edge = build_mlp([dim_node*2+dim_edge, (dim_node+dim_edge), dim_edge],
                                 do_bn=use_bn, on_last=False)

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        '''update'''
        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, data):

        x = data['node'].x
        node_class_indices = data['node'].y
        
        edge_feature = data['node', 'to', 'node'].x
        edge_class_indices = data['node', 'to', 'node'].y
        
        if edge_class_indices.dim() > 1: # multi-label (one-hot vector processing)
            edge_class_indices = edge_class_indices.argmax(dim=1)
        
        edge_index = data['node', 'to', 'node'].edge_index
        
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        
        text_embeddings = []
        for i in range(len(edge_class_indices)):
            src_idx = src_indices[i]
            dst_idx = dst_indices[i]
            
            src_class = self.node_class_names[node_class_indices[src_idx].item()]
            dst_class = self.node_class_names[node_class_indices[dst_idx].item()]
            edge_class = self.edge_class_names[edge_class_indices[i].item()]
            
            text_embedding = self.clip_encoder.get_text_embedding(src_class, edge_class, dst_class)
            text_embeddings.append(text_embedding)
        
        text_embeddings = torch.stack(text_embeddings).to(x.device)
        
        proj_text_embeddings = self.proj_text(text_embeddings.float())
        
        node_feature, edge_feature, probs_3d, probs_text, kl_divs = self.propagate(
            edge_index, x=x, edge_feature=edge_feature, 
            text_embedding=proj_text_embeddings, x_ori=x)
        
        return node_feature, edge_feature, probs_3d, probs_text, kl_divs
    
    def message(self, x_i, x_j, edge_feature, text_embedding):
        num_edges = x_i.size(0)
        
        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature, x_j], dim=1)
        triplet_feature = self.nn_edge(triplet_feature)
        
        q = self.proj_q(x_i).view(num_edges, self.dim_node_proj, self.num_head)  # ρ_i
        k = self.proj_k(edge_feature).view(num_edges, self.dim_edge_proj, self.num_head)  # ρ_ij
        v = self.proj_v(x_j)  # ρ_j
        
        t = text_embedding.view(num_edges, self.dim_edge_proj, self.num_head)  # ρ_text
        
        att_3d = self.nn_att_3d(torch.cat([q, k], dim=1))  # MLP_att(ρ_i || ρ_ij)
        prob_3d = torch.nn.functional.softmax(att_3d/self.temperature, dim=1)
        prob_3d = self.dropout(prob_3d)
        value_3d = prob_3d.reshape_as(v) * v
        
        if self.use_text_attention:
            att_text = self.nn_att_text(torch.cat([q, k, t], dim=1))  # MLP_att(ρ_i || ρ_ij || ρ_text)
            prob_text = torch.nn.functional.softmax(att_text/self.temperature, dim=1)
            prob_text = self.dropout(prob_text)
            value_text = prob_text.reshape_as(v) * v
        else:
            prob_text = prob_3d
            value_text = value_3d
        
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(prob_3d, dim=1),
            torch.nn.functional.softmax(prob_text, dim=1),
            reduction='none'
        ).sum(dim=1).mean()
        
        return [value_3d, triplet_feature, prob_3d, prob_text, kl_div]
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                           dim_size=dim_size, reduce=self.aggr)
        return inputs
    
    def update(self, inputs, x_ori):
        updated_node = self.update_node(torch.cat([x_ori, inputs[0]], dim=1))
        
        prob_3d = inputs[2]
        prob_text = inputs[3]
        kl_div = inputs[4]
        
        return updated_node, inputs[1], prob_3d, prob_text, kl_div
    

class MultiModalAttenNetworkLayers(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
            
        self.node_class_names = kwargs.get('node_class_names', [])
        self.edge_class_names = kwargs.get('edge_class_names', [])

        for _ in range(self.num_layers):
            self.gconvs.append(MSG_MMAN(
                dim_node=kwargs['dim_node'],
                dim_edge=kwargs['dim_edge'],
                dim_atten=kwargs['dim_atten'],
                dim_clip=512,
                num_heads=kwargs['num_heads'],
                use_bn=kwargs['use_bn'],
                aggr=kwargs['aggr'],
                attn_dropout=kwargs.get('attn_dropout', 0.1),
                node_class_names=self.node_class_names,
                edge_class_names=self.edge_class_names,
                use_text_attention=True,
                flow=kwargs.get('flow', 'target_to_source')
            ))

    def forward(self, data):
        probs_3d_all = []
        probs_text_all = []
        kl_divs_all = []
        
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, probs_3d, probs_text, kl_div = gconv(data)
            
            data['node'].x = node_feature
            data['node', 'to', 'node'].x = edge_feature

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)
                
                data['node'].x = node_feature
                data['node', 'to', 'node'].x = edge_feature

            if probs_3d is not None:
                probs_3d_all.append(probs_3d.detach())
            
            if probs_text is not None:
                probs_text_all.append(probs_text.detach())
                
            if kl_div is not None:
                kl_divs_all.append(kl_div.item())
                
        return node_feature, edge_feature, probs_3d_all, probs_text_all, kl_divs_all
    
class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        self.text_template = "a {object1} is {relation} {object2}."
        
        self.text_embedding_cache = {}
        
    def get_text_embedding(self, subject_class, relation_class, object_class):
        cache_key = f"{subject_class}_{relation_class}_{object_class}"
        
        if cache_key in self.text_embedding_cache:
            return self.text_embedding_cache[cache_key]
        
        text = self.text_template.format(
            object1=subject_class, relation=relation_class, object2=object_class)
        
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            
        text_embedding = text_features[0] / text_features[0].norm()
        
        self.text_embedding_cache[cache_key] = text_embedding
        
        return text_embedding
    
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

        self.combined_mlp = build_mlp([dim_node*2, dim_edge], do_bn=use_bn, on_last=False)#
        
        self.cross_att1_q = build_mlp([dim_edge, dim_edge])#
        self.cross_att1_k = build_mlp([dim_edge, dim_edge])#
        self.cross_att1_v = build_mlp([dim_edge, dim_edge])#
        
        self.cross_att2_q = build_mlp([dim_edge, dim_edge])#
        self.cross_att2_k = build_mlp([dim_edge, dim_edge])#
        self.cross_att2_v = build_mlp([dim_edge, dim_edge])#
        
        self.edge_update_mlp = build_mlp([dim_edge*3, dim_edge*2, dim_edge], 
                                         do_bn=use_bn, on_last=False)#

        self.dropout = torch.nn.Dropout(
            attn_dropout) if attn_dropout > 0 else torch.nn.Identity()

        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index, x=x, edge_feature=edge_feature, x_ori=x)

    def message(self, x_i: Tensor, x_j: Tensor, edge_feature: Tensor) -> Tensor:
        num_node = x_i.size(0)
        
        combined = self.combined_mlp(torch.cat([x_i, x_j], dim=1))#
        
        q1 = self.cross_att1_q(combined)#
        k1 = self.cross_att1_k(edge_feature)#
        v1 = self.cross_att1_v(edge_feature)#
        
        att1_scores = torch.matmul(q1, k1.transpose(-2, -1)) / math.sqrt(q1.size(-1))#
        att1_probs = torch.nn.functional.softmax(att1_scores, dim=-1)#
        att1_probs = self.dropout(att1_probs)#
        cross_att1_output = torch.matmul(att1_probs, v1)#
        
        q2 = self.cross_att2_q(edge_feature)#
        k2 = self.cross_att2_k(combined)#
        v2 = self.cross_att2_v(combined)#
        
        att2_scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(q2.size(-1))#
        att2_probs = torch.nn.functional.softmax(att2_scores, dim=-1)#
        att2_probs = self.dropout(att2_probs)#
        cross_att2_output = torch.matmul(att2_probs, v2)#
        
        updated_edge = self.edge_update_mlp(
            torch.cat([edge_feature, cross_att1_output, cross_att2_output], dim=1))#
        
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
    
class GraphEdgeAttenNetworkLayers_Edge_Update(torch.nn.Module):
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
    
class MSG_MMAN_Edge_Update(MessagePassing):
    def __init__(self,
                 dim_node: int, 
                 dim_edge: int, 
                 dim_atten: int,
                 dim_clip: int = 512,
                 num_heads: int = 8,
                 use_bn: bool = True,
                 aggr: str = 'max',
                 attn_dropout: float = 0.5,
                 node_class_names: List[str] = None,
                 edge_class_names: List[str] = None,
                 use_text_attention: bool = True,
                 flow: str = 'target_to_source'):
        super().__init__(aggr=aggr, flow=flow)
        
        self.node_class_names = node_class_names
        self.edge_class_names = edge_class_names
        self.use_text_attention = use_text_attention
        
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_atten = dim_atten
        self.dim_clip = dim_clip
        
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)

        self.nn_att_3d = MLP([self.dim_node_proj+self.dim_edge_proj, 
                           self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])
        
        self.nn_att_text = MLP([self.dim_node_proj+self.dim_edge_proj+self.dim_edge_proj, 
                               self.dim_node_proj+self.dim_edge_proj+self.dim_edge_proj,
                               self.dim_edge_proj])
        
        self.clip_encoder = CLIPTextEncoder()

        self.proj_q = build_mlp([dim_node, dim_node])
        self.proj_k = build_mlp([dim_edge, dim_edge])
        self.proj_v = build_mlp([dim_node, dim_atten])
        self.proj_text = build_mlp([dim_clip, dim_edge], do_bn=use_bn)

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

        '''update'''
        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                                     do_bn=use_bn, on_last=False)

    def forward(self, data):

        x = data['node'].x
        node_class_indices = data['node'].y
        
        edge_feature = data['node', 'to', 'node'].x
        edge_class_indices = data['node', 'to', 'node'].y
        
        edge_index = data['node', 'to', 'node'].edge_index
        
        if edge_class_indices.dim() > 1: # multi-label (one-hot vector processing)
            edge_class_indices = edge_class_indices.argmax(dim=1)
        
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        
        text_embeddings = []

        for i in range(len(edge_class_indices)):
            src_idx = src_indices[i]
            dst_idx = dst_indices[i]
            
            src_class = self.node_class_names[node_class_indices[src_idx].item()]
            dst_class = self.node_class_names[node_class_indices[dst_idx].item()]
            edge_class = self.edge_class_names[edge_class_indices[i].item()]
            
            text_embedding = self.clip_encoder.get_text_embedding(src_class, edge_class, dst_class)
            text_embeddings.append(text_embedding)
        
        text_embeddings = torch.stack(text_embeddings).to(x.device)
        
        proj_text_embeddings = self.proj_text(text_embeddings.float())
        
        node_feature, edge_feature, probs_3d, probs_text, kl_divs = self.propagate(
            edge_index, x=x, edge_feature=edge_feature, 
            text_embedding=proj_text_embeddings, x_ori=x)
        
        return node_feature, edge_feature, probs_3d, probs_text, kl_divs
    
    def message(self, x_i, x_j, edge_feature, text_embedding):
        num_edges = x_i.size(0)
        
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
        
        q = self.proj_q(x_i).view(num_edges, self.dim_node_proj, self.num_head)  # ρ_i
        k = self.proj_k(edge_feature).view(num_edges, self.dim_edge_proj, self.num_head)  # ρ_ij
        v = self.proj_v(x_j)  # ρ_j
        
        t = text_embedding.view(num_edges, self.dim_edge_proj, self.num_head)  # ρ_text
        
        att_3d = self.nn_att_3d(torch.cat([q, k], dim=1))  # MLP_att(ρ_i || ρ_ij)
        prob_3d = torch.nn.functional.softmax(att_3d/self.temperature, dim=1)
        prob_3d = self.dropout(prob_3d)
        value_3d = prob_3d.reshape_as(v) * v
        
        if self.use_text_attention:
            att_text = self.nn_att_text(torch.cat([q, k, t], dim=1))  # MLP_att(ρ_i || ρ_ij || ρ_text)
            prob_text = torch.nn.functional.softmax(att_text/self.temperature, dim=1)
            prob_text = self.dropout(prob_text)
            value_text = prob_text.reshape_as(v) * v
        else:
            prob_text = prob_3d
            value_text = value_3d
        
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(prob_3d, dim=1),
            torch.nn.functional.softmax(prob_text, dim=1),
            reduction='none'
        ).sum(dim=1).mean()
        
        return [value_3d, updated_edge, prob_3d, prob_text, kl_div]
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        inputs[0] = scatter(inputs[0], index, dim=self.node_dim,
                           dim_size=dim_size, reduce=self.aggr)
        return inputs
    
    def update(self, inputs, x_ori):
        updated_node = self.update_node(torch.cat([x_ori, inputs[0]], dim=1))
        
        prob_3d = inputs[2]
        prob_text = inputs[3]
        kl_div = inputs[4]
        
        return updated_node, inputs[1], prob_3d, prob_text, kl_div
    

class MultiModalAttenNetworkLayers_Edge_Update(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']

        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
            
        self.node_class_names = kwargs.get('node_class_names', [])
        self.edge_class_names = kwargs.get('edge_class_names', [])

        for _ in range(self.num_layers):
            self.gconvs.append(MSG_MMAN_Edge_Update(
                dim_node=kwargs['dim_node'],
                dim_edge=kwargs['dim_edge'],
                dim_atten=kwargs['dim_atten'],
                dim_clip=512,
                num_heads=kwargs['num_heads'],
                use_bn=kwargs['use_bn'],
                aggr=kwargs['aggr'],
                attn_dropout=kwargs.get('attn_dropout', 0.1),
                node_class_names=self.node_class_names,
                edge_class_names=self.edge_class_names,
                use_text_attention=True,
                flow=kwargs.get('flow', 'target_to_source')
            ))

    def forward(self, data):
        probs_3d_all = []
        probs_text_all = []
        kl_divs_all = []
        
        node_feature = data['node'].x
        edge_feature = data['node', 'to', 'node'].x

        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, probs_3d, probs_text, kl_div = gconv(data)
            
            data['node'].x = node_feature
            data['node', 'to', 'node'].x = edge_feature

            if i < (self.num_layers-1) or self.num_layers == 1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)

                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)
                
                data['node'].x = node_feature
                data['node', 'to', 'node'].x = edge_feature

            if probs_3d is not None:
                probs_3d_all.append(probs_3d.detach())
            
            if probs_text is not None:
                probs_text_all.append(probs_text.detach())
                
            if kl_div is not None:
                kl_divs_all.append(kl_div.item())
                
        return node_feature, edge_feature, probs_3d_all, probs_text_all, kl_divs_all