import torch
import torch.nn as nn
from ssg.models.encoder import point_encoder_dict
from torch_geometric.nn.conv import MessagePassing
import os
from codeLib.utils import onnx
import inspect
from collections import OrderedDict

import torch.nn.functional as F


class EdgeDescriptor_8(MessagePassing):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, flow="target_to_source"):
        '''
        about target_to_source or source_to_target. check https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        '''
        super().__init__(flow=flow)

    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(
            self.__user_args__, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature

    def __len__(self):
        return 8

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5:dims, 6:volume, 7:length
        # to
        # 0-2: offset centroid, 3-5: dim log ratio, 96 volume log ratio, 7: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:, 0:3] = x_i[:, 0:3]-x_j[:, 0:3]
        # dim log ratio
        edge_feature[:, 3:6] = torch.log(x_i[:, 3:6] / x_j[:, 3:6])
        # volume log ratio
        edge_feature[:, 6] = torch.log(x_i[:, 6] / x_j[:, 6])
        # length log ratio
        edge_feature[:, 7] = torch.log(x_i[:, 7] / x_j[:, 7])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)


class EdgeDescriptor_plane(MessagePassing):
    """ A sequence of scene graph convolution layers  """

    def __init__(self, flow="target_to_source"):
        '''
        about target_to_source or source_to_target. check https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        '''
        super().__init__(flow=flow)
        # self.dim=18
        self.dim = 18-6

    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(
            self.__user_args__, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature

    def __len__(self):
        return self.dim

    def point_2_plane_dist(self, pts, ori, normal):
        v = pts-ori
        d = v @ normal
        return d

    def norm(self, x):
        return x / x.norm(dim=1).view(-1, 1)

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5:dims, 6:volume, 7:length
        # to
        # 0-2: offset centroid, 3-5: dim log ratio, 6 volume log ratio, 7: length log ratio

        # 0-2:center, 3-5:dims,6:volume,7:length,
        # 8-10: x_max,11-13:x_min,14-16:y_max,17-19:y_min,20-22:z_max,23-25:z_min
        batch = x_i.shape[0]

        # Find gravity aligned plane
        center_i = x_i[:, :3]
        center_j = x_j[:, :3]
        center = (center_i+center_j)*0.5

        # center_o = center.clone()
        # center_o[:,2] -= 10
        # v_oi = center_i - center_o
        # v_oj = center_j - center_o
        # normal = self.norm(torch.cross(v_oi,v_oj,dim=1))

        '''normals'''
        normal_up = torch.zeros(batch, 3).to(x_i.device)
        normal_up[:, 2] = 1

        v_ij = self.norm(center_j-center_i)
        normal_right = torch.cross(v_ij, normal_up, dim=1)
        normal_front = torch.cross(
            normal_up, normal_right, dim=1)  # assume gravity align

        '''calculate plane distances'''
        pts_i = x_i[:, 8:].view(batch, 6, 3)
        pts_j = x_j[:, 8:].view(batch, 6, 3)
        center = center.view(batch, 1, 3)
        d_z_i = self.point_2_plane_dist(
            pts_i, center, normal_up.view(batch, 3, 1))
        d_z_j = self.point_2_plane_dist(
            pts_j, center, normal_up.view(batch, 3, 1))
        d_y_i = self.point_2_plane_dist(
            pts_i, center, normal_right.view(batch, 3, 1))
        d_y_j = self.point_2_plane_dist(
            pts_j, center, normal_right.view(batch, 3, 1))
        d_x_i = self.point_2_plane_dist(
            pts_i, center, normal_front.view(batch, 3, 1))
        d_x_j = self.point_2_plane_dist(
            pts_j, center, normal_front.view(batch, 3, 1))

        ''''''
        # [batch]
        def get_max_min(dx, dy, dz):
            return dx.min(1)[0], dx.max(1)[0], dy.min(1)[0], dy.max(1)[0], dz.min(1)[0], dz.max(1)[0]
        d_i = torch.cat(get_max_min(d_x_i, d_y_i, d_z_i), dim=1)
        d_j = torch.cat(get_max_min(d_x_j, d_y_j, d_z_j), dim=1)

        # d_ij = torch.cat([d_i,d_j],dim=1)

        edge_feature = torch.zeros([batch, self.dim]).to(x_i.device)
        # centroid offset
        edge_feature[:, 0:3] = x_i[:, 0:3]-x_j[:, 0:3]
        # dim log ratio
        edge_feature[:, 3:6] = torch.log(x_i[:, 3:6] / x_j[:, 3:6])

        edge_feature[:, 6:] = (d_i.abs()/(d_j.abs()+1e-12)).log()  # d_ij

        # # volume log ratio
        # edge_feature[:,6] = torch.log( x_i[:,6] / x_j[:,6])
        # # length log ratio
        # edge_feature[:,7] = torch.log( x_i[:,7] / x_j[:,7])
        # # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)


class EdgeDescriptor_VGfM(MessagePassing):
    def __init__(self, flow="target_to_source"):
        '''
        https://github.com/paulgay/VGfM/blob/52c6bdbb14623c355af353c244752a6bed16f540/lib/networks/models.py#L555
        '''
        super().__init__(flow=flow)
        self.dim = 16

    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(
            self.__user_args__, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature

    def __len__(self):
        return self.dim

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5:dims, 6:volume, 7:length
        # to
        # 0-2: offset centroid, 3-5: dim log ratio, 96 volume log ratio, 7: length log ratio

        # 0-2:center, 3-5:dims,6:volume,7:length,
        # 8-10: x_max,11-13:x_min,14-16:y_max,17-19:y_min,20-22:z_max,23-25:z_min
        batch = x_i.shape[0]

        # Find gravity aligned plane
        center_i = x_i[:, :3]
        center_j = x_j[:, :3]

        # CENTER
        center = (center_i+center_j)*0.5
        # DISTANCE
        distance = (center_i-center_j).norm()
        # AXIS LENGTH
        t1 = center_i-center
        t2 = center_j-center

        edge_feature = torch.zeros([batch, self.dim]).to(x_i.device)
        edge_feature[:, :3] = center
        edge_feature[:, 3:6] = t1
        edge_feature[:, 6:9] = t2
        edge_feature[:, 9:12] = x_i[:, 3:6]
        edge_feature[:, 12:15] = x_j[:, 3:6]
        edge_feature[:, 15] = distance
        return edge_feature


class EdgeEncoder_2DSSG(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.edge_descriptor = EdgeDescriptor_8()
        self.encoder = point_encoder_dict['pointnet'](point_size=len(self.edge_descriptor),
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.edge_encoder.with_bn)

    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature


class EdgeEncoder_2DSSG_1(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.name = 'edgeEncoderPlane'
        self.edge_descriptor = EdgeDescriptor_plane()
        self.encoder = point_encoder_dict['pointnet'](point_size=len(self.edge_descriptor),
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.edge_encoder.with_bn)

    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature

    def trace(self, pth='./tmp', name_prefix=''):
        params = inspect.signature(self.forward).parameters
        params = OrderedDict(params)
        names_i = [name for name in params.keys() if name != 'args']
        names_o = ['y']

        n_node = 2
        n_edge = 4
        x = torch.rand(n_node, 26)
        edge_index = torch.randint(0, n_node-1, [2, n_edge])
        edge_index[0] = torch.zeros([n_edge])
        edge_index[1] = torch.ones([n_edge])

        self.eval()

        # check input can be run
        self(x, edge_index)

        # names_i = ['x']
        name = name_prefix+'_'+self.name
        input_ = (x, edge_index)
        dynamic_axes = {names_i[0]: {0: 'n_node'}, names_i[1]: {1: 'n_edges'}}
        onnx.export(self, input_, os.path.join(pth, name),
                    input_names=names_i, output_names=names_o,
                    dynamic_axes=dynamic_axes)

        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input'] = names_i
        names['model_'+name]['output'] = names_o
        return names


class EdgeEncoder_VGfM(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.edge_descriptor = EdgeDescriptor_VGfM()
        self.encoder = torch.nn.Sequential(
            nn.Linear(len(self.edge_descriptor), 100),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(100, 512),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature


class EdgeDescriptor_SGFN(MessagePassing):  # TODO: move to model
    """ A sequence of scene graph convolution layers  """

    def __init__(self, flow="source_to_target"):
        super().__init__(flow=flow)

    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(
            self.__user_args__, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:, 0:3] = x_i[:, 0:3]-x_j[:, 0:3]
        # std  offset
        edge_feature[:, 3:6] = x_i[:, 3:6]-x_j[:, 3:6]
        # dim log ratio
        edge_feature[:, 6:9] = torch.log(x_i[:, 6:9] / x_j[:, 6:9])
        # volume log ratio
        edge_feature[:, 9] = torch.log(x_i[:, 9] / x_j[:, 9])
        # length log ratio
        edge_feature[:, 10] = torch.log(x_i[:, 10] / x_j[:, 10])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)


class EdgeEncoder_SGFN(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.edge_descriptor = EdgeDescriptor_SGFN()
        self.encoder = point_encoder_dict['pointnet'](point_size=11,
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.edge_encoder.with_bn)

    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature


class EdgeEncoder_SGPN(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self._device = device

        dim_pts = 3
        if cfg.model.use_rgb:
            dim_pts += 3
        if cfg.model.use_normal:
            dim_pts += 3
        self.dim_pts = dim_pts+1  # mask

        self.model = point_encoder_dict['pointnet'](point_size=self.dim_pts,
                                                    out_size=cfg.model.edge_feature_dim,
                                                    batch_norm=cfg.model.node_encoder.with_bn)

    def forward(self, x):
        return self.model(x)

# class EdgeEncoder_SGPN(nn.Module):
#     def __init__(self, cfg, device):
#         super().__init__()
#         self._device = device
#         self.cfg = cfg

#         # 입력 차원 설정
#         dim_pts = 3
#         if cfg.model.use_rgb:
#             dim_pts += 3
#         if cfg.model.use_normal:
#             dim_pts += 3
#         self.dim_pts = dim_pts + 1  # 마스크 포함
        
#         # HDSN의 계층적 구조
#         # 인스턴스 스테이지: 더 세밀한 특징 추출
#         self.instance_stage = nn.Sequential(
#             nn.Conv1d(self.dim_pts, 64, 1),
#             nn.BatchNorm1d(64, track_running_stats=True, momentum=0.01),
#             nn.ReLU(),
#             nn.Conv1d(64, 128, 1),
#             nn.BatchNorm1d(128, track_running_stats=True, momentum=0.01),
#             nn.ReLU(),
#             nn.Conv1d(128, 256, 1),
#             nn.BatchNorm1d(256, track_running_stats=True, momentum=0.01),
#             nn.ReLU()
#         )
        
#         # 관계 스테이지: 더 넓은 문맥 특징 추출
#         self.relation_stage = nn.Sequential(
#             nn.Conv1d(256, 512, 1),
#             nn.BatchNorm1d(512, track_running_stats=True, momentum=0.01),
#             nn.ReLU(),
#             nn.Conv1d(512, cfg.model.edge_feature_dim, 1),
#             nn.BatchNorm1d(cfg.model.edge_feature_dim, track_running_stats=True, momentum=0.01),
#             nn.ReLU()
#         )
        
#         # 계층적 다운샘플링 적용 (인스턴스 스테이지 16배, 관계 스테이지 4배)
#         self.instance_downsample = 16
#         self.relation_downsample = 4
        
#         # 최종 특징 통합 (배치 크기 1 대응을 위해 BatchNorm 대신 LayerNorm 사용)
#         self.global_feat = nn.Sequential(
#             nn.Linear(cfg.model.edge_feature_dim, cfg.model.edge_feature_dim),
#             nn.LayerNorm(cfg.model.edge_feature_dim),  # BatchNorm1d 대신 LayerNorm 사용
#             nn.ReLU()
#         )

#     def forward(self, x):
#         """
#         x: 입력 포인트 클라우드 (형태 자동 감지)
#         """
#         # 입력 텐서 형태 확인 및 조정
#         if len(x.shape) == 3:
#             if x.shape[1] == self.dim_pts and x.shape[2] > self.dim_pts:
#                 # 이미 [B, C, N] 형태인 경우
#                 batch_size, channels, num_points = x.shape
#             elif x.shape[2] == self.dim_pts and x.shape[1] > self.dim_pts:
#                 # [B, N, C] 형태인 경우 -> [B, C, N]으로 변환
#                 x = x.transpose(1, 2).contiguous()
#                 batch_size, channels, num_points = x.shape
#             else:
#                 # 차원 불명확한 경우
#                 raise ValueError(f"입력 텐서 형태가 예상과 다릅니다: {x.shape}")
#         else:
#             raise ValueError(f"입력 텐서는 3차원이어야 합니다: {x.shape}")
        
#         # 배치 크기가 1인 경우 평가 모드로 전환
#         if batch_size == 1:
#             # 배치 크기 1일 때 BatchNorm 문제 방지
#             was_training = self.training
#             self.eval()  # 평가 모드로 전환
            
#             # 인스턴스 스테이지 통과
#             instance_features = self.instance_stage(x)  # [B, 256, N]
            
#             # 인스턴스 스테이지 다운샘플링 (16배)
#             if num_points >= self.instance_downsample:
#                 instance_points = num_points // self.instance_downsample
#                 instance_features = self.downsample_features(instance_features, instance_points)  # [B, 256, N/16]
            
#             # 관계 스테이지 통과
#             relation_features = self.relation_stage(instance_features)  # [B, edge_feature_dim, N/16]
            
#             # 관계 스테이지 다운샘플링 (4배)
#             if num_points >= (self.instance_downsample * self.relation_downsample):
#                 relation_points = num_points // (self.instance_downsample * self.relation_downsample)
#                 relation_features = self.downsample_features(relation_features, relation_points)  # [B, edge_feature_dim, N/64]
            
#             # 원래 모드로 복원
#             if was_training:
#                 self.train()
#         else:
#             # 배치 크기가 1보다 큰 경우 정상 처리
#             # 인스턴스 스테이지 통과
#             instance_features = self.instance_stage(x)  # [B, 256, N]
            
#             # 인스턴스 스테이지 다운샘플링 (16배)
#             if num_points >= self.instance_downsample:
#                 instance_points = num_points // self.instance_downsample
#                 instance_features = self.downsample_features(instance_features, instance_points)  # [B, 256, N/16]
            
#             # 관계 스테이지 통과
#             relation_features = self.relation_stage(instance_features)  # [B, edge_feature_dim, N/16]
            
#             # 관계 스테이지 다운샘플링 (4배)
#             if num_points >= (self.instance_downsample * self.relation_downsample):
#                 relation_points = num_points // (self.instance_downsample * self.relation_downsample)
#                 relation_features = self.downsample_features(relation_features, relation_points)  # [B, edge_feature_dim, N/64]
        
#         # 전역 특징 추출 (최대 풀링)
#         global_feature = torch.max(relation_features, dim=2)[0]  # [B, edge_feature_dim]
#         global_feature = self.global_feat(global_feature)
        
#         return global_feature
    
#     def downsample_features(self, features, target_points):
#         """
#         특징 맵 다운샘플링 (단순 스트라이드 풀링)
#         features: [B, C, N] 형태의 특징 맵
#         target_points: 목표 포인트 수
#         """
#         batch_size, channels, num_points = features.shape
        
#         # 최소 1개 포인트 보장
#         target_points = max(1, target_points)
        
#         # 현재 포인트 수가 목표보다 적으면 그대로 반환
#         if num_points <= target_points:
#             return features
        
#         # 단순 스트라이드 풀링 (균등 간격 샘플링)
#         stride = num_points // target_points
#         indices = torch.arange(0, target_points, device=features.device) * stride
#         indices = torch.clamp(indices, 0, num_points - 1)  # 인덱스 범위 안전하게 조정
#         return features[:, :, indices]