#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import tensorly as tl
from tensorly.decomposition import tucker
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
import matplotlib.pyplot as plt
tl.set_backend('pytorch')

    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3,                 #  更新深度，默认为3，用于控制在锚点增长过程中递归的深度
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_denom = torch.empty(0)

        # new code
        self._mask = torch.empty(0)
        # end new
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()


    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    # new code
    @property
    def get_mask(self):
        return self._mask

    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)   # 随机打乱输入数据，避免后续处理出现排毒偏差
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size   # 这是体素化的核心代码，将输入的点云数据 data 进行体素化，体素大小为 voxel_size,保证每个体素内只保留一个点
        from mpl_toolkits.mplot3d import Axes3D

        # # 假设 data 是一个 (36099, 3) 的 numpy 数组，每一行是一个点 (x, y, z)
        # x, y, z = data[:, 0], data[:, 1], data[:, 2]
        #
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # 绘制点云
        # ax.scatter(x, y, z, c='blue', s=1, alpha=0.6)  # s 可调节点大小
        #
        # # 坐标轴与标题
        # ax.set_title('3D Point Cloud Visualization')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        #
        # plt.show()
        return data

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):      # 从点云数据（BasicPointCloud 类型）创建高斯模型，并初始化模型的各种参数
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]        # 从输入的点云 pcd 中每隔 self.ratio 个点取一个点
        max = points.max(0)[0]      # 计算点云的最大值
        min = points.min(0)[0]      # 计算点云的最小值

        if self.voxel_size <= 0:   # 如果类的 voxel_size 属性小于或等于0，则自动计算体素大小
            init_points = torch.tensor(points).float().cuda()      # 将点转换为 torch.tensor
            init_dist = distCUDA2(init_points).float().cuda()      # 计算每两点之间的距离
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))     # 计算距离的中位数
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()      # 删除中间变量并释放显存

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)         # 对点云进行体素化，并返回体素化后的点云
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()       # 创建融合后的点云
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()     # 创建偏移量矩阵， n*offsets*3，每个方向5个偏移量
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()    # 创建锚点特征矩阵，n*32
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 计算每两点之间的距离平方的平方根，得到每两点之间的距离。然后对距离取自然对数
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)  # 计算每两点之间的距离平方的平方根，得到每两点之间的距离。然后对距离取自然对数
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1    # 初始时，旋转矩阵的第0列被设置为1，表示初始旋转为单位四元数（即不旋转）

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))  # 初始化透明度为0.1
        # 将参数设置为可训练参数
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True)) # 前3维是offset的缩放系数，后3维表示neural-gs的cov的初值，对应论文公式8中的lv
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        # new code
        self._mask = nn.Parameter(torch.ones((self.get_anchor.shape[0], 1), device="cuda").requires_grad_(True))
        # end new


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_denom = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        # 初始化一个用于优化高斯模型参数的 Adam 优化器，通过创建一个参数组列表，指定了每个参数的学习率和名称，从而使得优化器可以针对不同的参数进行调整
        
        if self.use_feat_bank:
            l = [# 高斯参数
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # new code
                {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
                # end new

                # MLP参数
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # new code
                {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
                # end new

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                # new code
                {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},
                # end new

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    # 根据预设的策略动态调整偏移量的学习率，学习率衰减
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 在训练过程中收集一些梯度信息，用于指导锚点（anchor）的增长和调整
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_denom[anchor_visible_mask] += 1    # anchor_denom 用于记录每个锚点被访问的次数

        # update neural gaussian statis 更新神经高斯（Neural Gaussian）统计信息
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1) # 扩展维度，使得锚点可见性掩码与偏移量掩码维度一致
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask   # 记录哪些偏移量需要更新
        temp_mask = combined_mask.clone()   # 复制 combined_mask
        combined_mask[temp_mask] = update_filter   # 根据 temp_mask，将 update_filter 中的值赋给 combined_mask，进一步筛选出需要更新的偏移量
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # new code
        self._mask = optimizable_tensors["mask"]
        # end code

    
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)   # cur_threshold随着迭代次数增加而减小，这有助于在递归的过程中逐渐添加更细小的锚点
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick  用于随机挑选候选偏移量，概率随着迭代次数增加而减小，这样可以控制新增锚点的数量
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)
            # 计算所有偏移点的位置
            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            # 计算当前尺寸因子和网格坐标
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor    # 当前迭代中的体素大小
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()
            # 从所有偏移点中选择候选偏移点
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling 去重
            use_chunk = True    # 使用分块处理的方法来减少内存使用
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []  # 判断哪些新的坐标已经在现有锚点中,用于存储每个分块的去重结果
                for i in range(max_iters): # .all(-1)：检查每一对坐标是否完全相等（即在所有维度上都相等） .any(-1)：检查在当前分块中是否存在至少一对相等的坐标  .view(-1)：将结果转换为一维布尔数组
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates  # 中找出不重复的坐标（即 True 表示不重复）
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            # 创建新锚点的参数
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling) # new_scaling初始化为当前体素大小，然后取自然对数
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0
                # new_opacities初始化为0.1，然后通过inverse_sigmoid函数转换为对应的输入值
                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                # new_feat是从现有锚点特征中挑选出的特征，并通过scatter_max函数确保每个新的锚点具有唯一的特征
                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                # new code
                new_mask = torch.ones([candidate_anchor.shape[0], 1], dtype=torch.float, device="cuda") # 初值设置为1
                # end new
                # 更新锚点列表
                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "mask": new_mask,  # new code
                }
                

                temp_anchor_denom = torch.cat([self.anchor_denom, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_denom
                self.anchor_denom = temp_anchor_denom

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._mask = optimizable_tensors["mask"]  # new code


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors 添加锚点
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]    计算梯度的归一化值：在训练过程中累积的偏移量/梯度和访问次数
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1) # 计算梯度范数
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1) # 筛选出访问次数大于一定阈值的偏移量
        # 将筛选出的偏移量传入 anchor_growing 函数，用于增长锚点
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom 对于已经被选中用来增加新锚点的位置，重置访问次数和累积梯度，然后更新参数以匹配新的锚点数量
        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors 剪枝锚点
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_denom).squeeze(dim=1) # [N, 1]      筛选出透明度小于一定阈值的锚点
        anchors_mask = (self.anchor_denom > check_interval*success_threshold).squeeze(dim=1) # [N, 1]   筛选出访问次数大于一定阈值的锚点
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]   多次访问但透明度很低的锚点会被剪枝
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum  重置透明度累积值和最大半径
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_denom[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_denom = self.anchor_denom[~prune_mask]
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'): # split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    # new code
    def mask_prune(self, mask_threshold = 0.01):
        prune_mask = (torch.sigmoid(self._mask) < mask_threshold).squeeze(dim=1)
        self.prune_anchor(prune_mask)
        torch.cuda.empty_cache()

    # new code
    def decode_voxel_key(self, voxel_keys, max_indices, min_indices):
        if not torch.is_tensor(max_indices):
            max_indices = torch.tensor(max_indices, device=voxel_keys.device, dtype=voxel_keys.dtype)
        Y_max = max_indices[1]
        Z_max = max_indices[2]
        z = voxel_keys % Z_max
        y = (voxel_keys // Z_max) % Y_max
        x = voxel_keys // (Y_max * Z_max)
        coords = torch.stack([x+min_indices[0], y+min_indices[1], z+min_indices[2]], dim=0)
        return coords.T  # (N,3)

    # new code
    def build_statistics_tensor(self, voxel_size=2):
        # """构建密度张量"""
        # 确定体素尺寸
        voxel_indices = torch.floor(self.get_anchor / voxel_size).int()  # (N,3)
        # 将体素索引展平为一维索引
        min_indices = voxel_indices.min(dim=0, keepdim=True).values
        voxel_indices_shifted = voxel_indices - min_indices
        # 计算范围
        max_indices = voxel_indices_shifted.max(dim=0).values + 1
        voxel_keys = (voxel_indices_shifted[:, 0] * max_indices[1] * max_indices[2] +
                      voxel_indices_shifted[:, 1] * max_indices[2] +
                      voxel_indices_shifted[:, 2])
        # 获取唯一体素索引
        unique_voxel_keys, inverse_indices = torch.unique(voxel_keys, return_inverse=True)

        # 每个区块一个列表
        # voxel_to_anchors = [[] for _ in range(unique_voxel_keys.shape[0])]
        # for i in range(self.get_anchor.shape[0]):
        #     voxel_to_anchors[inverse_indices[i]].append(i)

        # 累加体素各类统计量
        num_voxels = unique_voxel_keys.shape[0]

        # 初始化体素累计张量
        voxel_density = torch.zeros(num_voxels, 1, device=self.get_anchor.device)
        voxel_opacity_sum = torch.zeros(num_voxels, 1, device=self.get_anchor.device)
        voxel_visit_sum = torch.zeros(num_voxels, 1, device=self.get_anchor.device)

        # scatter_add：将每个点的值加到对应的体素“槽”里
        voxel_density.scatter_add_(0, inverse_indices.unsqueeze(1), torch.ones_like(self.opacity_accum))
        voxel_opacity_sum.scatter_add_(0, inverse_indices.unsqueeze(1), self.opacity_accum)
        voxel_visit_sum.scatter_add_(0, inverse_indices.unsqueeze(1), self.anchor_denom)

        return max_indices, min_indices.squeeze(), unique_voxel_keys, inverse_indices, [voxel_density, voxel_opacity_sum, voxel_visit_sum]

    # new code
    def coarse_voxel_selection(self, itr, max_indices, unique_voxel_keys, inverse_indices, statistics_tensor, check_interval, success_threshold):
        """计算重要性"""
        voxel_density, voxel_opacity_sum, voxel_visit_sum = statistics_tensor
        unique_voxel_keys = unique_voxel_keys.long()

        import itertools
        # 生成26个邻居的3D offset（不包含(0,0,0)自身）
        offsets = [offset for offset in itertools.product([-1, 0, 1], repeat=3) if offset != (0, 0, 0)]
        offsets = torch.tensor(offsets, dtype=torch.int32, device=self.get_anchor.device)
        # 计算这些offset对应的线性偏移值
        linear_offsets = offsets[:, 0] * (max_indices[1] * max_indices[2]) + offsets[:, 1] * max_indices[2] + offsets[:, 2]  # (26,)
        # 计算每个体素的邻居索引
        neighbor_indices = unique_voxel_keys.unsqueeze(1) + linear_offsets.unsqueeze(0)  # 广播
        mask_neighbor = (neighbor_indices >= 0) & (neighbor_indices < max_indices[0]*max_indices[1]*max_indices[2])
        neighbor_indices[mask_neighbor == 0] = -1  # 只保留有效的邻居索引
        # 获取每个邻居的密度值等一系列统计量
        max_neighbor_idx = neighbor_indices.max().item() + 1
        lookup_table = torch.zeros(max_neighbor_idx, dtype=torch.bool, device=neighbor_indices.device)
        lookup_table[unique_voxel_keys] = True
        safe_indices = neighbor_indices.clone()
        safe_indices[safe_indices < 0] = 0
        neighbor_mask = lookup_table[safe_indices]
        neighbor_mask[neighbor_indices < 0] = False
        neighbor_count = neighbor_mask.sum(dim=1).unsqueeze(1)  # 计算每个体素的邻居数量
        # num_with_neighbors = (neighbor_count > 0).sum().item()  # 计算有邻居的体素数量
        # num_isolated_voxels = (neighbor_count == 0).sum().item()

        # 进行阈值剪枝
        opacity_threshold = 0.01
        density_threshold = 2
        neighboring_threshold = 1
        visits_threshold = check_interval * success_threshold * 0.5

        condition1 = (voxel_density < density_threshold)  # [N, 1]      筛选出透明度小于一定阈值的voxel
        condition2 = (voxel_opacity_sum < opacity_threshold)   # [N, 1]      筛选出密度小于一定阈值的voxel
        condition3 = (voxel_visit_sum > visits_threshold)  # [N, 1]   筛选出访问次数大于一定阈值的voxel
        condition4 = (neighbor_count < neighboring_threshold)  # [N, 1]   筛选出邻居数量大于一定阈值的voxel
        voxel_mask1 = torch.logical_and(condition1, condition2)
        voxel_mask2 = torch.logical_and(condition2, condition3)
        voxel_mask3 = torch.logical_and(condition1, condition4)
        voxel_mask = torch.logical_or(voxel_mask1, voxel_mask2)
        voxel_mask = torch.logical_or(voxel_mask, voxel_mask3)

        return voxel_mask

    # new code
    def fine_voxel_selection(self, statistics_tensor, coarse_voxel_size, fine_voxel_size, max_indices, min_indices, unique_voxel_keys, inverse_indices):
        """细粒度的锚点选择，选择重要程度低的锚点"""
        voxel_density, voxel_opacity_sum, voxel_visit_sum = statistics_tensor
        # 筛选出密度比较高的voxel
        density_threshold = coarse_voxel_size * 100
        fine_voxel_indices = (voxel_density.squeeze() > density_threshold).nonzero(as_tuple=False).squeeze()  # [M, 1]
        anchor_gradient_accum = self.offset_gradient_accum.squeeze().view(self.get_anchor.shape[0], self.n_offsets).sum(dim=1).unsqueeze(1)  # (N, 1)

        # 8.20
        # 体素聚类（GPU 上快速定位高密区域）
        # 数据规模（可设为 2e5；真实可到 1e6+，注意显存）

        KEEP_RATIO_PER_CLUSTER = 0.95
        ALPHA_SPATIAL = 0.3
        BETA_IMPORT = 0.4
        GAMMA_LOW_RANK = 0.3
        RANK_IMPORT = 3  # Tucker 第一维（3个重要度通道）秩
        RANK_POINT = 32  # 点维秩，这个值越大，聚类效果越好，但聚类速度越慢，是个超参数
        RANK_FEATURE = 3  # 特征维秩（<= 特征数，这里特征=4）

        num_voxels = unique_voxel_keys.numel()
        dense_voxel_mask = torch.zeros(num_voxels, dtype=torch.bool, device='cuda')
        dense_voxel_mask[fine_voxel_indices] = True
        points_in_dense = dense_voxel_mask[inverse_indices]  # [N] bool
        dense_coords = self.get_anchor[points_in_dense]  # [Nd,3]
        importance3 = torch.stack([anchor_gradient_accum.squeeze(), self.get_opacity.squeeze(), self.anchor_denom.squeeze()],
                                  dim=1)
        dense_importance3 = importance3[points_in_dense]  # [Nd,3]
        dense_inv = inverse_indices[points_in_dense]  # [Nd]
        print(
            f"[Info] 总点数={self.get_anchor.shape[0]}, 体素数={num_voxels}, 密集体素数={fine_voxel_indices.numel()}, 密集体素内点数={dense_coords.shape[0]}")

        # 每个密集体素内：Tucker(通道×点×特征) + 三分量融合
        final_scores = torch.zeros(self.get_anchor.shape[0], device='cuda')
        global_dense_idx = torch.nonzero(points_in_dense, as_tuple=False).squeeze(1)
        final_scores[global_dense_idx] = -1.0
        dense_voxel_ids = torch.unique(dense_inv)
        for vid in dense_voxel_ids:
            idx = torch.nonzero(dense_inv == vid, as_tuple=False).squeeze(1)
            global_idx = torch.nonzero(points_in_dense, as_tuple=False).squeeze(1)[idx]
            if idx.numel() == 0:
                continue
            pts = dense_coords[idx]  # [m,3]
            imp3 = dense_importance3[idx]  # [m,3] 三个通道
            # —— 局部归一化坐标 ——
            pmin = pts.min(dim=0).values
            pmax = pts.max(dim=0).values
            rng = (pmax - pmin).clamp_min(1e-6)
            pts_norm = (pts - pmin) / rng  # [m,3]

            # —— 重要度三通道分开归一化（按通道 min-max） ——
            ch_min = imp3.min(dim=0).values  # [3]
            ch_max = imp3.max(dim=0).values  # [3]
            imp3_norm = (imp3 - ch_min) / (ch_max - ch_min + 1e-8)  # [m,3]

            # —— 空间稀疏性分数（KNN 平均距离） ——
            m = pts_norm.shape[0]
            if m > 4096:
                ref_idx = torch.randint(0, m, (4096,), device='cuda')
                ref = pts_norm[ref_idx]
                d = torch.cdist(pts_norm, ref)  # [m,4096]
                knn = 8
                spatial_score, _ = d.topk(knn, largest=False)
                spatial_score = spatial_score.mean(dim=1)  # [m]
            elif m > 2:
            # if m > 2:
                d = torch.cdist(pts_norm, pts_norm)  # [m,m]
                knn = min(8, max(1, m - 1))
                spatial_score, _ = d.topk(knn + 1, largest=False)  # 含自身
                spatial_score = spatial_score[:, 1:].mean(dim=1)  # 去掉自身
            else:
                spatial_score = torch.zeros(m, device='cuda')

            s_sp = (spatial_score - spatial_score.min()) / (spatial_score.max() - spatial_score.min() + 1e-8)

            # —— 构造三通道特征张量：X.shape = [3, m, 4]
            # 每个通道的特征 = [x,y,z(归一化), 该通道的重要度]
            feats_c = [torch.cat([pts_norm, imp3_norm[:, c:c + 1]], dim=1) for c in range(3)]  # 3 × [m,4]
            feats_stack = torch.stack(feats_c, dim=0)  # [3, m, 4]

            # —— 采样用于 Tucker（控制显存），点维采样对所有通道一致 ——
            X = feats_stack  # [3, m, 4]

            # if m > MAX_POINTS_FOR_TUCKER:  # 可删
            #     samp_idx = torch.randint(0, m, (MAX_POINTS_FOR_TUCKER,), device=device)
            #     X = feats_stack[:, samp_idx, :]      # [3, Ms, 4]
            # else:
            #     X = feats_stack                      # [3, m, 4]

            # —— Tucker 分解：ranks = [r_import, r_point, r_feature] ——
            rI = min(RANK_IMPORT, X.shape[0])  # ≤3
            rP = min(RANK_POINT, X.shape[1])
            rF = min(RANK_FEATURE, X.shape[2])  # ≤4
            try:
                core, factors = tucker(X, ranks=[rI, rP, rF])
            except TypeError:
                core, factors = tucker(X, rank=[rI, rP, rF])

            # 因子矩阵：factors[0]=通道(U_importance) [3,rI]；factors[1]=点 [M,rP]；factors[2]=特征(U_feature) [4,rF]
            U_feature = factors[2]  # [4, rF]

            # —— 低秩分数（对三个通道分别投影后再平均） ——
            # 对全体点（未采样）计算，更稳健
            lr_list = []
            for c in range(3):
                low_rank_vec = feats_c[c] @ U_feature  # [m, rF]
                lr_list.append(torch.norm(low_rank_vec, dim=1))  # [m]
            s_lr = torch.stack(lr_list, dim=1).mean(dim=1)  # [m] 三通道平均
            s_lr = (s_lr - s_lr.min()) / (s_lr.max() - s_lr.min() + 1e-8)

            # —— 重要度汇总分数（把三通道的重要度做简单融合；也可换成加权和/最大值） ——
            s_imp = imp3_norm.mean(dim=1)  # [m] 也可用 .amax(dim=1).values

            # —— 融合三类分数 ——
            score = ALPHA_SPATIAL * s_sp + BETA_IMPORT * s_imp + GAMMA_LOW_RANK * s_lr  # [m]

            # —— 在该体素内保留前 KEEP_RATIO_PER_CLUSTER ——
            keep_m = max(1, int(m * KEEP_RATIO_PER_CLUSTER))
            _, keep_local = torch.topk(score, k=keep_m, largest=True)
            final_scores[global_idx[keep_local]] = score[keep_local]

        # 仅保留 final_scores >= 0 的点（被选中）
        global_mask = final_scores >= 0
        return ~global_mask

    # new code
    def visualize_3d_tensor_basic(self, tensor, threshold=0.5):
        """
        基础3D散点图可视化
        适用于较小规模的二值化张量
        """
        # 生成坐标网格
        ply_tensor = tensor.cpu().numpy()
        x, y, z = np.indices(ply_tensor.shape)

        # 筛选非零体素
        mask = ply_tensor > threshold
        x_pts = x[mask]
        y_pts = y[mask]
        z_pts = z[mask]

        # 创建3D画布
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制散点
        ax.scatter(x_pts, y_pts, z_pts, c=z_pts,
                   cmap='viridis', s=5, alpha=0.3)

        # 设置坐标轴
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Tensor Visualization')

        plt.show()

    # new code
    def coarse_adjustment(self, itr, voxel_size, check_interval, success_threshold):
        max_indices, min_indices, unique_voxel_keys, inverse_indices, statistics_tensor = self.build_statistics_tensor(voxel_size=voxel_size)
        masked_voxels = self.coarse_voxel_selection(itr, max_indices, unique_voxel_keys, inverse_indices, statistics_tensor, check_interval, success_threshold)
        masked_voxels = masked_voxels.squeeze()
        masked_anchors = masked_voxels[inverse_indices]

        return masked_anchors
        # self.visualize_3d_tensor_basic(statistics_tensor)

    # new code
    def fine_adjustment(self, coarse_voxel_size, fine_voxel_size):
        max_indices, min_indices, unique_voxel_keys, inverse_indices, statistics_tensor = self.build_statistics_tensor(voxel_size=coarse_voxel_size)
        masked_anchors = self.fine_voxel_selection(statistics_tensor, coarse_voxel_size, fine_voxel_size, max_indices, min_indices, unique_voxel_keys, inverse_indices)

        return masked_anchors

    # new code
    def my_adjust_anchor(self, itr, coarse_voxel_size, voxel_size, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors 添加锚点
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]    计算梯度的归一化值：在训练过程中累积的偏移量/梯度和访问次数
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)  # 计算梯度范数
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)  # 筛选出访问次数大于一定阈值的偏移量
        # 将筛选出的偏移量传入 anchor_growing 函数，用于增长锚点
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom 对于已经被选中用来增加新锚点的位置，重置访问次数和累积梯度，然后更新参数以匹配新的锚点数量
        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)


        # prune anchors 剪枝锚点
        # new code
        coarse_adjustment_starting = 2000
        coarse_adjustment_ending = 5000
        fine_adjustment_ending = 15000
        prune_mask = torch.zeros(self.get_anchor.shape[0], device="cuda", dtype=torch.bool)

        if itr >= coarse_adjustment_starting and itr <= coarse_adjustment_ending:
            prune_mask = self.coarse_adjustment(itr, coarse_voxel_size, check_interval, success_threshold)
        # if itr >= coarse_adjustment_ending and itr <= fine_adjustment_ending and itr % 200 == 0:
        #     prune_mask = self.fine_adjustment(coarse_voxel_size, voxel_size)

        # # prune anchors 剪枝锚点，原方法
        prune_mask_ori = (self.opacity_accum < min_opacity*self.anchor_denom).squeeze(dim=1) # [N, 1]      筛选出透明度小于一定阈值的锚点
        anchors_mask_ori = (self.anchor_denom > check_interval*success_threshold).squeeze(dim=1) # [N, 1]   筛选出访问次数大于一定阈值的锚点
        prune_mask_ori = torch.logical_and(prune_mask_ori, anchors_mask_ori) # [N]   多次访问但透明度很低的锚点会被剪枝

        prune_mask = torch.logical_or(prune_mask, prune_mask_ori) # [N]   合并剪枝结果

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum  重置透明度累积值和最大半径
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_denom = self.anchor_denom[~prune_mask]
        del self.anchor_denom
        self.anchor_denom = temp_anchor_denom

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

