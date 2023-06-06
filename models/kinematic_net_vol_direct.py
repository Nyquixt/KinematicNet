import models.resnet as resnet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from utils import volumetric, op

def module_to_dict(module):
    return {key: getattr(module, key) for key in module.__all__}

class Regressor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear = nn.Linear(in_features, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.residual_linear1 = ResidualLinear(1024)
        self.residual_linear2 = ResidualLinear(1024)
        self.out = nn.Linear(1024, out_features)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.residual_linear1(x)
        x = self.residual_linear2(x)
        x = self.out(x)
        return x

class ResidualLinear(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, features),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.layers(x)
        out = z + x
        return out

class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Encoder(nn.Module):
    def __init__(self, volume_size):
        super().__init__()

        self.volume_size = volume_size

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        if self.volume_size == 16:
            self.encoder_pool3 = Pool3DBlock(2)
            self.encoder_res3 = Res3DBlock(128, 128)
        
        if self.volume_size == 32:
            self.encoder_pool3 = Pool3DBlock(2)
            self.encoder_res3 = Res3DBlock(128, 128)
            self.encoder_pool4 = Pool3DBlock(2)
            self.encoder_res4 = Res3DBlock(128, 128)

        if self.volume_size == 64:
            self.encoder_pool3 = Pool3DBlock(2)
            self.encoder_res3 = Res3DBlock(128, 128)
            self.encoder_pool4 = Pool3DBlock(2)
            self.encoder_res4 = Res3DBlock(128, 128)
            self.encoder_pool5 = Pool3DBlock(2)
            self.encoder_res5 = Res3DBlock(128, 128)

    def forward(self, x):
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        if self.volume_size == 16:
            x = self.encoder_pool3(x)
            x = self.encoder_res3(x)

        if self.volume_size == 32:
            x = self.encoder_pool3(x)
            x = self.encoder_res3(x)
            x = self.encoder_pool4(x)
            x = self.encoder_res4(x)
        
        if self.volume_size == 64:
            x = self.encoder_pool3(x)
            x = self.encoder_res3(x)
            x = self.encoder_pool4(x)
            x = self.encoder_res4(x)
            x = self.encoder_pool5(x)
            x = self.encoder_res5(x)

        return x

class KinematicNetDirect(nn.Module):
    def __init__(self, config,
                 deconv_with_bias=False,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4)):
        super().__init__()
        self.num_joints = config.num_joints

        self.volume_size = config.model.volume_size
        self.cuboid_side = config.model.cuboid_side
        self.root_mode = config.model.root_mode # local or global
        
        self.deconv_with_bias = deconv_with_bias
        self.num_deconv_layers, self.num_deconv_filters, self.num_deconv_kernels = num_deconv_layers, num_deconv_filters, num_deconv_kernels
        self.rotation_type = config.model.rotation_type
        # rotation type
        if self.rotation_type == "quaternion":
            self.rot_dim = 4
        elif self.rotation_type == "euler":
            self.rot_dim = 3
        elif self.rotation_type == "6d":
            self.rot_dim = 6

        self.backbone = module_to_dict(resnet)[config.model.backbone.name](config.model.backbone.pretrained)
        # used for deconv layers
        if config.model.backbone.name == 'resnet34' or config.model.backbone.name == 'resnet18':
            self.inplanes = 512
        else:
            self.inplanes = 2048
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers,
            self.num_deconv_filters,
            self.num_deconv_kernels,
        )
        
        # 1x1 conv to squeeze channels
        self.process_features = nn.Conv2d(256, 32, 1)

        self.encoder = Encoder(self.volume_size)

        self.regressor = Regressor(1024, self.num_joints * self.rot_dim)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views, c, h, w = images.size()
        images = images.view(-1, c, h, w) # B*V, C, H, W
        x = self.backbone(images)
        heatmaps = self.deconv_layers(x) # B*V, c', h', w'

        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        # calcualte shapes
        image_shape, heatmaps_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmaps_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device) # store pelvis coordinates
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device) # b x 64 x 64 x 64 x 3
        for batch_i in range(batch_size):
            if self.root_mode == 'global':
                keypoints_3d = batch['keypoints_3d'][batch_i]
                base_point = keypoints_3d[0, :3] # global root
            else:
                base_point = np.array([0., 0., 0.]) # local root

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side]) # 2500 x 2500 x 2500
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)
            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = heatmaps.view(-1, *heatmaps.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method='softmax') # B, C, 16, 16, 16
        # feed forward to encoder and regressor
        volumes = self.encoder(volumes) # B, C', 2, 2, 2
        volumes = torch.flatten(volumes, 1)
        representation = self.regressor(volumes) # B, J*rot_dim

        representation = representation.view(batch_size, self.num_joints, self.rot_dim) # B, J, rot_dim
        if self.rotation_type == "quaternion":
            representation = F.normalize(representation, dim=-1)

        return representation