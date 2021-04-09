
'''
    file:   model.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    https://github.com/akanazawa/cmr
    Modifiers: Anil Armagan and Seungryul Baek
'''

from LinearModel import LinearModel
import torch.nn as nn
import numpy as np
import torch
from pose import util
from SMPL import SMPL
from pose import config
import Resnet
import pickle
import sys
import nnutils
import torch.nn.functional as F
import os
from pose.config import args

device = 'cuda:0'

posenet_finetune_path = args.posenet_finetune_path

# 2d pose estimator - pretrained
class CPM2DPose(nn.Module):
    def __init__(self):
        super(CPM2DPose, self).__init__()
        weight_dict = None
        if os.path.exists(posenet_finetune_path):
            with open(posenet_finetune_path, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in list()])}
            print('Setting pretrained weights for CPM!')
        else:
            print('model weights {} not exist!'.format(posenet_finetune_path))

        self.scoremap_list = []
        self.layers_per_block = [2, 2, 4, 2]
        self.out_chan_list = [64, 128, 256, 512]
        self.pool_list = [True, True, True, False]

        self.relu = F.leaky_relu
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True) # conv0_1
        self.conv1_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv1_1/weights']).permute(3, 2, 0, 1).float())
        self.conv1_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv1_1/biases']).float())
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) # conv0_2
        self.conv1_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv1_2/weights']).permute(3, 2, 0, 1).float())
        self.conv1_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv1_2/biases']).float())
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv2_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv2_1/weights']).permute(3, 2, 0, 1).float())
        self.conv2_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv2_1/biases']).float())
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv2_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv2_2/weights']).permute(3, 2, 0, 1).float())
        self.conv2_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv2_2/biases']).float())
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_1/weights']).permute(3, 2, 0, 1).float())
        self.conv3_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_1/biases']).float())
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_2/weights']).permute(3, 2, 0, 1).float())
        self.conv3_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_2/biases']).float())
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_3.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_3/weights']).permute(3, 2, 0, 1).float())
        self.conv3_3.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_3/biases']).float())
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv3_4.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_4/weights']).permute(3, 2, 0, 1).float())
        self.conv3_4.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv3_4/biases']).float())
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_1/weights']).permute(3, 2, 0, 1).float())
        self.conv4_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_1/biases']).float())
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_2/weights']).permute(3, 2, 0, 1).float())
        self.conv4_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_2/biases']).float())
        self.conv4_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_3.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_3/weights']).permute(3, 2, 0, 1).float())
        self.conv4_3.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_3/biases']).float())
        self.conv4_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_4.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_4/weights']).permute(3, 2, 0, 1).float())
        self.conv4_4.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_4/biases']).float())
        self.conv4_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_5.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_5/weights']).permute(3, 2, 0, 1).float())
        self.conv4_5.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_5/biases']).float())
        self.conv4_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_6.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_6/weights']).permute(3, 2, 0, 1).float())
        self.conv4_6.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_6/biases']).float())
        self.conv4_7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True)  # conv0_2
        self.conv4_7.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_7/weights']).permute(3, 2, 0, 1).float())
        self.conv4_7.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv4_7/biases']).float())
        self.conv5_1 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv5_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv5_1/weights']).permute(3, 2, 0, 1).float())
        self.conv5_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv5_1/biases']).float())
        self.conv5_2 = nn.Conv2d(512, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv5_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv5_2/weights']).permute(3, 2, 0, 1).float())
        self.conv5_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv5_2/biases']).float())
        self.conv6_1 = nn.Conv2d(149, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_1/weights']).permute(3, 2, 0, 1).float())
        self.conv6_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_1/biases']).float())
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_2/weights']).permute(3, 2, 0, 1).float())
        self.conv6_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_2/biases']).float())
        self.conv6_3 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_3.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_3/weights']).permute(3, 2, 0, 1).float())
        self.conv6_3.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_3/biases']).float())
        self.conv6_4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_4.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_4/weights']).permute(3, 2, 0, 1).float())
        self.conv6_4.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_4/biases']).float())
        self.conv6_5 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv6_5.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_5/weights']).permute(3, 2, 0, 1).float())
        self.conv6_5.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_5/biases']).float())
        self.conv6_6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv6_6.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_6/weights']).permute(3, 2, 0, 1).float())
        self.conv6_6.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_6/biases']).float())
        self.conv6_7 = nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv6_7.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_7/weights']).permute(3, 2, 0, 1).float())
        self.conv6_7.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv6_7/biases']).float())
        self.conv7_1 = nn.Conv2d(149, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_1.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_1/weights']).permute(3, 2, 0, 1).float())
        self.conv7_1.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_1/biases']).float())
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_2.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_2/weights']).permute(3, 2, 0, 1).float())
        self.conv7_2.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_2/biases']).float())
        self.conv7_3 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_3.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_3/weights']).permute(3, 2, 0, 1).float())
        self.conv7_3.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_3/biases']).float())
        self.conv7_4 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_4.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_4/weights']).permute(3, 2, 0, 1).float())
        self.conv7_4.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_4/biases']).float())
        self.conv7_5 = nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3, bias=True)  # conv0_2
        self.conv7_5.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_5/weights']).permute(3, 2, 0, 1).float())
        self.conv7_5.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_5/biases']).float())
        self.conv7_6 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv7_6.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_6/weights']).permute(3, 2, 0, 1).float())
        self.conv7_6.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_6/biases']).float())
        self.conv7_7 = nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0, bias=True)  # conv0_2
        self.conv7_7.weight.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_7/weights']).permute(3, 2, 0, 1).float())
        self.conv7_7.bias.data = torch.nn.Parameter(torch.tensor(weight_dict['PoseNet2D/conv7_7/biases']).float())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.relu(self.conv3_4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.relu(self.conv4_4(x))
        x = self.relu(self.conv4_5(x))
        x = self.relu(self.conv4_6(x))
        encoding = self.relu(self.conv4_7(x))
        x = self.relu(self.conv5_1(encoding))
        scoremap = self.conv5_2(x)

        x = torch.cat([scoremap, encoding],1)
        x = self.relu(self.conv6_1(x))
        x = self.relu(self.conv6_2(x))
        x = self.relu(self.conv6_3(x))
        x = self.relu(self.conv6_4(x))
        x = self.relu(self.conv6_5(x))
        x = self.relu(self.conv6_6(x))
        scoremap = self.conv6_7(x)
        x = torch.cat([scoremap, encoding], 1)
        x = self.relu(self.conv7_1(x))
        x = self.relu(self.conv7_2(x))
        x = self.relu(self.conv7_3(x))
        x = self.relu(self.conv7_4(x))
        x = self.relu(self.conv7_5(x))
        x = self.relu(self.conv7_6(x))
        x = self.conv7_7(x)
        return x

class ThetaRegressor(LinearModel):
    def __init__(self, fc_layers, fc_layers2, use_dropout, drop_prob, use_ac_func, iterations):
        super(ThetaRegressor, self).__init__(fc_layers, fc_layers2, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations

        batch_size = max(args.batch_size1, args.batch_size2 , args.batch_size_eval)
        mean_theta = np.tile(util.load_mean_theta(), batch_size).reshape((batch_size, -1))
        self.register_buffer('mean_theta', torch.from_numpy(mean_theta).float())
    '''
        param:
            inputs: is the output of encoder, which has 2048 features
        
        return:
            a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations X N X 85(or other theta count)
    '''
    def forward(self, inputs, skeletons2d):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        skk = skeletons2d.view(-1, 42)

        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta, skk], 1)
            theta = theta + self.fc_blocks(total_inputs)
            total_inputs2 = torch.cat([inputs, theta, skk], 1)
            skk = skk + self.fc_blocks2(total_inputs2)
            thetas.append(theta)
        return thetas, skk

class HMRNetBase(nn.Module):
    def __init__(self):
        super(HMRNetBase, self).__init__()
        self._read_configs()

        print('start creating sub modules...')
        self._create_sub_modules()

    def _read_configs(self):
        def _check_config():
            encoder_name = args.encoder_network
            feature_count = args.feature_count           
            if encoder_name == 'resnet50':
                assert args.crop_size == 96
            else:
                msg = 'invalid encoder network, only {} is allowd, got {}'.format(args.allowed_encoder_net, encoder_name)
                sys.exit(msg)
            assert config.encoder_feature_count[encoder_name] == feature_count

        _check_config()
        
        self.encoder_name = args.encoder_network
        self.total_theta_count = args.total_theta_count
        self.feature_count = args.feature_count
        
    def _create_sub_modules(self):
        '''
            ddd smpl model, SMPL can create a mesh from beta & theta
        '''
        self.smpl = SMPL()

        '''
            only resnet50
        '''
        if self.encoder_name == 'resnet50':
            print('creating ResNet50')
            self.encoder = Resnet.load_Res50RGBModel()
        else:
            assert 0

        '''
            regressor can predict theta parameters(needed by SMPL) in a iteratirve way
        '''
        fc_layers = [self.feature_count + 42 + self.total_theta_count, 1024, 1024, 63]
        fc_layers2 = [self.feature_count + 42 + self.total_theta_count, 1024, 1024, 42]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False] #unactive the last layer
        iterations = 3
        self.regressor = ThetaRegressor(fc_layers, fc_layers2, use_dropout, drop_prob, use_ac_func, iterations)
        self.iterations = iterations
        # self.interp3 = torch.nn.Upsample(size=(256, 256), mode='bilinear')
        print('finished create the encoder modules...')

    def forward(self, inputs, skeletons2d):

        feature = self.encoder(inputs)

        self.feature = feature
        self.skeletons2d = skeletons2d
        thetas, improvedskk = self.regressor(feature, skeletons2d)
        self.iskeleton2d = improvedskk.reshape(-1, 21, 2)
        detail_info = []
        for theta in thetas:
            detail_info.append(self._calc_detail_info(theta))
        return detail_info

    '''
        purpose:
            calc verts, joint2d, joint3d, Rotation matrix

        inputs:
            theta: N X (3 + 72 + 10)

        return:
            thetas, verts, j2d, j3d, Rs
    '''
    
    def _calc_detail_info(self, theta):
        cam = theta[:, 0:4].contiguous()
        q = theta[:, 4:8].contiguous()
        pose = theta[:, 8:53].contiguous()
        shape = theta[:, 53:].contiguous()
        verts, j3d, Rs = self.smpl(beta = shape, theta = pose, quat=q, get_skin = True)

        # unproject
        proj_fn = nnutils.geom_utils.orthographic_proj_withz

        j2dv = proj_fn(j3d, cam, 0)
        j2d= proj_fn(j3d, cam, 0)
        j2d[:, :, :3] = (j2d[:, :, :3] * 0.5) + 0.5
        j2dv[:, :, :3] = (j2dv[:, :, :3] * 0.5) + 0.5

        return (theta, verts, j2d, j3d, Rs, cam, j2dv, shape, self.feature, self.skeletons2d, self.iskeleton2d)


if __name__ == '__main__':
    cam = np.array([[0.9, 0, 0]], dtype = np.float)
    pose= np.array([
            -1.22162998e+00,   5.17162502e+00,   1.16706634e+00,
            +1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
            +1.52801601e-02,  10.16911056e-02,  -6.02894090e-03,
            1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
            2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
            7.79098943e-02,   2.96353287e-01,   6.44420758e-02,
            -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
            -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
            -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
            9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
            -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
            -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
            -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
            -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
            -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02],dtype = np.float)

    beta = np.array([[-3.54196257,  0.90870435, -1.0978663 , -0.20436199,  0.18589762, 0.55789026, -0.18163599,  0.12002746, -0.09172286,  0.4430783 ]])
    real_shapes = torch.from_numpy(beta).float().to(device)
    real_poses  = torch.from_numpy(pose).float().to(device)

    net = HMRNetBase().to(device)
    nx = torch.rand(2, 3, 96, 96).float().to(device)
