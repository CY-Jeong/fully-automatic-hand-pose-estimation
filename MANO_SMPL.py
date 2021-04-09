'''
    file:   MANO_SMPL.py
    date:   2019_07_15
    modifier: Seungryul Baek
    source:   This code is modified from SMPL.py of https://github.com/MandyMo/pytorch_HMR.
    If you use this code for your research, please cite:

    @article{sbaek_cvpr_2019,
      title={Pushing the envelope for RGB-based dense 3D hand pose estimation via neural rendering},
      author={Seungryul Baek and Kwang In Kim and Tae-Kyun Kim},
      journal={CVPR},
      year={2019}
    }
    and
    @article{zhang2019end,
      title={End-to-end Hand Mesh Recovery from a Monocular RGB Image},
      author={Zhang, Xiong and Li, Qiang and Zhang, Wenbo and Zheng, Wen},
      journal={arXiv preprint arXiv:1902.09305},
      year={2019}
    }
'''

import torch
import numpy as np
import torch.nn as nn
import pickle
from torch.autograd import Variable

import torch.nn.functional as F

# joint mapping indices from mano to bighand
mano2bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]

class MANO_SMPL(nn.Module):
    def __init__(self, mano_pkl_path):
        super(MANO_SMPL, self).__init__()

        # Load the MANO_RIGHT.pkl
        with open(mano_pkl_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[745][0] = 1
        np_J_addition[333][1] = 1
        np_J_addition[444][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[672][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
        np_J_regressor[:, 0] = 0
        np_J_regressor[95][0] = 0.5
        np_J_regressor[22][0] = 0.5
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_hand_component = np.array(model['hands_components'], dtype=np.float)
        self.register_buffer('hands_comp', torch.from_numpy(np_hand_component).float())

        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)
        hands_mean = np.zeros(np_hand_component.shape[1])
        self.register_buffer('hands_mean', torch.from_numpy(hands_mean).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None



    def forward(self, beta, theta1, quat, get_skin=False):
        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # theta = theta1*torch.FloatTensor([0,0,0, 0,1,1, 0,0,1, 0,0,0, 0,1,1, 0,0,1, 0,0,0, 0,1,1, 0,0,1, 0,0,0,\
        #                                   0,1,1, 0,0,1, 1,1,1, 1,1,1, 1,1,1])[None,:].cuda()
        theta = theta1
        Rs = self.batch_rodrigues((torch.matmul(theta, self.hands_comp[:, :45].T) + self.hands_mean).view(-1, 3)).view(-1, 15, 3, 3)

        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = self.batch_global_rigid_transformation(torch.cat([self.batch_rodrigues(quat).view(-1, 1, 3, 3), Rs], dim=1), J[:, :16, :], self.parents, rotate_base=True)
        #self.J_transformed, A = self.batch_global_rigid_transformation(torch.cat([self.quat2mat(quat).view(-1, 1, 3, 3), Rs], dim=1), J[:, :16, :], self.parents, rotate_base=True)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1).cuda()], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

        return rotMat

    def batch_rodrigues(self, theta):
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim=1)

        return self.quat2mat(quat)

    def batch_global_rigid_transformation(self, Rs, Js, parent, rotate_base=False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
            root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3]
        Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 16, 1, 1)).cuda()], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A

    # returns joints in bighand skeleton indexing
    def get_mano_vertices(self, quat, pose, shape, cam, rotate=False):
        """
        :param shape: mano global rotation params in quaternion representation [batch_size, 4]
        :param shape: mano articulation params [batch_size, 45]
        :param shape: mano shape params [batch_size, 10]
        :param pose: mano scale and translation params [batch_size, 4]
        :return:
        """
        # check if not tensor: wrap
        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, dtype=torch.float).cuda()

        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float).cuda()

        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float).cuda()

        if not isinstance(cam, torch.Tensor):
            cam = torch.tensor(cam, dtype=torch.float).cuda()

        verts, joints, Rs = self.forward(shape, pose, quat, get_skin=True)

        if rotate:
            verts[0, :, 1] *= -1
            # verts[0, :, 2] *= -1


        scale = cam[:, 0].contiguous().view(-1, 1, 1)
        trans = cam[:, 1:].contiguous().view(cam.size(0), 1, -1)

        verts = verts * scale + trans

        proj = scale * joints
        proj_xy = proj[:, :, :2] + trans[:, :, :2]
        proj_z = proj[:, :, 2] + trans[:, :, 2]
        proj_z = proj_z[:, :, None]
        joints = torch.cat((proj_xy, proj_z), 2)

        joints[:, :, :3] = joints[:, :, :3] * 0.5 + 0.5

        return verts.cpu().detach().numpy(), joints.cpu().detach().numpy()[0][mano2bighand_skeidx, :]