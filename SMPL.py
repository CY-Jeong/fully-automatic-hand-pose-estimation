'''
    file:   SMPL.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
    mark:   the algorithm is cited from original SMPL
'''
import torch
from pose.config import args
import numpy as np
from pose.util import batch_global_rigid_transformation, batch_rodrigues
import torch.nn as nn
import pickle

mano_path = args.mano_path

class SMPL(nn.Module):
    def __init__(self):
        super(SMPL, self).__init__()
        fname = mano_path
        with open(fname, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[745][0] = 1
        np_J_addition[333][1] = 1
        np_J_addition[444][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[672][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_hand_component = np.array(model['hands_components'], dtype=np.float)
        self.register_buffer('hands_comp', torch.from_numpy(np_hand_component).float())

        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)
        self.register_buffer('hands_mean', torch.from_numpy(np_hand_mean).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        batch_size = 400 #max(args.batch_size1, args.batch_size2, args.batch_size_eval)
        np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, beta, theta, quat, get_skin=False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues((torch.matmul(theta, self.hands_comp) + self.hands_mean).view(-1, 3)).view(-1, 15, 3, 3)
        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        from pose.util import quat2mat
        self.J_transformed, A = batch_global_rigid_transformation(
            torch.cat([quat2mat(quat).view(-1, 1, 3, 3), Rs], dim=1), J[:, :16, :], self.parents, rotate_base=True)

        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
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


if __name__ == '__main__':
    device = torch.device('cuda', 0)
    smpl = SMPL(args.smpl_model, obj_saveable=True).to(device)

    pose = np.array(
        [0.603789, -0.367011, 0.453897, 0.651877, -0.656605, -0.212550, -0.632679, 0.205984, 1.017134, -0.031939,
         0.332727,
         -0.415808, -1.454459, 1.093990, -0.064683, 0.319805, 0.742956, 1.634443, 0.273575, 0.269810, 0.660156,
         -0.086993, -1.947417, -0.632760, -0.609980, 1.683600, 2.866103, 2.875561, -2.064497, -0.469832, -1.280301,
         -0.474019, -3.393826, -0.557894, -0.295235, 1.362216, -1.732486, 0.374566, -0.569821, 0.691290, 0.227309,
         -1.397949, 2.431975, 0.010688, 1.306405], dtype=np.float)
    # pose = np.ones([45],dtype=np.float)
    beta = np.array([-0.25349993, 0.25009069, 0.21440795, 0.78280628, 0.08625954,
                     0.28128183, 0.06626327, -0.26495767, 0.09009246, 0.06537955])

    q = np.array([0, 0, 0, 1])

    vbeta = torch.tensor(np.array([beta])).float().to(device)
    vpose = torch.tensor(np.array([pose])).float().to(device)
    q = torch.tensor(np.array([q])).float().to(device)

    verts, j, r = smpl(vbeta, vpose, q, get_skin=True)

    smpl.save_obj(verts[0].cpu().numpy(), './meshRIGHT3.obj')
