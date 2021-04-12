import torch
import numpy as np
import pickle
import tqdm
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt

import nnutils
from config_opt import cfg
from nnutils.nmr import NeuralRenderer, NeuralRenderer_depth
from nnutils.loss_utils import LaplacianLoss
from pose import util
from MANO_SMPL  import MANO_SMPL
from utils import plot_hand

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

proj_fn = nnutils.geom_utils.orthographic_proj_withz

class MANOFIT(torch.nn.Module):
    def __init__(self, theta):
        super(MANOFIT, self).__init__()
        vertices_var = torch.FloatTensor(theta).to(device)
        self.cam = torch.nn.Parameter(vertices_var[:, 0:4].contiguous())
        self.q = torch.nn.Parameter(vertices_var[:, 4:7].contiguous())
        self.pose = torch.nn.Parameter(vertices_var[:, 8:53].contiguous())
        self.shape = torch.nn.Parameter(vertices_var[:, 53:].contiguous())
    def forward(self, mano):
        verts, j3d, Rs = mano(beta=self.shape, theta1=self.pose, quat=self.q, get_skin=True)
        j2d = proj_fn(j3d, self.cam, 0)
        j2d[:, :, :3] = j2d[:, :, :3] * 0.5 + 0.5
        return j2d, verts, j3d


class MANOFITTER(object):
    def __init__(self, calibration):
        self.calibration = calibration
        self.mask_renderer = NeuralRenderer(cfg.SEG_SIZE).to(device)
        self.depth_renderer = NeuralRenderer_depth(cfg.SEG_SIZE).to(device)
        self.frame = None
        with open(cfg.MANO_FILE, 'rb') as f:
            mano_data = pickle.load(f, encoding='latin1')
        self.faces = mano_data['f']
        facesl = torch.from_numpy(np.array(self.faces[None, :, :], dtype=np.long))
        facesl1 = facesl.repeat(cfg.BATCH_SIZE, 1, 1)
        self.triangle_loss = LaplacianLoss(facesl1.to(device))

        faces20 = torch.from_numpy(np.array(self.faces[None, :, :], dtype=np.int32))
        faces20 = faces20.repeat(cfg.BATCH_SIZE, 1, 1)
        self.faces20 = torch.autograd.Variable(faces20).cuda()
        theta = np.tile(util.load_mean_theta(), cfg.BATCH_SIZE).reshape((cfg.BATCH_SIZE, -1))
        self.opt_model = MANOFIT(theta)
        self.mano = MANO_SMPL(cfg.MANO_FILE).to(device)
        self.skkidx = [0, 8, 7, 6, 12, 11, 10, 20, 19, 18, 16, 15, 14, 4, 3, 2, 1, 5, 9, 13, 17]
        self.skkidx2 = [0, 16, 15, 14, 13, 17, 3, 2, 1, 18, 6, 5, 4, 19, 12, 11, 10, 20, 9, 8, 7]

    def skel_fit(self, points_3d):
        print('start')
        print('Optimizing Vertices: ')
        iteration = cfg.OPT_VERT_ETERATION

        self.opt_model.pose = torch.nn.Parameter(self.opt_model.pose[:, :].clone().cuda(), requires_grad=True)
        self.opt_model.q = torch.nn.Parameter(self.opt_model.q[:, :].clone().cuda(), requires_grad=True)
        self.opt_model.cam = torch.nn.Parameter(self.opt_model.cam[:, :].clone().cuda(), requires_grad=True)
        self.opt_model.shape = torch.nn.Parameter(self.opt_model.shape[:, :].clone().cuda(), requires_grad=False)
        points_3d = torch.FloatTensor(points_3d[None, :, :]).to(device)
        optimizer = torch.optim.Adam(self.opt_model.parameters(), lr=1e-2, betas=(0.9, 0.999))
        for ix in tqdm.tqdm(range(iteration)):
            for view in range(cfg.VIEW_NUM):
                optimizer.zero_grad()
                skk2d, vertsp, skk3d = self.opt_model.forward(self.mano)

                loss = torch.nn.MSELoss()(skk2d[:, 0, :3], points_3d[0, 0, :3])*1000. + \
                        torch.nn.MSELoss()(skk2d[:, :, :3], points_3d[0, self.skkidx, :3]) + \
                        self.triangle_loss(vertsp)
                loss.backward()
                optimizer.step()
            print(f"epoch : {ix} loss : {loss}")

    def seg_fit(self, seg):
        iteration = cfg.OPT_SEG_ETERATION
        self.opt_model.q = torch.nn.Parameter(self.opt_model.q[:, :].clone().cuda(), requires_grad=True)
        self.opt_model.cam = torch.nn.Parameter(self.opt_model.cam[:, :].clone().cuda(), requires_grad=True)
        self.opt_model.pose = torch.nn.Parameter(self.opt_model.pose[:, :].clone().cuda(), requires_grad=True)
        optimizer = torch.optim.Adam(self.opt_model.parameters(), lr=1e-2, betas=(0.9, 0.999))

        for ix in tqdm.tqdm(range(iteration)):
            for view in range(cfg.VIEW_NUM):
                optimizer.zero_grad()
                skk2d, vertsp, skk3d = self.opt_model.forward(self.mano)

                g_loss = self._geometric_loss(skk2d)

                vertsp2d = proj_fn(vertsp, self.opt_model.cam, 0)
                vertsp2d[:, :, :3] = vertsp2d[:, :, :3] * 0.5 + 0.5
                s_loss = self._seg_loss(view, vertsp2d, seg[view], ix)

                if s_loss > 0.5:
                    continue
                loss = self.triangle_loss(vertsp) + s_loss + g_loss

                loss.backward()
                optimizer.step()

    def _geometric_loss(self, skk2d):
        bone0_0 = skk2d[:, 17] - skk2d[:, 3]
        bone0_1 = skk2d[:, 3] - skk2d[:, 2]
        bone0_2 = skk2d[:, 2] - skk2d[:, 1]
        bone0_3 = skk2d[:, 1] - skk2d[:, 0]

        bone1_0 = skk2d[:, 18] - skk2d[:, 6]
        bone1_1 = skk2d[:, 6] - skk2d[:, 5]
        bone1_2 = skk2d[:, 5] - skk2d[:, 4]
        bone1_3 = skk2d[:, 4] - skk2d[:, 0]

        bone2_0 = skk2d[:, 19] - skk2d[:, 12]
        bone2_1 = skk2d[:, 12] - skk2d[:, 11]
        bone2_2 = skk2d[:, 11] - skk2d[:, 10]
        bone2_3 = skk2d[:, 10] - skk2d[:, 0]

        bone3_0 = skk2d[:, 20] - skk2d[:, 9]
        bone3_1 = skk2d[:, 9] - skk2d[:, 8]
        bone3_2 = skk2d[:, 8] - skk2d[:, 7]
        bone3_3 = skk2d[:, 7] - skk2d[:, 0]

        dotloss = torch.abs(torch.clamp(torch.matmul(bone0_0, bone0_1.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone0_1, bone0_2.T), max=0)) + \
                  torch.abs(torch.clamp(torch.matmul(bone0_2, bone0_3.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone1_0, bone1_1.T), max=0)) + \
                  torch.abs(torch.clamp(torch.matmul(bone1_1, bone1_2.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone1_2, bone1_3.T), max=0)) + \
                  torch.abs(torch.clamp(torch.matmul(bone2_0, bone2_1.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone2_1, bone2_2.T), max=0)) + \
                  torch.abs(torch.clamp(torch.matmul(bone2_2, bone2_3.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone3_0, bone3_1.T), max=0)) + \
                  torch.abs(torch.clamp(torch.matmul(bone3_1, bone3_2.T), max=0)) + torch.abs(
            torch.clamp(torch.matmul(bone3_2, bone3_3.T), max=0))

        cross0 = torch.cross(bone0_0, bone0_1)
        dotloss1 = torch.abs(torch.matmul(cross0, bone0_2.T))
        cross1 = torch.cross(bone1_0, bone1_1)
        dotloss2 = torch.abs(torch.matmul(cross1, bone1_2.T))
        cross2 = torch.cross(bone2_0, bone2_1)
        dotloss3 = torch.abs(torch.matmul(cross2, bone2_2.T))
        cross3 = torch.cross(bone3_0, bone3_1)
        dotloss4 = torch.abs(torch.matmul(cross3, bone3_2.T))

        cross4 = torch.cross(bone0_1, bone0_2)
        cross5 = torch.cross(bone1_1, bone1_2)
        cross6 = torch.cross(bone2_1, bone2_2)
        cross7 = torch.cross(bone3_1, bone3_2)

        dotloss5 = torch.abs(torch.clamp(torch.matmul(cross0, cross4.T), max=0))
        dotloss6 = torch.abs(torch.clamp(torch.matmul(cross1, cross5.T), max=0))
        dotloss7 = torch.abs(torch.clamp(torch.matmul(cross2, cross6.T), max=0))
        dotloss8 = torch.abs(torch.clamp(torch.matmul(cross3, cross7.T), max=0))

        return dotloss + dotloss1 + dotloss2 + dotloss3 + dotloss4 + dotloss5 + dotloss6 + dotloss7 + dotloss8

    def _seg_loss(self, view, vertsp2d, seg, ix):
        exmat = torch.zeros((cfg.VIEW_NUM, 3, 3))
        trans = torch.zeros((cfg.VIEW_NUM, 1, 3))
        loss = 0

        exmat[view] = torch.FloatTensor(R.from_rotvec(np.array(self.calibration["extrinsic"][str(view)]['rvec']).squeeze()).as_matrix()).cuda()
        trans[view] = torch.FloatTensor(np.reshape(self.calibration["extrinsic"][str(view)]['tvec'], (1, 3))).cuda()
        inmat = torch.FloatTensor(self.calibration['cam_matrix']).cuda()

        a1 = exmat[view, 0, 0] * vertsp2d[0, :, 0] + exmat[view, 0, 1] * vertsp2d[0, :, 1] + exmat[
            view, 0, 2] * vertsp2d[0, :, 2] + trans[view, 0, 0]
        b1 = exmat[view, 1, 0] * vertsp2d[0, :, 0] + exmat[view, 1, 1] * vertsp2d[0, :, 1] + exmat[
            view, 1, 2] * vertsp2d[0, :, 2] + trans[view, 0, 1]
        c1 = exmat[view, 2, 0] * vertsp2d[0, :, 0] + exmat[view, 2, 1] * vertsp2d[0, :, 1] + exmat[
            view, 2, 2] * vertsp2d[0, :, 2] + trans[view, 0, 2]
        vertsp2d[0, :, 0] = a1
        vertsp2d[0, :, 1] = b1
        vertsp2d[0, :, 2] = c1
        D = vertsp2d[0, :, 2].clone().detach()  # -0.5) #* 2.
        vertsp2d[0, :, 0] = vertsp2d[0, :, 0] / D
        vertsp2d[0, :, 1] = vertsp2d[0, :, 1] / D
        vertsp2d[0, :, 2] = vertsp2d[0, :, 2] / D
        a2 = inmat[0, 0] * vertsp2d[0, :, 0] + inmat[0, 1] * vertsp2d[0, :, 1] + inmat[0, 2] * vertsp2d[0, :, 2]
        b2 = inmat[1, 0] * vertsp2d[0, :, 0] + inmat[1, 1] * vertsp2d[0, :, 1] + inmat[1, 2] * vertsp2d[0, :, 2]
        c2 = inmat[2, 0] * vertsp2d[0, :, 0] + inmat[2, 1] * vertsp2d[0, :, 1] + inmat[2, 2] * vertsp2d[0, :, 2]
        vertsp2d[0, :, 0] = a2
        vertsp2d[0, :, 1] = b2
        vertsp2d[0, :, 2] = c2
        vertsp2d[:, :, 0] = (vertsp2d[:, :, 0] / 1200. - 0.5) * 2.
        vertsp2d[:, :, 1] = (vertsp2d[:, :, 1] / 900. - 0.5) * 2.
        seg_pred1 = self.mask_renderer(vertsp2d, self.faces20, self.opt_model.cam)
        seg_gt = torch.Tensor(seg)[None, :, :].to(device)
        loss += torch.nn.MSELoss()(seg_pred1, seg_gt)

        if ix % 10 == 0:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(121)
            ax1.imshow(seg_pred1.cpu().detach().numpy().squeeze())

            ax2 = fig.add_subplot(122)
            ax2.imshow(seg_gt.cpu().detach().numpy().squeeze())
            name = f'/data/cyj/etri/result/{view}th_seg'
            if not os.path.exists(name):
                os.makedirs(name)
            plt.savefig(name+f'/{ix}.png')
            plt.clf()

        return loss

    def save_images(self, i, images):
        exmat = torch.zeros((cfg.VIEW_NUM, 3, 3))
        trans = torch.zeros((cfg.VIEW_NUM, 1, 3))

        for view in range(cfg.VIEW_NUM):
            uvd = np.zeros((21, 3))
            exmat[view] = torch.FloatTensor(
                R.from_rotvec(np.array(self.calibration["extrinsic"][str(view)]['rvec']).squeeze()).as_matrix()).cuda()
            trans[view] = torch.FloatTensor(np.reshape(self.calibration["extrinsic"][str(view)]['tvec'], (1, 3))).cuda()
            inmat = torch.FloatTensor(self.calibration['cam_matrix']).cuda()
            with torch.no_grad():
                skk2d, vertsp, skk3d = self.opt_model.forward(self.mano)
                a1 = exmat[view, 0, 0] * skk2d[0, :, 0] + exmat[view, 0, 1] * skk2d[0, :, 1] + exmat[
                    view, 0, 2] * skk2d[0, :, 2] + trans[view, 0, 0]
                b1 = exmat[view, 1, 0] * skk2d[0, :, 0] + exmat[view, 1, 1] * skk2d[0, :, 1] + exmat[
                    view, 1, 2] * skk2d[0, :, 2] + trans[view, 0, 1]
                c1 = exmat[view, 2, 0] * skk2d[0, :, 0] + exmat[view, 2, 1] * skk2d[0, :, 1] + exmat[
                    view, 2, 2] * skk2d[0, :, 2] + trans[view, 0, 2]
                skk2d[0, :, 0] = a1
                skk2d[0, :, 1] = b1
                skk2d[0, :, 2] = c1
                D = skk2d[0, :, 2].clone().detach()
                skk2d[0, :, 0] = skk2d[0, :, 0] / D
                skk2d[0, :, 1] = skk2d[0, :, 1] / D
                skk2d[0, :, 2] = skk2d[0, :, 2] / D
                a2 = inmat[0, 0] * skk2d[0, :, 0] + inmat[0, 1] * skk2d[0, :, 1] + inmat[0, 2] * skk2d[0, :, 2]
                b2 = inmat[1, 0] * skk2d[0, :, 0] + inmat[1, 1] * skk2d[0, :, 1] + inmat[1, 2] * skk2d[0, :, 2]
                c2 = inmat[2, 0] * skk2d[0, :, 0] + inmat[2, 1] * skk2d[0, :, 1] + inmat[2, 2] * skk2d[0, :, 2]
            uvd[:, 0] = a2.cpu().detach().numpy().squeeze()
            uvd[:, 1] = b2.cpu().detach().numpy().squeeze()
            uvd[:, 2] = D.cpu().detach().numpy().squeeze()
            result_folder = os.path.join(cfg.RESULT_DIR, f"{view}th_view")
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            plotfile = '%s/%d.png' % (result_folder, i)
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.imshow(images[view])
            impoints = np.array(uvd[self.skkidx2, :2])
            impoints = impoints.squeeze()
            plot_hand(impoints, ax1)
            plt.savefig(plotfile)
            plt.clf()