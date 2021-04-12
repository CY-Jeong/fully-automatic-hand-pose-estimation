'''
    file:   test_SHD.py
    author: zhangxiong(1025679612@qq.com)

    https://github.com/akanazawa/cmr
    Modifiers: Anil Armagan and Seungryul Baek
'''
import matplotlib.pyplot as plt
from pose.model import CPM2DPose
from pose.config import args
from pose import config
import torch

torch.backends.cudnn.benchmark = True

import os
import numpy as np

import torchvision
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda:0'

loading_path = args.loading_path
mano_path = args.mano_path

skkconvert1 = [0, 8, 7, 6, 12, 11, 10, 20, 19, 18, 16, 15, 14, 4, 3, 2, 1, 5, 9, 13, 17]
skkconvert2 = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]

input_size = args.input_size
num_joints = 21

data_test_param = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}

class HMRTrainer(object):
    def __init__(self):
        self.pix_format = 'NCHW'
        self.normalize = True
        self.flip_prob = 0.5
        self.use_flip = False
        self.w_smpl = torch.ones((config.args.batch_size_eval)).float().to(device)
        self._build_model()
        self.images = []
        self.frame = -1

    def _build_model(self):
        print('start building model.')

        # 2d pose estimator
        poseNet = CPM2DPose()
        self.poseNet = poseNet.to(device)
        self.poseNet.eval()
        print('finished build model.')


    def test(self):
        transform0 = torchvision.transforms.ToPILImage()
        transform1 = torchvision.transforms.ToTensor()
        transform2 = torchvision.transforms.Resize((256, 256))
        images = self.images
        image_skeletons = []
        for idx, img in enumerate(images):
            name = idx
            sample_original = transform1(transform2(Image.fromarray(img))).unsqueeze(0)
            sample = sample_original - 0.5

            heatmapsPoseNet = self.poseNet(sample.cuda()).cpu().detach().numpy()
            skeletons_in = np.zeros((1, num_joints, 2))

            for m in range(1):
                for i in range(num_joints):
                    v, u = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                    skeletons_in[m, i, 0] = v * 8
                    skeletons_in[m, i, 1] = u * 8
                # skeletons_in[m, 0, 0] -= skeletons_in[m, 12, 0] - skeletons_in[m, 0, 0]
                # skeletons_in[m, 0, 1] -= skeletons_in[m, 12, 1] - skeletons_in[m, 0, 1]

            image_skeleton = skeletons_in.squeeze().tolist()
            image_skeletons.append(skeletons_in.squeeze().tolist())

            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(111)
            # ax1.imshow(sample_original.squeeze(0).permute(1, 2, 0))
            # plot_hand(skeletons_in[0], ax1)
            #plt.show()

            # from config_opt import cfg
            # result = f"{cfg.RESULT_DIR}/{idx}th_pose"
            # if not os.path.exists(result):
            #     os.makedirs(result)
            # plt.savefig(os.path.join(result, f"0.png"))
            # plt.clf()

        return image_skeletons


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

def main():
    trainer = HMRTrainer()
    trainer.test()

if __name__ == '__main__':
    main()
