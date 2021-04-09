from config_opt import cfg
from segmentation import Segmentation
from utils import *
import manofit
from pose import pose
from projection import Projection
import os
import cv2
import numpy as np
import json


# def check(images, masks, skeletons, boxes, detected_images):
#     for i in range(len(images)):
#         detect = cv2.flip(detected_images[i], 0)
#         width = boxes[i][3] - boxes[i][1]
#         height = boxes[i][2] - boxes[i][0]
#         image = cv2.flip(images[i], 0)
#         mask1 = cv2.flip(masks[i], 0)
#         points = np.array(skeletons[i]).copy()
#         temp = np.array(skeletons[i])
#         points[:, 0] = temp[:, 1] / 256 * width + boxes[i][1]
#         points[:, 1] = temp[:, 0] / 256 * height + boxes[i][0]
#         points[:, 1] = 900 - points[:, 1]
#         import matplotlib.pyplot as plt
#         fig = plt.figure(1)
#         ax1 = fig.add_subplot(131)
#         ax1.imshow(image)
#         impoints = np.array(points)
#         impoints = impoints.squeeze()
#         plot_hand(impoints, ax1)
#
#         # ax2 = fig.add_subplot(132)
#         # ax2.imshow(mask1)
#
#         ax3 = fig.add_subplot(133)
#         ax3.imshow(detect)
#         plt.show()



if __name__=='__main__':
    captures = []
    paths = []
    view_num = cfg.VIEW_NUM

    file_names = os.listdir(cfg.VIDEOS_DIR)
    file_extension = file_names[0].split('.')[-1]
    names = []
    for name in file_names:
        names.append(int(name.split('.')[0]))
    names.sort()
    assert len(names) == view_num, f'There should be only hand videos files in {cfg.VIDEOS_DIR}'
    for i in names:
        paths.append(os.path.join(cfg.VIDEOS_DIR, f'{i}.{file_extension}'))
    for view in range(view_num):
        is_videofile(paths[view])
        captures.append(cv2.VideoCapture(paths[view]))

    minFrame = captures[0].get(cv2.CAP_PROP_FRAME_COUNT)
    for i in range(view_num):
        if minFrame > captures[i].get(cv2.CAP_PROP_FRAME_COUNT):
            minFrame = captures[i].get(cv2.CAP_PROP_FRAME_COUNT)
    calibration = {}
    with open(cfg.INTRINSIC_FILE, 'r') as icf:
        data = json.load(icf)
        calibration['cam_matrix'] = np.array(data['intrinsic'])
        calibration['distortion_coefficients'] = \
            np.array(data['distortion_coefficients']) \
                .reshape(np.size(data['distortion_coefficients']), 1)

    with open(cfg.EXTRINSIC_FILE, 'r') as ecf:
        data = json.load(ecf)
        calibration['extrinsic'] = {}
        for idx in range(cfg.VIEW_NUM):
            calibration['extrinsic'][str(idx)] = np.array(data['extrinsic' + str(idx)])[0]

    i = 0
    seg = Segmentation()
    fitter = manofit.MANOFITTER(calibration)
    while(i < minFrame):
        images = []
        for view in range(view_num):
            ret, frame = captures[view].read()
            if ret == True:
                frame = cv2.resize(frame, (1200, 900))
                if cfg.HAND == 'RIGHT':
                    frame = cv2.flip(frame, cfg.AXIS)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
            else:
                continue
        if len(images) < view_num:
            i = i + 1
            continue
        detected_images, masks, boxes = seg.run(images, cfg.AXIS)
        for c in range(len(images)):
            images[c] = cv2.flip(images[c], cfg.AXIS)
        if i == 0:
            trainer = pose.HMRTrainer()
            trainer.images = detected_images
            trainer.frame = i
            skeletons = trainer.test()
            #check(images, masks, skeletons, boxes, detected_images)
            project_points = Projection(calibration, images, skeletons, boxes)
            points_3d = project_points.run()
            fitter.skel_fit(points_3d)

        fitter.seg_fit(masks)
        fitter.save_images(i, images)
        i = i + 1



