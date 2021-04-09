import numpy as np
import cv2

from config_opt import cfg
class Segmentation():

    def __init__(self):
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.boxes = []

    def run(self, images, axis):
        masks = []
        boxes = []
        mask = np.zeros(images[0].shape[:2], np.uint8)
        w, h = images[0].shape[:2]
        detected_images = []
        for i in range(len(images)):
            image = images[i]
            mask[:, :] = 3
            for u in range(w):
                for v in range(h):
                    if image[u, v, 0] < image[u, v, 1] and image[u, v, 2] < image[u, v, 1]:
                        mask[u, v] = 0
            cv2.grabCut(image, mask, None, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            img = image * mask2[:, :, np.newaxis]

            grimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grimg = cv2.threshold(grimg, 5, 255, cv2.THRESH_BINARY)[1]
            num_labels, labels = cv2.connectedComponents(grimg)
            max = 0
            index = 0
            for n in range(1, num_labels):
                tmax = len(np.where(labels == n)[0])
                if max < tmax:
                    max = tmax
                    index = n
            labels = np.where(labels == index, 1, 0)
            mask_orig = np.uint8(np.copy(labels))

            ymin = np.where(labels > 0)[0].min()
            ymax = np.where(labels > 0)[0].max()
            xmin = np.where(labels > 0)[1].min()
            xmax = np.where(labels > 0)[1].max()
            width = xmax - xmin
            height = ymax - ymin
            img = image * labels[:, :, None]

            xmin = np.int32(xmin - width * 0.5)
            xmax = np.int32(xmax + width * 0.5)
            ymin = np.int32(ymin - height * 0.5)
            ymax = np.int32(ymax + height * 0.5)
            if xmin < 0:
                xmin = 0
            if xmax > 1199:
                xmax = 1199
            if ymin < 0:
                ymin = 0
            if ymax > 899:
                ymax = 899
            detected_images.append(image[ymin:ymax, xmin:xmax, :])
            nseg = cv2.resize(mask_orig, (cfg.SEG_SIZE, cfg.SEG_SIZE))
            nseg = cv2.flip(nseg, axis)
            masks.append(nseg)
            boxes.append([ymin, xmin, ymax, xmax])
        return detected_images, masks, boxes