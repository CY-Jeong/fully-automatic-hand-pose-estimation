import numpy as np
import cv2
import math

class Projection(object):
    def __init__(self, calibrations, images, image_points, boxes):
        self.calibrations = calibrations
        self.projection_matrices = {}
        self.projection_matrices["extrinsic"] = {
            str(idx): self.projection_matrix(np.squeeze(calibrations["extrinsic"][str(idx)]['rvec']), \
                                             np.squeeze(calibrations["extrinsic"][str(idx)]['tvec']).reshape(3, 1))
            for idx in range(len(images))}
        self.images = images
        self.boxes = boxes
        self.len = len(self.images)
        self.len_points = np.array(image_points).shape[1]
        self.image_points = np.array(image_points)
        self.refined_2d = []

    def make_projections(self, camera_matrices, projection_matrices):
        projections = []
        for projection_matrix in projection_matrices:
            projections.append(np.matrix(np.matrix(camera_matrices) * np.matrix(projection_matrix)))

        return projections
    def projection_matrix(self, rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection = np.hstack((rotation_matrix, tvec))
        return projection

    def get_3d_points(self, projections, points, points_count):
        matrA = np.zeros((3 * len(projections), 4))

        for view, proj in enumerate(projections):
            x = points[view][points_count][0]
            y = points[view][points_count][1]
            for val in range(4):
                matrA[view * 3 + 0, val] = x * proj[2, val] - proj[0, val]
                matrA[view * 3 + 1, val] = y * proj[2, val] - proj[1, val]
                matrA[view * 3 + 2, val] = x * proj[1, val] - y * proj[0, val]
        _, _, v = cv2.SVDecomp(matrA)
        point3d = v[3, :]
        point3d = point3d / point3d[3]
        return (point3d[0], point3d[1], point3d[2])

    def make_3d_points(self):
        cam_matrix = self.calibrations['cam_matrix']
        point3ds = np.zeros((self.len_points, 3))
        ob3dpoint = np.zeros((self.len, self.len, self.len_points, 3))
        cal = self.calibrations
        errors = np.ones((self.len, self.len, 1))*10000
        for i in range(self.len-1):
            for j in range(1, self.len - i):
                if np.sum(self.image_points[i]) == 0:
                    break
                elif np.sum(self.image_points[i + j]) == 0:
                    continue
                projection = []
                projection.append(self.projection_matrices["extrinsic"][str(i)])
                projection.append(self.projection_matrices["extrinsic"][str(i + j)])
                projections = self.make_projections(cam_matrix, projection)

                image_points = []
                image_points.append(self.image_points[i])
                image_points.append(self.image_points[i + j])

                for c in range(len(image_points[0])):
                    point3d = self.get_3d_points(projections, image_points, c)
                    point3ds[c] = point3d
                opoint = np.array(point3ds)
                ob3dpoint[i][i + j] = opoint
                error = 0
                for k in range(self.len_points):
                    impoint1, _ = cv2.projectPoints(opoint[k],
                                                    np.reshape(cal["extrinsic"][str(i)]['rvec'], (3, 1)),
                                                    np.reshape(cal["extrinsic"][str(i)]['tvec'], (3, 1)),
                                                    cal['cam_matrix'],
                                                    cal['distortion_coefficients'].reshape(1, 5))
                    impoint2, _ = cv2.projectPoints(opoint[k],
                                                    np.reshape(cal["extrinsic"][str(i + j)]['rvec'],
                                                               (3, 1)),
                                                    np.reshape(cal["extrinsic"][str(i + j)]['tvec'],
                                                               (3, 1)),
                                                    cal['cam_matrix'],
                                                    cal['distortion_coefficients'].reshape(1, 5))
                    e1 = math.sqrt(pow(impoint1.squeeze()[0] - self.image_points[i][k][0], 2) +
                                   pow(impoint1.squeeze()[1] - self.image_points[i][k][1], 2))
                    e2 = math.sqrt(pow(impoint2.squeeze()[0] - self.image_points[i + j][k][0], 2) +
                                   pow(impoint2.squeeze()[1] - self.image_points[i + j][k][1], 2))
                    error = error + e1 + e2
                errors[i][i + j] = error
        camera1, camera2 = np.argmin(errors) // errors.shape[1], np.argmin(errors) % errors.shape[1]
        print(f"camera {camera1}, {camera2}")
        errors[camera1, camera2] = 100000
        camera3, camera4 = np.argmin(errors) // errors.shape[1], np.argmin(errors) % errors.shape[1]
        print(f"camera {camera3}, {camera4}")
        errors[camera3, camera4] = 100000
        camera5, camera6 = np.argmin(errors) // errors.shape[1], np.argmin(errors) % errors.shape[1]
        print(f"camera {camera5}, {camera6}")
        point3ds = []
        point3ds.append(ob3dpoint[camera1, camera2])
        point3ds.append(ob3dpoint[camera3, camera4])
        point3ds.append(ob3dpoint[camera5, camera6])
        point3d = np.zeros((self.len_points, 3))
        for c in range(21):
            for i in range(3):
                mean = np.mean(np.array(point3ds)[:, c, i])
                point3d[c][i] = mean
        self.point3d = np.array(point3d)

        ###
        for i in range(15):
            for k in range(self.len_points):
                impoint1, _ = cv2.projectPoints(self.point3d,
                                                np.reshape(cal["extrinsic"][str(i)]['rvec'], (3, 1)),
                                                np.reshape(cal["extrinsic"][str(i)]['tvec'], (3, 1)),
                                                cal['cam_matrix'],
                                                cal['distortion_coefficients'].reshape(1, 5))
            import matplotlib.pyplot as plt
            from utils import plot_hand
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.imshow(self.images[i])
            plot_hand(impoint1.squeeze(), ax)
            #plt.show()
            plt.clf()

    def project2d(self):
        for camera in range(self.len):
            cal = self.calibrations
            if self.point3d is not None:
                opoint = np.array((self.point3d))
                impoints = []
                for idx in range(len(opoint)):
                    impoint, _ = cv2.projectPoints(opoint[idx], np.reshape(cal["extrinsic"][str(camera)]['rvec'], (3, 1)), \
                                                   np.reshape(cal["extrinsic"][str(camera)]['tvec'], (3, 1)),
                                                   cal['cam_matrix'],
                                                   cal['distortion_coefficients'].reshape(1, 5))
                    impoints.append(impoint)

                self.refined_2d.append(impoints)


    def run(self):
        self._resize_points()
        self.make_3d_points()
        return self.point3d

    def _proj_matrix(rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection = np.hstack((rotation_matrix, tvec))
        return projection

    def _resize_points(self, axis=0):
        size = self.images[0].shape[axis]

        for i in range(self.len):
            width = self.boxes[i][3] - self.boxes[i][1]
            height = self.boxes[i][2] - self.boxes[i][0]
            skeletons = np.array(self.image_points[i]).copy()
            self.image_points[i][:, 1] = skeletons[:, 0] / 256 * height + self.boxes[i][0]
            self.image_points[i][:, 0] = skeletons[:, 1] / 256 * width + self.boxes[i][1]
            self.image_points[i][:, abs(axis-1)] = size - self.image_points[i][:, abs(axis-1)]
