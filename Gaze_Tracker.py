from os import path
from Model import Model
from imutils import face_utils
import dlib
import cv2
import numpy as np


class Gaze_Tracker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            path.join('data', 'shape_predictor_68_face_landmarks.dat'))
        #self.model = Model()
        # self.model.load('data/')
        self.size = (144, 90)

        self.Lx = None
        self.Rx = None
        self.Uy = None
        self.Dy = None
        self.cal = False
        self.cal_points = []

    def reset(self):
        self.Lx = None
        self.Rx = None
        self.Uy = None
        self.Dy = None
        self.cal = False

    def search_eye(self, image, shape):
        leftXs, leftYs = [], []

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)

        left = image[min(leftYs) - 5:max(leftYs) + 5,
                     min(leftXs) - 5:max(leftXs) + 5]
        return cv2.resize(left, self.size)

    def face_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        shape = None

        for (i, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

        return shape

    @staticmethod
    def test_draw_face_points(image, shape):
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        return image

    def predict_eye(self, image):
        image = cv2.resize(image, self.size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = gray * 1. / 255
        Zz = np.expand_dims(gray, axis=0)
        Z = Zz[..., np.newaxis]
        result = self.model.predict(Z)
        return [result[0][0], result[0][1],
                result[0][2], result[0][3],
                result[0][4], result[0][5]]

    @staticmethod
    def calculate_coordinate(p):
        p3 = np.array(p[0:2])
        p2 = np.array(p[2:4])
        p1 = np.array(p[4:6])

        c = np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1]))
        y = np.cross(p3 - p1, p2 - p1) / np.linalg.norm(p3 - p1)
        x = np.sqrt(np.square(c) - np.square(y))

        return [x, y]

    def calculate_position(self, eye_point, res):
        if self.Lx is None or self.Dy is None:
            self.Lx = self.calculate_coordinate(self.cal_points[0])[0]
            self.Rx = self.calculate_coordinate(self.cal_points[1])[0]
            self.Uy = self.calculate_coordinate(self.cal_points[2])[1]
            self.Dy = self.calculate_coordinate(self.cal_points[3])[1]

        Cx, Cy = self.calculate_coordinate(eye_point)

        return [int((Cx - self.Lx) / (self.Rx - self.Lx) * res[0]), int((Cy - self.Uy) / (self.Dy - self.Uy) * res[1])]

    def test_cal_points(self, image):
        image = cv2.flip(image, +1)

        face = self.face_points(image)
        if face is not None:
            im = self.search_eye(image, face)
            eye = self.predict_eye(im)

    def add_to_cal_points(self, image):
        image = cv2.flip(image, +1)
        face = self.face_points(image)
        if face is not None:
            im = self.search_eye(image, face)
            eye = self.predict_eye(im)
            self.cal_points.append(eye)

    def predict_position(self, image, screenX, screenY):
        image = cv2.flip(image, +1)
        face = self.face_points(image)
        if face is not None:
            im = self.search_eye(image, face)
            eye = self.predict_eye(im)
            return self.calculate_position(eye, (screenX, screenY))
        else:
            return None

    @staticmethod
    def test_draw_points(image, points):
        cv2.circle(image, (int(points[0]), int(points[1])), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(points[2]), int(points[3])), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(points[4]), int(points[5])), 1, (0, 0, 255), -1)
        return image

    @staticmethod
    def test_draw_cursor(image, point):
        si = 20
        pX = int(point[0])
        pY = int(point[1])

        cv2.line(image, (pX - si, pY), (pX + si, pY), (0, 0, 255), 3)
        cv2.line(image, (pX, pY - si), (pX, pY + si), (0, 0, 255), 3)
        return image


if __name__ == '__main__':
    gt = Gaze_Tracker()
    cap = cv2.VideoCapture(0)

    while True:
        _, image = cap.read()
        image = cv2.flip(image, +1)

        shape = gt.face_points(image)
        im = gt.search_eye(image, shape)
        pkt = gt.predict_eye(im)

        cv2.imshow("Face points", gt.test_draw_face_points(image, shape))
        cv2.imshow("Eye points", gt.test_draw_points(im, pkt))

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
