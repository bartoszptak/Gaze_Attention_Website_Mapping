from os import path
from imutils import face_utils
import dlib
import cv2
import numpy as np
import tensorflow as tf
import data.Data_utils as dataUT

class Gaze_Tracker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            path.join('data', 'shape_predictor_68_face_landmarks.dat'))
        self.session = self.create_session(
            path.join('data', 'model', 'inference.pb')
        )
        self.input = self.session.graph.get_tensor_by_name("import/input_1:0")
        self.output = self.session.graph.get_tensor_by_name("import/0_conv_1x1_parts/BiasAdd:0")

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

    @staticmethod
    def create_session(graph_path):
        tf.reset_default_graph()
        session = tf.Session()

        with tf.gfile.GFile(graph_path,'rb') as f:
            graph_def = tf.GraphDef()        
            graph_def.ParseFromString(f.read())
        
        session.graph.as_default()
        tf.import_graph_def(graph_def)
        return session

    def predict(self, image):
        return self.session.run(self.output, feed_dict={self.input: image})[0]   

    def predict_eye(self, image):
        input_image = dataUT.preprocessing(image)
        predicted = self.predict(input_image)
        return dataUT.postprocessing(predicted)

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

        cv2.circle(image, (int(points[6]), int(points[7])), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(points[8]), int(points[9])), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(points[10]), int(points[11])), 1, (0, 0, 255), -1)
        cv2.circle(image, (int(points[12]), int(points[13])), 1, (0, 0, 255), -1)
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
    CAMERA_FLAG = False

    gt = Gaze_Tracker()

    if CAMERA_FLAG:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/videos/video_test_1.mp4')

    while True:
        _, image = cap.read()
        image = cv2.resize(image, (720,405))
        image = cv2.flip(image, +1)

        shape = gt.face_points(image)
        if shape is None:
            continue
        im = gt.search_eye(image, shape)
        pkt = gt.predict_eye(im)

        #cv2.imshow("Face points", gt.test_draw_face_points(image, shape))
        cv2.imshow("Eye points", gt.test_draw_points(im, pkt))

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
