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
        self.input = self.session.graph.get_tensor_by_name("import/input_2:0")
        self.output = self.session.graph.get_tensor_by_name(
            "import/0_conv_1x1_parts/BiasAdd:0")

        self.size = (128, 90)

        self.L = None
        self.R = None
        self.U = None
        self.D = None
        self.buffer = None
        self.cal = False
        self.cal_points = []

    def reset(self):
        self.L = None
        self.R = None
        self.U = None
        self.D = None
        self.cal = False
        self.buffer = None

    def search_eye(self, image, shape):
        leftXs, leftYs = [], []
        rightXs, rightYs = [], []

        for x, y in shape[36:42]:
            leftXs.append(x)
            leftYs.append(y)
        for x, y in shape[42:48]:
            rightXs.append(x)
            rightYs.append(y)

        left = image[min(leftYs) - 5:max(leftYs) + 5,
                     min(leftXs) - 5:max(leftXs) + 5]

        right = image[min(rightYs) - 5:max(rightYs) + 5,
                      min(rightXs) - 5:max(rightXs) + 5]

        return [left, right]

    def face_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        shape = None

        for (_, rect) in enumerate(rects):
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

        return shape

    @staticmethod
    def create_session(graph_path):
        tf.reset_default_graph()
        session = tf.Session()

        with tf.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        session.graph.as_default()
        tf.import_graph_def(graph_def)
        return session

    def predict(self, image):
        return self.session.run(self.output, feed_dict={self.input: image})

    def predict_eye(self, images):
        input_data = dataUT.preprocessing(images)
        predicted = self.predict(input_data)
        return dataUT.postprocessing(predicted)

    @staticmethod
    def calculate_coordinate(pkts):
        l = np.array(pkts[0], dtype=np.int32)
        r = np.array(pkts[1], dtype=np.int32)

        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return [x, y]
        c_0 = line_intersection(pkts[3:5], pkts[5:7])

        c_1 = pkts[2]
        c = np.mean((c_0, c_1), axis=0, dtype=np.int32)

        cor = np.stack([l, c, r])-l

        def get_angle(point):
            a = np.array((0, 0))
            b = np.array((point[0], 0))
            c = np.array((0, point[1]))

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / \
                (np.linalg.norm(ba) * np.linalg.norm(bc))
            return np.arccos(cosine_angle)*np.sign(-point[1])
        rad = get_angle(cor[2])

        def rotate(x, y, theta):
            xr = np.cos(theta)*x-np.sin(theta)*y
            yr = np.sin(theta)*x+np.cos(theta)*y
            return [xr, np.abs(yr)]
        cor[1] = rotate(*cor[1], rad)
        cor[2] = rotate(*cor[2], rad)

        cor = cor.astype(np.float32)
        cor[1][0] /= cor[2][0]
        cor[1][1] /= cor[2][0]
        return cor[1]

    @staticmethod
    def make_buffer(cor, prev, buff_size=3):
        if prev is None:
            prev = np.stack([cor for _ in range(buff_size)])

        buffer = np.empty((buff_size, 3, 2))

        buffer[:buff_size-1] = prev[1:]
        buffer[buff_size-1] = cor

        prev = buffer

        sw = sum(buffer[i]*(i+1)
                 for i in range(buff_size))/sum(range(buff_size+1))

        return np.array(sw, dtype=np.int32), buffer

    def calculate_position(self, eye_point, res):
        if self.L is None or self.D is None:
            self.L = self.calculate_coordinate(self.cal_points[0][0])
            self.R = self.calculate_coordinate(self.cal_points[1][0])
            self.U = self.calculate_coordinate(self.cal_points[2][0])
            self.D = self.calculate_coordinate(self.cal_points[3][0])

            self.R = np.abs(self.R-self.L)
            self.D = np.abs(self.D-self.U)

        cor = self.calculate_coordinate(eye_point[0])
        cor, self.buffer = self.make_buffer(cor, self.buffer)

        cor[0] = np.abs(cor[0]-self.L)
        cor[1] = np.abs(cor[1]-self.U)

        cor -= 1
        cor = np.arcsin(cor)
        cor += np.pi/2

        return (int(cor[0]*res[0]/np.pi), int(cor[1]*res[1]/np.pi))

    def test_cal_points(self, image):
        image = cv2.flip(image, +1)

        face = self.face_points(image)
        if face is not None:
            eyes = self.search_eye(image, face)
            eyes[1] = cv2.flip(eyes[1], +1)
            _ = self.predict_eye(eyes)

    def add_to_cal_points(self, image):
        image = cv2.flip(image, +1)
        face = self.face_points(image)
        if face is not None:
            eyes = self.search_eye(image, face)
            eyes[1] = cv2.flip(eyes[1], +1)
            pkts = self.predict_eye(eyes)
            self.cal_points.append(pkts)

    def predict_position(self, image, screenX, screenY):
        image = cv2.flip(image, +1)
        face = self.face_points(image)
        if face is not None:
            eyes = self.search_eye(image, face)
            eyes[1] = cv2.flip(eyes[1], +1)
            pkts = self.predict_eye(eyes)
            return self.calculate_position(pkts, (screenX, screenY))
        else:
            return None

    @staticmethod
    def test_draw_points(image, points):
        image = cv2.resize(image, (128, 90))

        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)

        return image

    @staticmethod
    def test_draw_cursor(image, point):
        si = 20
        pX = int(point[0])
        pY = int(point[1])

        cv2.line(image, (pX - si, pY), (pX + si, pY), (0, 0, 255), 3)
        cv2.line(image, (pX, pY - si), (pX, pY + si), (0, 0, 255), 3)
        return image

    @staticmethod
    def test_draw_face_points(image, shape):
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

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
        image = cv2.resize(image, (720, 405))
        image = cv2.flip(image, +1)

        shape = gt.face_points(image)
        if shape is None:
            continue
        eyes = gt.search_eye(image, shape)
        eyes[1] = cv2.flip(eyes[1], +1)

        pkts = gt.predict_eye(eyes)

        l = gt.test_draw_points(eyes[0], pkts[0])
        r = gt.test_draw_points(eyes[1], pkts[1])
        r = cv2.flip(r, +1)

        cv2.imshow("Eyes", np.concatenate((l, r), axis=1))

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
