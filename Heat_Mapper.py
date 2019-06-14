import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


class Heat_Mapper:
    def __init__(self, x, y):
        self.map = np.zeros([y, x], np.float32)

    def check_value(self, x, y):
        return int(self.map[y,x] + 1)

    def draw_circle(self, x, y, size, value):
        cv2.circle(self.map, (x, y), size, value, -1)

    def increment_value(self, x, y):
        if y < 0 or y > self.map.shape[0]:
            return
        if x < 0 or x > self.map.shape[1]:
            return 
        self.draw_circle(x, y, 35, self.check_value(x,y))

    def normalize(self):
        m = np.max(self.map)
        self.map = np.multiply(self.map, 255. / m)

    def gaussian_blur(self):
        for i in range(30):
            self.map = cv2.GaussianBlur(self.map, (35,35), 0)

    def get_map(self):
        self.normalize()
        self.gaussian_blur()
        self.map = np.asarray(self.map, dtype=np.uint8)
        return cv2.applyColorMap(self.map, cv2.COLORMAP_HOT)

    def appyling_heatmap(self, img):
        return cv2.addWeighted(img, 0.2, cv2.resize(self.get_map(), (img.shape[1], img.shape[0])), 0.8, 0)


if __name__ == '__main__':
    img = cv2.imread('examples/site.png')
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    hm = Heat_Mapper(img.shape[1]//2, img.shape[0]//2)

    for i in range(255):
        hm.increment_value(125,25)
        hm.increment_value(175,25)

    for i in range(100):
        hm.increment_value(0,0)


    cv2.imshow('Image with map', hm.appyling_heatmap(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
