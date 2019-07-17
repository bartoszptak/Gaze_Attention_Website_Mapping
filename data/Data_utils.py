import pandas as pd
import cv2
import numpy as np


class DataGenerator:
    def __init__(self, data_folder, csvfile, inres, outres, nparts, is_train=False):
        self.inres = inres
        self.outres = outres

        self.is_train = is_train
        self.nparts = nparts
        self.data_folder = data_folder

        if is_train:
            df = pd.read_csv(data_folder+csvfile)
            self.anno = df.sample(frac=1).reset_index(drop=True).values
        else:
            self.anno = pd.read_csv(data_folder+csvfile).values

    def __len__(self):
        return len(self.anno)

    def generator(self, batch_size, num_hgstack, sigma=1):
        train_input = np.zeros(
            shape=(batch_size, self.inres[0], self.inres[1], 3), dtype=np.float)
        gt_heatmap = np.zeros(
            shape=(batch_size, self.outres[0], self.outres[1], self.nparts), dtype=np.float)

        while True:
            for i, row in enumerate(self.anno):
                _image, _gthtmap = self.process_image(row, sigma)
                _index = i % batch_size

                train_input[_index, :, :, :] = _image
                gt_heatmap[_index, :, :, :] = _gthtmap

                if i % batch_size == (batch_size-1):
                    out_hmaps = []
                    for m in range(num_hgstack):
                        out_hmaps.append(gt_heatmap)

                    yield train_input, out_hmaps

    def process_image(self, row, sigma):
        image = cv2.imread(self.data_folder+row[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        y, x, _ = image.shape
        image = cv2.resize(image, (128, 90))
        joins = extend_joins(row[1:])
        joins = [int(l*128/x) if i % 2 == 0 else int(l*90/y)
                 for i, l in enumerate(joins)]

        if self.is_train and np.random.choice(2, 1)[0] == 0:
            image, joins = flip(image, joins)

        image = normalize(image)
        joins, shape = transform_joins(joins, image.shape, self.outres)
        gtmap = generate_gtmap(joins, sigma, shape)

        _image = np.zeros(
            shape=(self.inres[0], self.inres[1], 3), dtype=np.float)

        _gthtmap = np.zeros(
            shape=(self.outres[0], self.outres[1], self.nparts), dtype=np.float)

        y2, x2, _ = gtmap.shape

        y, x, _ = image.shape
        _image[:y, :x, :] = image

        y, x, _ = gtmap.shape
        _gthtmap[:y, :x, :] = gtmap

        return _image, _gthtmap


def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 90))
    image = normalize(image)

    _image = np.zeros(shape=(1, 128, 128, 3), dtype=np.float)

    y, x, _ = image.shape
    _image[0, :y, :x, :] = image

    return _image

def postprocessing(heatmap):
    heatmap =  np.transpose(heatmap, (2,0,1))
    masks = np.zeros((10, 2))
    for j, hm in enumerate(heatmap):            
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(hm)            
        masks[j] = maxLoc

    masks = np.multiply(masks, 4)
    z = masks.tolist()
    return [z[0][0],z[0][1], z[2][0],z[2][1], z[4][0], z[4][1]]

def extend_joins(joins):
    return [joins[0], joins[1], joins[0]+1, joins[1]+1,
            joins[2], joins[3], joins[2]+1, joins[3]+1,
            joins[4], joins[5], joins[4]+1, joins[5]+1,
            joins[6], joins[7], joins[8], joins[9], joins[10], joins[11], joins[12], joins[13]
            ]


def normalize(imgdata):
    imgdata = imgdata / 255.0
    color_mean = np.array([0.4404, 0.4440, 0.4327], dtype=np.float)

    for i in range(imgdata.shape[-1]):
        imgdata[:, :, i] -= color_mean[i]

    return imgdata


def flip(image, joins):
    flipimage = cv2.flip(image, flipCode=1)
    joints = [128-l if i % 2 == 0 else l for i, l in enumerate(joins)]

    return flipimage, joints


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def transform_joins(joins, shape, outres):
    f = outres[1]/shape[1]
    return [int(x*f) for x in joins], (int(shape[0]*f), int(shape[1]*f))


def generate_gtmap(joints, sigma, outres):
    npart = len(joints)//2
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)

    for i in range(npart):
        gtmap[:, :, i] = draw_labelmap(
            gtmap[:, :, i], (joints[2*i], joints[2*i+1],), sigma)

    return gtmap
