import torch
import torch.utils.data as data
import numpy as np
import shutil
import time
import random
import os
import math
import json
from PIL import Image
import cv2
import transforms

def read_openpose_list(file_pth):
    _list = []
    with open(file_pth, 'r') as fp:
        line = fp.readline()
        while line:
            _list.append(line.strip())
            line = fp.readline()
    return _list


def read_openpose_json(file_pth):
    """
        filename: JSON file
        return: two list: key_points list and centers list
    """
    fp = open(file_pth)
    data = json.load(fp)
    kpts = []
    centers = []
    scales = []

    for info in data:
        kpt = []
        center = []
        scale = []
        lists = info['info']
        for x in lists:
            kpt.append(x['keypoints'])
            center.append(x['pos'])
            scale.append(x['scale'])
        kpts.append(kpt)
        centers.append(center)
        scales.append(scale)
    fp.close()

    return kpts, centers, scales

def generate_heatmap(heatmap, kpt, stride, sigma):

    height, width, num_point = heatmap.shape
    start = stride / 2.0 - 0.5

    num = len(kpt)
    length = len(kpt[0])
    for i in range(num):
        for j in range(length):
            if kpt[i][j][2] > 1:
                continue
            x = kpt[i][j][0]
            y = kpt[i][j][1]
            for h in range(height):
                for w in range(width):
                    xx = start + w * stride
                    yy = start + h * stride
                    dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
                    if dis > 4.6052:
                        continue
                    heatmap[h][w][j + 1] += math.exp(-dis)
                    if heatmap[h][w][j + 1] > 1:
                        heatmap[h][w][j + 1] = 1

    return heatmap

def generate_vector(vector, cnt, kpts, vec_pair, stride, theta):

    height, width, channel = cnt.shape
    length = len(kpts)

    for j in range(length):
        for i in range(channel):
            a = vec_pair[0][i]
            b = vec_pair[1][i]
            if kpts[j][a][2] > 1 or kpts[j][b][2] > 1:
                continue
            ax = kpts[j][a][0] * 1.0 / stride
            ay = kpts[j][a][1] * 1.0 / stride
            bx = kpts[j][b][0] * 1.0 / stride
            by = kpts[j][b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay

                    dis = abs(bay * px - bax * py)
                    if dis <= theta:
                        vector[h][w][2 * i] = (vector[h][w][2 * i] * cnt[h][w][i] + bax) / (cnt[h][w][i] + 1)
                        vector[h][w][2 * i + 1] = (vector[h][w][2 * i + 1] * cnt[h][w][i] + bay) / (cnt[h][w][i] + 1)
                        cnt[h][w][i] += 1

    return vector

class CocoOpenposeData(data.Dataset):

    def __init__(self, cfg, data_root, info_root, image_set, transformer=None, max_sample=-1):
        self.data_root = data_root
        self.info_root = info_root
        self.image_set = image_set
        self.img_list = read_openpose_list(os.path.join(info_root, '{}_pose.txt'.format(image_set)))
        self.mask_list = read_openpose_list(os.path.join(info_root, '{}_pose_mask.txt'.format(image_set)))
        self.kpt_list, self.center_list, self.scale_list = read_openpose_json(os.path.join(info_root, '{}_pose.json'.format(image_set)))
        self.transformer = transformer
        self.vec_pair = [[2,3,5,6,8,9, 11,12,0,1,1, 1,1,2, 5, 0, 0, 14,15],
                         [3,4,6,7,9,10,12,13,1,8,11,2,5,16,17,14,15,16,17]] # different from openpose
        self.pixel_mean = cfg.pixel_mean
        self.pixel_std = cfg.pixel_std
        self.stride = cfg.OPOSE.stride
        self.theta = cfg.OPOSE.theta
        self.sigma = cfg.OPOSE.sigma

        if 0 < max_sample <= len(self.img_list):
            self.img_list = self.img_list[0:max_sample]
        num_sample = len(self.img_list)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))


    def __getitem__(self, index):

        img_path = os.path.join(self.data_root, self.image_set, self.img_list[index])
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        mask_path = os.path.join(self.info_root, 'pose_mask', self.img_list[index].replace('.jpg', '.npy'))
        mask = np.load(mask_path)
        mask = np.array(mask, dtype=np.float32)

        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        img, mask, kpt, center = self.transformer(img, mask, kpt, center, scale)

        height, width, _ = img.shape

        mask = cv2.resize(mask, (width / self.stride, height / self.stride)).reshape((height / self.stride, width / self.stride, 1))

        heatmap = np.zeros((height / self.stride, width / self.stride, len(kpt[0]) + 1), dtype=np.float32)
        heatmap = generate_heatmap(heatmap, kpt, self.stride, self.sigma)
        heatmap[:,:,0] = 1.0 - np.max(heatmap[:,:,1:], axis=2) # for background
        heatmap = heatmap * mask

        vecmap = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0]) * 2), dtype=np.float32)
        cnt = np.zeros((height / self.stride, width / self.stride, len(self.vec_pair[0])), dtype=np.int32)

        vecmap = generate_vector(vecmap, cnt, kpt, self.vec_pair, self.stride, self.theta)
        vecmap = vecmap * mask

        img = transforms.normalize(transforms.to_tensor(img), [128.0, 128.0, 128.0], [256.0, 256.0, 256.0]) # mean, std
        mask = transforms.to_tensor(mask)
        heatmap = transforms.to_tensor(heatmap)
        vecmap = transforms.to_tensor(vecmap)

        return img, heatmap, vecmap, mask

    def __len__(self):
        return len(self.img_list)
