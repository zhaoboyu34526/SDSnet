# coding:utf-8
import numpy as np
from PIL import Image


class RandomFlip():
    def __init__(self, prob=0.5):
        super(RandomFlip, self).__init__()
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:, :, ::-1].copy()
            label = label[:, ::-1].copy()
        if np.random.rand() < self.prob:
            image = image[:, ::-1, :].copy()
            label = label[::-1, :].copy()
        return image, label


class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        super(RandomCrop, self).__init__()
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # w, h, c = image.shape
            c,w,h = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[:, w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label


class RandomCropOut():  # 两种 crop 的 不同效果？
    def __init__(self, crop_rate=0.2, prob=1.0):
        super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # w, h, c = image.shape
            c, w, h = image.shape

            crop_windows_h = np.random.randint(0, h*self.crop_rate)
            crop_windows_w = np.random.randint(0, w*self.crop_rate)

            h1 = np.random.randint(0, h-crop_windows_h-1)
            w1 = np.random.randint(0, w-crop_windows_w-1)

            image[:,w1:w1+crop_windows_w, h1:h1+crop_windows_h] = 0
            label[w1:w1+crop_windows_w, h1:h1+crop_windows_h] = 0

        return image, label


class RandomBrightness():
    def __init__(self, bright_range=0.15, prob=0.9):
        super(RandomBrightness, self).__init__()
        self.bright_range = bright_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            # 为归一化后的数据计算亮度调整因子
            bright_factor = np.random.uniform(
                1 - self.bright_range, 1 + self.bright_range)
            new_image = image * bright_factor
            
            # 确保调整后的数据仍然在[0, 1]的范围内
            new_image = np.clip(new_image, 0, 1)
            
            image = new_image.astype(image.dtype)

        return image, label


class RandomNoise():
    def __init__(self, noise_range=0.05, prob=0.9):  # 调整默认的噪声范围为0.05
        super(RandomNoise, self).__init__()
        self.noise_range = noise_range
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            c, h, w = image.shape
            # 生成与图像形状相同的噪声数组
            noise = np.random.uniform(
                -self.noise_range,
                self.noise_range,
                (c, h, w)
            )

            # 添加噪声并确保结果仍然在[0, 1]范围内
            image = np.clip(image + noise, 0, 1).astype(image.dtype)

        return image, label
