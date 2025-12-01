# Dataset construction based on WHU-OHS
import os
import torch
from torch.utils import data
import numpy as np
from osgeo import gdal
from util.util import resize_img

class WHU_OHS_Dataset(data.Dataset):
    def __init__(self, image_file_list, label_file_list, ori_label, test_ori_label, domain, input_h, input_w, resize, transform=None, use_3D_input=False, channel_last=False):
        self.image_file_list = image_file_list
        self.label_file_list = label_file_list
        self.use_3D_input = use_3D_input
        self.channel_last = channel_last
        self.ori_label = ori_label
        self.test_ori_label = test_ori_label
        self.domain = domain
        self.input_h = input_h
        self.input_w = input_w
        self.resize = resize
        self.transform = transform
    # Statistics of samples of each class in the dataset
    def sample_stat(self):
        sample_per_class = torch.zeros([24])
        for label_file in self.label_file_list:
            label = gdal.Open(label_file, gdal.GA_ReadOnly)
            label = label.ReadAsArray()
            count = np.bincount(label.ravel(), minlength=25)
            count = count[1:25]
            count = torch.tensor(count)
            sample_per_class = sample_per_class + count

        return sample_per_class

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]
        name = os.path.basename(image_file)
        image_dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
        label_dataset = gdal.Open(label_file, gdal.GA_ReadOnly)

        image = image_dataset.ReadAsArray()
        label = label_dataset.ReadAsArray()

        if self.transform:
            for func in self.transform:
                image, label = func(image, label)
        set1 = set(self.ori_label)
        set2 = set(self.test_ori_label)
        common_label = list(set1 & set2)
        if self.domain == 'source':
            non_common_label = list(set1 - set2)
        elif self.domain == 'target':
            non_common_label = list(set2 - set1)
        non_label_map = {label_i: 0 for i, label_i in enumerate(non_common_label)}
        # mapped_labels = np.vectorize(non_label_map.get)(label)
        label_map = {label_i: i for i, label_i in enumerate(common_label)}
        label_map.update(non_label_map)
        mapped_labels = np.vectorize(label_map.get)(label)

        if(self.channel_last):
            image = image.transpose(1, 2, 0)

        # image = image.transpose(2, 0, 1)
        # The image patches were normalized and scaled by 10000 to reduce storage cost
        # image = torch.tensor(image, dtype=torch.float) / 10000.0
        image = torch.tensor(image.astype(float), dtype=torch.float)  / 10000.0

        # img = np.asarray(image, dtype='float32')
        
        # m, n, d = img.shape[0], img.shape[1], img.shape[2]
        # img= img.reshape((m*n,-1))
        # img = img/img.max()
        # img_temp = np.sqrt(np.asarray((img**2).sum(1)))
        # img_temp = np.expand_dims(img_temp,axis=1)
        # img_temp = img_temp.repeat(d,axis=1)
        # img_temp[img_temp==0]=1
        # img = img/img_temp
        # img = np.reshape(img,(m,n,-1))
        # image = torch.tensor(img)
        
        if(self.use_3D_input):
            image = image.unsqueeze(0)

        # 分割的话会不会reshape成其他形状
        if self.resize:
            image = torch.tensor(resize_img(image.numpy(), self.input_h, self.input_w))
            label =  torch.tensor(resize_img(mapped_labels.astype('uint8'), self.input_h, self.input_w))
        else:
            label = torch.tensor(mapped_labels, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long) - 1
        return image[:16,:,:], label, name
        # return image, label, name


