# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import tifffile as tiff
from osgeo import gdal
import cv2
from util.augmentation import RandomFlip, RandomBrightness, RandomNoise, RandomCrop, RandomCropOut #,RandomScratch,RandomBlur,RandomDistortion,RandomRoll
# from util.util import resize_img
import random
from scipy.special import comb


class My_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir, label_seffix, have_label, input_h=512, input_w=512, transform=[]):
        super(My_dataset, self).__init__()

        map_set = []
        dem_set = []
        label_set = []
        maptype_length = len(map_seffix)

        listfile = os.listdir(map_dir)
        for path in listfile:
            if path[(-maptype_length):].upper() != map_seffix.upper():
                continue
            map_set.append(map_dir + path)
            label_set.append(label_dir + path)

        self.map_set = map_set
        self.label_set = label_set
        self.input_h = input_h
        self.input_w = input_w
        self.transform = transform
        self.is_train = have_label
        self.map_seffix = map_seffix
        self.label_seffix = label_seffix
        self.n_data = len(self.map_set)

    def read_image(self, name, folder):
        if folder == 'images':
            image = gdal.Open(name)
            image_wid = image.RasterXSize
            image_hei = image.RasterYSize
            image = image.ReadAsArray(0, 0, image_wid, image_hei)
            return image
        else:
            image = gdal.Open(name)
            image_wid = image.RasterXSize
            image_hei = image.RasterYSize
            image = image.ReadAsArray(0, 0, image_wid, image_hei)
            image[image == 0] = 1
            image -= 1
            return image

    def get_train_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]
        label_name = label_name.replace(self.map_seffix, self.label_seffix)
        image = self.read_image(map_name, 'images')
        label = self.read_image(label_name, 'labels')

        for func in self.transform:
            image, label = func(image, label)
        image = np.array(image, dtype="int32")
        label = np.array(label, dtype="int64")

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        name = self.names[index]
        image = self.read_image(name, 'images')
        return torch.tensor(image), name  

    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return self.n_data

class huanghe_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir, label_seffix, is_index, is_train, resize=False, transform=[]):
        super(huanghe_dataset, self).__init__()

        map_set = []
        label_set = []
        maptype_length = len(map_seffix)

        listfile = os.listdir(map_dir)
        for path in listfile:
            if path[(-maptype_length):].upper() != map_seffix.upper():
                continue
            map_set.append(map_dir + 'img' + path[path.find("_"):])
            label_set.append(label_dir + "label" + path[path.find("_"):])
           
        self.map_set = map_set
        self.label_set = label_set
        self.is_index = is_index
        self.map_seffix = map_seffix
        self.label_seffix = label_seffix
        self.n_data = len(self.map_set)
        self.is_train = is_train
        self.transform = transform
        if self.is_index == True:
            index_name = os.path.dirname(os.path.dirname(os.path.dirname(map_dir))) + "/index.npy"
            self.index_val = np.load(index_name)

    def get_train_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]  
        num = int(name.split('_')[-1].split('.')[0])     
        image = np.load(map_name)
        label = np.load(label_name)
        image.flags.writeable = True
        label.flags.writeable = True
        if self.transform !=None:
            for func in self.transform:
                func = eval(func)
                image, label = func(image, label)
        # image = np.array(image, dtype="int32")
        # label = np.array(label, dtype="int64")
        return torch.tensor(image), torch.tensor(label), num
    
    def get_val_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]
        pslabel_name = self.label_set[index].replace('label', 'pslabel')  
        num = int(name.split('_')[-1].split('.')[0])      
        image = np.load(map_name)
        label = np.load(label_name)
        pslabel = np.load(pslabel_name)
        return torch.tensor(image), torch.tensor(label), torch.tensor(pslabel), num

    def get_item_and_index(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]
        image = np.load(map_name)
        label = np.load(label_name)
        number = int(name.split('_')[1].split('.')[0])
        return torch.tensor(image), torch.tensor(label), name, self.index_val[number]  
    
    def __getitem__(self, index):

        if self.is_index is False:
            if self.is_train is True:
                return self.get_train_item(index)
            else:
                return self.get_val_item(index)
        else:
            return self.get_item_and_index(index)

    def __len__(self):
        return self.n_data
    
    def get_HW(self):
        map = gdal.Open('/mnt/backup/zby/Domain_torch/originaldata/2022/img.tif')
        image_wid = map.RasterXSize + 512
        image_hei = map.RasterYSize + 512
        return image_hei, image_wid 

class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self.background_threshold=background_threshold
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point,end_point=inputs.min(),inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image=self.non_linear_transformation(image, inverse=False)
        image=self.location_scale_transformation(image).astype(np.float32)
        return image

    def Local_Location_Scale_Augmentation(self,image, mask):
        output_image = np.zeros_like(image)

        mask = mask.astype(np.int32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
            mask = np.repeat(mask, image.shape[0], axis=0)

        if (mask==0).sum()!=0:
            output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask==0], inverse=True, inverse_prop=1))

        for c in range(1,np.max(mask)+1):
            if (mask==c).sum()==0:continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))

        if self.background_threshold>=self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]

        return output_image

class slaug_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir, label_seffix, is_index, is_train, resize=False, transform=[]):
        super(slaug_dataset, self).__init__()

        map_set = []
        label_set = []
        maptype_length = len(map_seffix)

        listfile = os.listdir(map_dir)
        for path in listfile:
            if path[(-maptype_length):].upper() != map_seffix.upper():
                continue
            map_set.append(map_dir + 'img' + path[path.find("_"):])
            label_set.append(label_dir + "label" + path[path.find("_"):])
           
        self.map_set = map_set
        self.label_set = label_set
        self.is_index = is_index
        self.map_seffix = map_seffix
        self.label_seffix = label_seffix
        self.n_data = len(self.map_set)
        self.is_train = is_train
        self.transform = transform
        self.location_scale = LocationScaleAugmentation(vrange=(0.,1.), background_threshold=0.01)
        if self.is_index == True:
            index_name = os.path.dirname(os.path.dirname(os.path.dirname(map_dir))) + "/index.npy"
            self.index_val = np.load(index_name)

    def get_train_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]  
        num = int(name.split('_')[-1].split('.')[0])     
        image = np.load(map_name)
        label = np.load(label_name)

        GLA = self.location_scale.Global_Location_Scale_Augmentation(image.copy())
        GLA = np.clip(GLA, 0, 1).astype(image.dtype)
        LLA = self.location_scale.Local_Location_Scale_Augmentation(image.copy(), label.copy().astype(np.int32))
        LLA = np.clip(LLA, 0, 1).astype(image.dtype)

        return torch.tensor(image), torch.tensor(GLA), torch.tensor(LLA), torch.tensor(label), num
    
    def get_val_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]  
        num = int(name.split('_')[-1].split('.')[0])     
        image = np.load(map_name)
        label = np.load(label_name)

        return torch.tensor(image), torch.tensor(label), num

    def get_item_and_index(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]  
        label_name = self.label_set[index]
        image = np.load(map_name)
        label = np.load(label_name)
        number = int(name.split('_')[1].split('.')[0])
        return torch.tensor(image), torch.tensor(label), name, self.index_val[number]  
    
    def __getitem__(self, index):

        if self.is_index is False:
            if self.is_train is True:
                return self.get_train_item(index)
            else:
                return self.get_val_item(index)
        else:
            return self.get_item_and_index(index)

    def __len__(self):
        return self.n_data
    
    def get_HW(self):
        map = gdal.Open('/mnt/backup/zby/Domain_torch/originaldata/2022/img.tif')
        image_wid = map.RasterXSize + 512
        image_hei = map.RasterYSize + 512
        return image_hei, image_wid 
    



if __name__ == '__main__':
    train_map = '/mnt/backup/zby/Domain_torch/data512/experiment2022/val/img/'
    train_label = '/mnt/backup/zby/Domain_torch/data512/experiment2022/val/label/'
    x = huanghe_dataset(map_dir=train_map, map_seffix='.npy', label_dir=train_label, label_seffix='.npy', is_train=True)
    image = x.get_val_item(1)
    index = x.get_index()
    print(image.shape)


