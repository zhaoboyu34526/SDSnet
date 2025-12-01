import random

import numpy as np
import tifffile as tiff
import os
import scipy.io as sio
from scipy.cluster.vq import whiten
import sys

import torch
sys.path.append(".")
from config import variables
import cv2
from PIL import Image
from PIL import Image
# from osgeo import gdal

def read_image(file_name, data_name, data_type):
    mdata = []
    if data_type == 'tif':
        mdata = tiff.imread(file_name)
        return mdata
    if data_type == 'mat':
        mdata = sio.loadmat(file_name)
        mdata = np.array(mdata.get(data_name))
    if data_type == 'png':
        mdata = cv2.imread(file_name)
    if data_type == 'tiff':
        mdata = tiff.imread(file_name)
    return mdata

def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)

def minmax_normalize(array):
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def minmax_normalize_nozero(array):
    amax = np.max(array)
    array_no_zeros = np.where(array==0, amax+1, array)
    amin = np.min(array_no_zeros)
    array_aux = (array - amin) / (amax - amin)
    array_norm = np.where(array_aux>1, 0, array_aux)   
    return array_norm

def reverse_minmax_normalize_nozero(array_norm, amin, amax):
    array = array_norm * (amax - amin) + amin
    array = np.where(array_norm == 1, 0, array)
    return array

def create_val():
    # 我们在for循环中，将i，j作为坐标
    listfile = os.listdir(variables.PATH_ALL)  # 将路径中的文件名以字符串的形式排列并按照顺序填入列表listfile
    listfile.sort()
    c = 0
    Z = []
    # 将前三年的map和label加载进列表X，Y
    for x in range(len(listfile)):  # 我们取前几个作为训练集，最后的2020年作为测试集
        if (listfile[x] == "2022"):
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tif")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          
            Spartina_label = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.label_name), "mask_test",
                                        "tif")
            Spartina_label = Spartina_label[:, :, 0]
            shape = (6800, 7200)
            start_row = (Spartina_label.shape[0] - shape[0]) // 2
            end_row = start_row + shape[0]
            start_col = (Spartina_label.shape[1] - shape[1]) // 2
            end_col = start_col + shape[1]
            Spartina_map = Spartina_map[:,start_row:end_row, start_col:end_col]
            Spartina_label = Spartina_label[start_row:end_row, start_col:end_col]
            print('done read!')                      
            Spartina_map = np.pad(Spartina_map, ((0, 0), (variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')
            Spartina_label = np.pad(Spartina_label, ((variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')

            Spartine_label_zero = Spartina_map[0].copy()
            Spartine_label_zero[Spartine_label_zero != 0] = 1
    
            Spartina_map = minmax_normalize_nozero(Spartina_map)
            for i in range(0, np.size(Spartina_map, 1), variables.ksize):
                for j in range(0, np.size(Spartina_map, 2), variables.ksize):
                    if (i + variables.ksize > np.size(Spartina_map, 1)):
                        break
                    elif (j + variables.ksize > np.size(Spartina_map, 2)):
                        break
                    if (Spartine_label_zero[i:i + variables.ksize, j:j + variables.ksize].sum() == 0):
                        continue
                    tmp_map = Spartina_map[:, i:i + variables.ksize, j:j + variables.ksize]
                    tmp_label = Spartina_label[i:i + variables.ksize, j:j + variables.ksize]

                    imgname = '/mnt/backup/zby/MARNetadd/experiment2022_128/val/img/img_{}.npy'.format(c)
                    labelname = '/mnt/backup/zby/MARNetadd/experiment2022_128/val/label/label_{}.npy'.format(c)
                    done = "done_{}".format(c)
                    np.save(imgname, tmp_map)
                    np.save(labelname, tmp_label)  
                    Z.append((i, j))
                    print(done)
                    c = c + 1
            print("done cut!")
    Z = np.array(Z)
    np.save('/mnt/backup/zby/MARNetadd/experiment2022_128/val/index.npy', Z)
    # X = np.array(X)
    # Y = np.array(Y)
    # Z = np.array(Z)
    
    # np.save('/mnt/backup/zby/MARNetadd/data/data512/experiment2022/val_img.npy', X)
    # np.save('/mnt/backup/zby/MARNetadd/data/data512/experiment2022/val_label.npy', Y)
    # np.save('/mnt/backup/zby/MARNetadd/data/data512/experiment2022/val_index.npy', Z)

def create_train():
    # 我们在for循环中，将i，j作为坐标
    listfile = os.listdir(variables.PATH_ALL)  # 将路径中的文件名以字符串的形式排列并按照顺序填入列表listfile
    listfile.sort()
    c = 0
    # 将前三年的map和label加载进列表X，Y
    for x in range(len(listfile)):  # 我们取前几个作为训练集，最后的2020年作为测试集
        if (listfile[x] != "e"):  
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tif")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          
            Spartina_label = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.label_name), "mask_test",
                                        "tif")
            # unique, counts = np.unique(Spartina_label, return_counts=True)
            # print("Label values and counts:", dict(zip(unique, counts)))
            # print('done read!')

            zero_mask = np.all(Spartina_map == 0, axis=0)
            zero_mask = zero_mask.astype(np.uint8) * 255
            Spartina_label = np.where(zero_mask == 255, 255, Spartina_label)
            unique, counts = np.unique(Spartina_label, return_counts=True)
            print("Label values and counts:", dict(zip(unique, counts)))
            
            Spartina_map = np.pad(Spartina_map, ((0, 0), (variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')
            Spartina_label = np.pad(Spartina_label, ((variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')

            Spartine_label_zero = Spartina_map[0].copy()
            Spartine_label_zero[Spartine_label_zero != 0] = 1

            Spartina_map = minmax_normalize_nozero(Spartina_map)
            
            for i in range(0, np.size(Spartina_map, 1), variables.r):
                for j in range(0, np.size(Spartina_map, 2), variables.r):
                    if (i + variables.ksize > np.size(Spartina_map, 1)):
                        break
                    elif (j + variables.ksize > np.size(Spartina_map, 2)):
                        break
                    if (Spartine_label_zero[i:i + variables.ksize, j:j + variables.ksize].sum() == 0):
                        continue
                    tmp_map = Spartina_map[:, i:i + variables.ksize, j:j + variables.ksize]
                    tmp_label = Spartina_label[i:i + variables.ksize, j:j + variables.ksize]

                    imgname = '/mnt/backup/zby/MARNetadd/yellowriver/img/img_{}.npy'.format(c)
                    labelname = '/mnt/backup/zby/MARNetadd/yellowriver/label/label_{}.npy'.format(c)
                    done = "done_{}".format(c)
                    np.save(imgname, tmp_map)
                    np.save(labelname, tmp_label)     
                    print(done)
                    c = c + 1            
            print("done cut!")

def create_train_add():
    # 我们在for循环中，将i，j作为坐标
    listfile = os.listdir(variables.PATH_ALL)  # 将路径中的文件名以字符串的形式排列并按照顺序填入列表listfile
    listfile.sort()
    c = 0
    Z = []
    # 将前三年的map和label加载进列表X，Y
    for x in range(len(listfile)):  # 我们取前几个作为训练集，最后的2020年作为测试集
        # if (listfile[x] == "b" or listfile[x] == "d" or listfile[x] == "e"): 
        # if (listfile[x] == "a" or listfile[x] == "c" or listfile[x] == "f"):  
        if (listfile[x] == "e"):  
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tif")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          
            Spartina_label = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.label_name), "mask_test", "png")
            Spartina_label = Spartina_label[:, :, 0]
            shape = (6800, 7200)
            start_row = (Spartina_label.shape[0] - shape[0]) // 2
            end_row = start_row + shape[0]
            start_col = (Spartina_label.shape[1] - shape[1]) // 2
            end_col = start_col + shape[1]
            Spartina_map = Spartina_map[:,start_row:end_row, start_col:end_col]
            Spartina_label = Spartina_label[start_row:end_row, start_col:end_col]

            print('done read!')
            Spartina_map = np.pad(Spartina_map, ((0, 0), (variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')
            Spartina_label = np.pad(Spartina_label, ((variables.pad, variables.pad), (variables.pad, variables.pad)), 'symmetric')

            Spartine_label_zero = Spartina_map[0].copy()
            Spartine_label_zero[Spartine_label_zero != 0] = 1

            Spartina_map = minmax_normalize_nozero(Spartina_map)
            
            for i in range(0, np.size(Spartina_map, 1), variables.r):
                for j in range(0, np.size(Spartina_map, 2), variables.r):
                    if (i + variables.ksize > np.size(Spartina_map, 1)):
                        break
                    elif (j + variables.ksize > np.size(Spartina_map, 2)):
                        break
                    if (Spartine_label_zero[i:i + variables.ksize, j:j + variables.ksize].sum() == 0):
                        continue
                    tmp_map = Spartina_map[:, i:i + variables.ksize, j:j + variables.ksize]
                    tmp_label = Spartina_label[i:i + variables.ksize, j:j + variables.ksize]

                    imgname = '/mnt/backup/zby/MARNetadd/experimentadd256/val_e/img/img_{}.npy'.format(c)
                    labelname = '/mnt/backup/zby/MARNetadd/experimentadd256/val_e/label/label_{}.npy'.format(c)
                    done = "done_{}".format(c)
                    np.save(imgname, tmp_map)
                    np.save(labelname, tmp_label)
                    print(done)
                    Z.append((i, j))
                    print(done)
                    c = c + 1
            print("done cut!")
    Z = np.array(Z)
    np.save('/mnt/backup/zby/MARNetadd/experimentadd256/index.npy', Z)
    # ID = np.random.permutation(np.arange(len(X)))
    # data = list(zip(X, Y))

    # # 随机打乱data列表
    # random.shuffle(data)

    # # 解压打乱后的data列表，得到新的X和Y
    # X, Y = zip(*data)

    # X = np.array(X)
    # Y = np.array(Y)
    # print("done shuffle!")
    # np.save('/mnt/backup/zby/MARNetadd/data/data512/experiment2022/train_img.npy', X)
    # np.save('/mnt/backup/zby/MARNetadd/data/data512/experiment2022/train_label.npy', Y)
    # hdf5storage.savemat("/mnt/backup/zby/MARNetadd/data/data512/experiment2021/train.mat", {'img':X, 'label':Y}, matlab_compatible=True)


def num_cal():
    # 我们在for循环中，将i，j作为坐标
    listfile = os.listdir(variables.PATH_ALL)  # 将路径中的文件名以字符串的形式排列并按照顺序填入列表listfile
    listfile.sort()
    c = 0
    # 将前三年的map和label加载进列表X，Y
    train_0 = 0
    train_1 = 0
    train_2 = 0
    train_3 = 0
    train_4 = 0
    train_5 = 0
    val_0 = 0
    val_1 = 0
    val_2 = 0
    val_3 = 0
    val_4 = 0
    val_5 = 0
    for x in range(len(listfile)):  # 我们取前几个作为训练集，最后的2020年作为测试集
        if (listfile[x] == "a" or listfile[x] == "c" or listfile[x] == "f"):  
        # if (listfile[x] == "a"):  
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tiff")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          
            Spartina_label = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.label_name), "mask_test",
                                        "png")
            Spartina_label = Spartina_label[:, :, 0]
            shape = (6800, 7200)
            start_row = (Spartina_label.shape[0] - shape[0]) // 2
            end_row = start_row + shape[0]
            start_col = (Spartina_label.shape[1] - shape[1]) // 2
            end_col = start_col + shape[1]
            Spartina_map = Spartina_map[:,start_row:end_row, start_col:end_col]
            Spartina_label = Spartina_label[start_row:end_row, start_col:end_col]
            train_0 += (Spartina_label == 0).sum()
            train_1 += (Spartina_label == 1).sum()
            train_2 += (Spartina_label == 2).sum()
            train_3 += (Spartina_label == 3).sum()
            train_4 += (Spartina_label == 4).sum()
            train_5 += (Spartina_label == 5).sum()
        if (listfile[x] == "b" or listfile[x] == "d" or listfile[x] == "e"):  
        # if (listfile[x] == "a"):  
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tiff")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          
            Spartina_label = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.label_name), "mask_test",
                                        "png")
            Spartina_label = Spartina_label[:, :, 0]
            shape = (6800, 7200)
            start_row = (Spartina_label.shape[0] - shape[0]) // 2
            end_row = start_row + shape[0]
            start_col = (Spartina_label.shape[1] - shape[1]) // 2
            end_col = start_col + shape[1]
            Spartina_map = Spartina_map[:,start_row:end_row, start_col:end_col]
            Spartina_label = Spartina_label[start_row:end_row, start_col:end_col]
            val_0 += (Spartina_label == 0).sum()
            val_1 += (Spartina_label == 1).sum()
            val_2 += (Spartina_label == 2).sum()
            val_3 += (Spartina_label == 3).sum()
            val_4 += (Spartina_label == 4).sum()
            val_5 += (Spartina_label == 5).sum()
    print(train_0, train_1, train_2, train_3, train_4, train_5, '\n')
    print(val_0, val_1, val_2, val_3, val_4, val_5)


def mean():
    # 我们在for循环中，将i，j作为坐标
    listfile = os.listdir(variables.PATH_ALL)  # 将路径中的文件名以字符串的形式排列并按照顺序填入列表listfile
    listfile.sort()
    c = 0
    mean, var = [], []
    # 将前三年的map和label加载进列表X，Y
    for x in range(len(listfile)):  # 我们取前几个作为训练集，最后的2020年作为测试集
        if (listfile[x] != "2022"):
            listfile[x] = listfile[x] + "/"
            Spartina_map = read_image(os.path.join(variables.PATH_ALL, listfile[x], variables.map_name), "data", "tif")
            if (Spartina_map.shape[2] < Spartina_map.shape[0]):
                Spartina_map = Spartina_map.transpose(2, 0, 1)          

            Spartina_map = minmax_normalize_nozero(Spartina_map)
            Spartina_map = torch.tensor(Spartina_map)
            channel_mean = torch.mean(Spartina_map.view(4, -1), dim=1)
            channel_var = torch.var(Spartina_map.view(4, -1), dim=1, unbiased=False)
            mean.append(channel_mean)
            var.append(channel_var)
    total_mean = torch.stack(mean).mean(0)
    total_var = torch.stack(var).mean(0)
    print(total_mean, total_var)
    return total_mean, total_var
            


if __name__ == '__main__':
    variables.NUM_CLASS = 2
    variables.PATH_ALL = './originaldata2/'
    variables.ksize = 256
    variables.r = 256
    variables.pad = 128 # val 256 train 200
    variables.map_name = "data.tiff"
    variables.label_name = "label.png"  # 这个标签不只有0-1，还有2，我们需要对其处理

    # variables.NUM_CLASS = 2
    # variables.PATH_ALL = './originaldata/'
    # variables.ksize = 512
    # variables.r = 256
    # variables.pad = 256 # val 256 train 200
    # variables.map_name = "img.tif"
    # variables.label_name = "label.tif"  # 这个标签不只有0-1，还有2，我们需要对其处理
    
    create_train()
    # create_train_add()
    create_val()
    # num_cal()
    # mean()

# 2022 1040