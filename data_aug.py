import numpy as np
from PIL import Image
import tifffile as tiff
from osgeo import gdal
import os
from collections import Counter
import random
import math
import matplotlib.pyplot as plt

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def plot_segmentation_labels(segmentation_labels):
    num_classes = np.max(segmentation_labels) + 1

    # Define a color map for each class
    cmap = plt.cm.get_cmap('tab20', num_classes)

    # Plot the segmentation labels
    plt.imshow(segmentation_labels, cmap=cmap, vmin=0, vmax=num_classes - 1)
    plt.colorbar(ticks=np.arange(num_classes) + 0.5, label='Class')

    plt.title('Segmentation Labels')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()

def writeTiff(im_data, im_width, im_height, im_bands, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def extract_class_pixels(label_matrix, image, classes):
    # if label_matrix.shape[:2] != image.shape[:2]:
    #     raise ValueError("Label matrix and image must have the same size.")

    # class_matrices = [np.zeros_like(image) for _ in range(num_classes)]
    class_matrices = [np.empty((0, 32), dtype='int16') for _ in range(len(classes))]
    image = np.transpose(image,(1,2,0))
    for k, class_idx in enumerate(classes):
        class_pixel = image[(label_matrix == class_idx)]
        if class_pixel.size > 0:
            # class_matrices[k] = [np.concatenate((class_matrices[k],class_pixel[i])) for i in range(len(class_pixel))]
            class_matrices[k] = np.concatenate((class_matrices[k],class_pixel))
    return class_matrices

def generate_random_segmentation_labels(all_class_matrices, image_height, image_width, classes, num_fragments):
    if image_height <= 0 or image_width <= 0:
        raise ValueError("Image height and width must be positive integers.")
    # if num_classes <= 0:
    #     raise ValueError("Number of classes must be a positive integer.")

    # Create an empty label matrix
    img_aug = np.zeros((image_hei, image_wid, 32), dtype='int16')
    random_labels = np.zeros((image_height, image_width), dtype=np.int32)

    # Generate random fragments for each class
    for _ in range(num_fragments):
        for i, class_idx in enumerate(classes):
            # Generate random fragment size between 10% to 40% of the image size
            # fragment_height = random.randint(image_height // 10, image_height // 2)
            # fragment_height = random.randint(image_height // 10, image_height // 2)
            fragment_height = random.randint(image_height // 5, image_height//2)
            fragment_width = random.randint(image_height // 5, image_width//2)

            # Generate random position for the fragment
            x_start = random.randint(0, image_width - fragment_width)
            y_start = random.randint(0, image_height - fragment_height)

            # Fill the fragment with the current class index
            random_labels[y_start:y_start + fragment_height, x_start:x_start + fragment_width] = class_idx
            num_pixel = fragment_height*fragment_width
            if all_class_matrices[i].shape[0] >= num_pixel:
                indices = list(range(len(all_class_matrices[i])))
                random.shuffle(indices)
                tmp = all_class_matrices[i][np.array((indices[:num_pixel]),dtype=int)]
                img_aug[y_start:y_start + fragment_height, x_start:x_start + fragment_width, :] = tmp.reshape(fragment_height,fragment_width,32)
            else:
                class_mi = all_class_matrices[i]
                nums_re = math.ceil(num_pixel/len(class_mi))
                class_mi = np.repeat(class_mi,nums_re,axis=0)
                # tmp = np.concatenate((class_mi, class_mi[:num_pixel-len(class_mi)]))
                tmp = class_mi[:num_pixel]
                img_aug[y_start:y_start + fragment_height, x_start:x_start + fragment_width, :] = tmp.reshape(fragment_height,fragment_width,32)

    return random_labels, img_aug

data_root = '/home/zyx/data/WHU-OHS/transfer-S4/trainset/'
save_img_path = '/home/zyx/data/WHU-OHS/transfer-S4_aug/trainset/image/tr/'
save_label_path = '/home/zyx/data/WHU-OHS/transfer-S4_aug/trainset/label/tr/'
data_path_train_image = os.path.join(data_root, 'image', 'tr')
train_image_list = []
train_label_list = []
image_prefix = 'S4'
num_classes_aug = [3, 10, 12, 14]
num_fragments_per_class = 10
all_class_matrices = [np.empty((0, 32), dtype='int16') for _ in range(len(num_classes_aug))]
for root, paths, fnames in sorted(os.walk(data_path_train_image)):
    for fname in fnames:
        if is_image_file(fname):
            if ((image_prefix + '_') in fname):
                image_path = os.path.join(data_path_train_image, fname)
                train_image_list.append(image_path)
                label_path = image_path.replace('image', 'label')
                image_dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
                label_dataset = gdal.Open(label_path, gdal.GA_ReadOnly)
                image_wid = image_dataset.RasterXSize
                image_hei = image_dataset.RasterYSize
                image = image_dataset.ReadAsArray()
                label = label_dataset.ReadAsArray()
                class_matrices = extract_class_pixels(label, image, num_classes_aug)
                all_class_matrices = [np.concatenate((all_class_matrices[i],class_matrices[i])) for i in range(len(num_classes_aug))]

num_augimg = len(train_image_list) // 2
for k in range(num_augimg):
    random_labels, img_aug = generate_random_segmentation_labels(all_class_matrices, image_wid, image_hei, 
                                                        num_classes_aug, num_fragments_per_class)
    name = f'{image_prefix}_aug_{str(k+1).zfill(4)}.tif'
    writeTiff(np.transpose(img_aug,(2,0,1)).astype(np.int16), 32, 512, 512, os.path.join(save_img_path, name))
    writeTiff(np.expand_dims(random_labels, 0).astype(np.int16), 1, 512, 512, os.path.join(save_label_path, name))

