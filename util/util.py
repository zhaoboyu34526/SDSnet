# coding:utf-8
import numpy as np
import chainer
from PIL import Image
import torch
from osgeo import gdal
import cv2
from torch.optim import SGD, Adam, AdamW
import cuml
import cudf
import random
# from ipdb import set_trace as st


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
 
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

# def resize_img(img_data, resize_img_hei, resize_img_wid):  # 数据resize，label对应什么处理
#     if len(img_data.shape) > 2:
#         if img_data.shape[2] > img_data.shape[0]:
#             band_num = img_data.shape[0]
#         else:
#             band_num = img_data.shape[2]
#         if img_data.shape[2] > img_data.shape[0]:
#             import cv2
#             import numpy as np
#             new_img_data = np.zeros([band_num, resize_img_hei, resize_img_wid])
#             for b in range(band_num):
#                 new_band = img_data[b]
#                 new_band = cv2.resize(
#                     new_band, (resize_img_hei, resize_img_wid))
#                 new_img_data[b] = new_band
#         else:
#             import cv2
#             import numpy as np
#             new_img_data = np.zeros([resize_img_hei, resize_img_wid, band_num])
#             for b in range(band_num):
#                 new_band = img_data[:, :, b]
#                 new_band = cv2.resize(
#                     new_band, (resize_img_hei, resize_img_wid))
#                 new_img_data[:, :, b] = new_band
#     else:
#         band_num = 1
#         import cv2
#         new_img_data = cv2.resize(img_data, (resize_img_hei, resize_img_wid))
#     return new_img_data

def resize_img(img_data, resize_img_hei, resize_img_wid):
    if len(img_data.shape) > 2:
        if img_data.shape[2] > img_data.shape[0]:
            band_num = img_data.shape[0]
        else:
            band_num = img_data.shape[2]
        if img_data.shape[2] > img_data.shape[0]:
            import cv2
            import numpy as np
            new_img_data = np.zeros([band_num, resize_img_hei, resize_img_wid])
            for b in range(band_num):
                new_band = img_data[b]
                # print(new_band.shape)
                # print(new_band.max())
                # print(new_band.min())
                # print(b)
                # print(img_data.shape)
                # print(new_band.shape)
                # print(resize_img_hei,resize_img_wid)
                new_band = cv2.resize(new_band, (resize_img_hei, resize_img_wid))
                new_img_data[b] = new_band
        else:
            import cv2
            import numpy as np
            new_img_data = np.zeros([resize_img_hei, resize_img_wid, band_num])
            for b in range(band_num):
                new_band = img_data[:, :, b]
                new_band = cv2.resize(
                    new_band, (resize_img_hei, resize_img_wid))
                new_img_data[:, :, b] = new_band
    else:
        band_num = 1
        import cv2
        new_img_data = cv2.resize(img_data, (resize_img_hei, resize_img_wid))
    return new_img_data

def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(num_classes, logits, labels):  # 各个类是混在一起的嘛
    logits = logits.argmax(0)
    intersect = logits[logits == labels]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes)
        # intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        logits.float(), bins=(num_classes), min=0, max=num_classes)
        # logits.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        labels.float(), bins=(num_classes), min=0, max=num_classes)
        # labels.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

def total_intersect_and_union(num_classes,logits,labels):  # ？？？
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)

def calculate_accuracy(logits, labels, numClass):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels == -1).sum()  # 没看到 置为 -1的操作
    count = ((predictions == labels)*(labels != -1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    
    mask = (labels != -1)
    label = numClass * labels[mask] + predictions[mask]
    count_conf = torch.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count_conf.reshape(numClass, numClass)
    return acc, confusionMatrix

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class, n_class))
    IoU = np.zeros(n_class)
    conf[:, 0] = cf[:, 0]/cf[:, 0].sum()  # 为啥第 0 列 要单独算
    for cid in range(0, n_class):
        if cf[:, cid].sum() > 0:
            conf[:, cid] = cf[:, cid]/cf[:, cid].sum()
            IoU[cid] = cf[cid, cid]/(cf[cid, 0:].sum()+cf[0:, cid].sum()-cf[cid, cid])
    overall_acc = np.diag(cf[0:, 0:]).sum()/cf[0:, :].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU

def calculate_index(confusionmat):
    if isinstance(confusionmat, torch.Tensor):
        confusionmat = confusionmat.cpu().detach().numpy()

    unique_index = np.where(np.sum(confusionmat, axis=1) != 0)[0]
    confusionmat = confusionmat[unique_index, :]
    confusionmat = confusionmat[:, unique_index]

    a = np.diag(confusionmat)
    b = np.sum(confusionmat, axis=0)
    c = np.sum(confusionmat, axis=1)

    eps = 0.0000001

    PA = a / (c + eps)
    UA = a / (b + eps)

    F1 = 2 * PA * UA / (PA + UA + eps)

    mean_F1 = np.nanmean(F1)

    OA = np.sum(a) / np.sum(confusionmat)

    PE = np.sum(b * c) / (np.sum(c) * np.sum(c))
    Kappa = (OA - PE) / (1 - PE)

    intersection = np.diag(confusionmat)
    union = np.sum(confusionmat, axis=1) + np.sum(confusionmat, axis=0) - np.diag(confusionmat)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)
    return PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU


# for visualization
def get_palette():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(names, predictions):
    palette = get_palette()

    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy()
        img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for cid in range(1, int(predictions.max())):  # ???
            img[pred == cid] = palette[cid]

        img = Image.fromarray(np.uint8(img))
        img.save(names[i].replace('.png', '_pred.png'))

def WriteImage(drc_path, image, datatype=gdal.GDT_Byte, bands=1, proj='', gt='', compress=False):
    width = image.shape[1]  # 设置输出影像的参数-图片长
    height = image.shape[0]
    driver = gdal.GetDriverByName("GTiff")  # 开辟输出的tiff的空间
    if compress:
        dataset = driver.Create(drc_path, width, height, bands, datatype, options=[
            "TILED=YES", "COMPRESS=LZW"])
    else:
        dataset = driver.Create(drc_path, width, height, bands, datatype)
    if bands == 1 and len(image.shape) == 2:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for b in range(bands):
            new_band = image[:, :, b]
            dataset.GetRasterBand(b+1).WriteArray(new_band)
    if gt != '':
        dataset.SetGeoTransform(gt)
    if proj != '':
        dataset.SetProjection(proj)
    del dataset
    # return image_new


class ReflectImage:
    def __init__(self, image, standard_width=1000, standard_height=1000):
        self.image = image
        self.standard_width = standard_width
        self.standard_height = standard_height

    def operation(self):
        if len(self.image.shape) == 3:
            image_new = np.pad(self.image, ((
                0, self.standard_width-self.image.shape[0]), (0, self.standard_height-self.image.shape[1]), (0, 0)), 'reflect')
        if len(self.image.shape) == 2:
            image_new = np.pad(self.image, ((
                0, self.standard_width-self.image.shape[0]), (0, self.standard_height-self.image.shape[1])), 'reflect')
        return image_new


def ResizeImage(image, shape_0, shape_1):
    print(image.shape)
    if len(image.shape)==2:
        new_img_data = cv2.resize(
            image, (shape_0, shape_1), interpolation=cv2.INTER_NEAREST)
    else:
        if 'int8' in image.dtype.name:
            result_datatype = 'uint8'
        elif 'int16' in image.dtype.name:
            result_datatype = 'uint16'
        else:
            result_datatype = 'uint32'
        new_img_data = np.zeros((shape_0,shape_1,image.shape[2]), dtype=result_datatype)
        for i in range(image.shape[2]):
            image = image[:,:,i]
            image = cv2.resize(image, (shape_0, shape_1),
                               interpolation=cv2.INTER_NEAREST)
            new_img_data[i] = image
    return new_img_data

#直方图均衡化
class EqualizeHist:
    def __init__(self, image, bins=65536, normalize_max=255, normalize_type='uint8'):
        self.image = image
        self.bins = image.max()+1
        self.normalize_max = normalize_max
        self.normalize_type = normalize_type

    def get_histogram(self, image):
        # array with size of bins, set to zeros
        histogram = np.zeros(self.bins)
        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1
        # return our final result
        return histogram

    def cumsum(self, a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def operation(self):
        flat = self.image.flatten()
        hist = self.get_histogram(flat)
        # execute the fn
        cs = self.cumsum(hist)
        # numerator & denomenator
        nj = (cs - cs.min()) * self.normalize_max
        N = cs.max() - cs.min()
        # re-normalize the cdf
        cs = nj / N
        cs = cs.astype(self.normalize_type)
        image_new = cs[flat]
        image_new = np.reshape(image_new, self.image.shape)
        return image_new

#百分比截断


class TruncatedLinearStretch:
    def __init__(self, image, truncated_value=2, max_out=255, min_out=0, normalize_type='uint8'):
        self.image = image
        self.truncated_value = truncated_value
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        truncated_down = np.percentile(self.image, self.truncated_value)
        truncated_up = np.percentile(self.image, 100 - self.truncated_value)
        image_new = (self.image - truncated_down) / (truncated_up -
                                                     truncated_down) * (self.max_out - self.min_out) + self.min_out
        image_new[image_new < self.min_out] = self.min_out
        image_new[image_new > self.max_out] = self.max_out
        image_new = image_new.astype(self.normalize_type)
        return image_new

##标准差拉伸


class StandardDeviation:
    def __init__(self, image, parameter=2, max_out=255, min_out=0, normalize_type='uint8'):
        self.image = image
        self.parameter = parameter
        self.max_out = max_out
        self.min_out = min_out
        self.normalize_type = normalize_type

    def operation(self):
        Mean = np.mean(self.image)
        StdDev = np.std(self.image, ddof=1)
        ucMax = Mean + self.parameter * StdDev
        ucMin = Mean - self.parameter * StdDev
        k = (self.max_out - self.min_out) / (ucMax - ucMin)
        b = (ucMax * self.min_out - ucMin * self.max_out) / (ucMax - ucMin)
        if (ucMin <= 0):
            ucMin = 0

        image_new = np.select([self.image == self.min_out, self.image <= ucMin, self.image >= ucMax,  k*self.image+b < self.min_out, k*self.image+b > self.max_out,
                               (k*self.image+b > self.min_out) & (k*self.image+b < self.max_out)],
                              [self.min_out, self.min_out, self.max_out, self.min_out, self.max_out, k * self.image + b], self.image)
        image_new = image_new.astype(self.normalize_type)
        return image_new


def make_optimizer(params, args, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'muti': muti_step
    }[args.optim]
    if args.optim == 'sgd':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adam':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adamw':
        optimizer_spec = {
            'args': dict(lr=args.lr_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'muti':
        optimizer_spec = {
            'args': dict(base_optimizer=eval(args.mutioptim), rho=args.rho, adaptive=args.adaptive, lr=args.lr_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    optimizer = Optimizer(params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def make_g_optimizer(params, args, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'muti': muti_step
    }[args.optim]
    if args.optim == 'sgd':
        optimizer_spec = {
            'args': dict(lr=args.lr_vae_start, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adam':
        optimizer_spec = {
            'args': dict(lr=args.lr_vae_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'adamw':
        optimizer_spec = {
            'args': dict(lr=args.lr_vae_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    elif args.optim == 'muti':
        optimizer_spec = {
            'args': dict(base_optimizer=eval(args.mutioptim), rho=args.rho, adaptive=args.adaptive, lr=args.lr_vae_start, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay),
            'sd': None
        }
    optimizer = Optimizer(params, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def prepare_training(args, model):
    optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_cycle, eta_min=args.lr_min)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
    elif args.lr_scheduler == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epo: args.lr_decay ** ((epo - 1) / 10 + 1))
    elif args.lr_scheduler == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay) 
    elif args.lr_scheduler == 'reduce': 
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay) 
    elif args.lr_scheduler == 'cosinewarm':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.lr_min)
    return optimizer, lr_scheduler

def prepare_training_g(args, model, generator):
    optimizer = make_optimizer(filter(lambda p: p.requires_grad, model.parameters()), args)
    optimizer_g = make_g_optimizer(filter(lambda p: p.requires_grad, generator.parameters()), args)

    if args.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_max, eta_min=args.lr_min)
        lr_scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, args.epoch_max, eta_min=args.lr_min)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=args.step_size, gamma=args.lr_vae_decay)
    elif args.lr_scheduler == 'poly':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epo: args.lr_decay ** ((epo - 1) / 10 + 1))
        lr_scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lambda epo: args.lr_vae_decay ** ((epo - 1) / 10 + 1))
    elif args.lr_scheduler == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay) 
        lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=args.lr_vae_decay)
    elif args.lr_scheduler == 'reduce': 
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay) 
        lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=args.lr_vae_decay)
    return optimizer, optimizer_g, lr_scheduler, lr_scheduler_g

class muti_step(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(muti_step, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # torch.no_grad()只是关闭了自动求导，但是不会影响模型的权重更新
        # 第一步实际上是收集信息的过程
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                self.state[p]["used_in_step"] = False
                if p.grad is None: continue
                # 保留原始的参数值
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["used_in_step"] = True
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        i = 0
        for group in self.param_groups:
            i+=1
            for p in group["params"]:
                if p.grad is None: continue
                if not self.state[p]["used_in_step"]: continue
                # 恢复成原始的参数值
                p.data = self.state[p]["old_p"] # get back to "w" from "w + e(w)"
                self.state[p]['used_in_step'] = False # clean the state of the parameter

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

def sample_random_points(coords, total_num, num_divisions):
    # 生成每个区间的起始和结束索引
    intervals = np.linspace(0, total_num, num_divisions + 1, dtype=int)
    start_indices = intervals[:-1]  # 所有区间的起始索引
    end_indices = intervals[1:]  # 所有区间的结束索引

    # 为每个区间生成一个随机偏移量
    # random_offsets = np.random.randint(0, (end_indices - start_indices), size=num_divisions)
    random_offsets = np.random.randint(0, (end_indices - start_indices), size=num_divisions)
    random_indices = start_indices + random_offsets

    # 选取随机索引对应的坐标
    selected_coords = coords[random_indices]

    return selected_coords

def sample_midpoint_points(coords, total_num, num_divisions):
    # 生成每个区间的起始和结束索引
    intervals = np.linspace(0, total_num, num_divisions + 1, dtype=int)
    start_indices = intervals[:-1]  # 所有区间的起始索引
    end_indices = intervals[1:]  # 所有区间的结束索引

    # 为每个区间计算中点索引
    mid_indices = (start_indices + end_indices) // 2

    # 选取中点索引对应的坐标
    selected_coords = coords[mid_indices]

    return selected_coords

def promt_generate(labels, is_clu):
    prompt = []
    num = 200
    if is_clu == True:
        for i in range(labels.size(0)):
            if isinstance(labels, torch.Tensor):
                labels_numpy = labels[i].detach().cpu().numpy()
            else:
                labels_numpy = labels[i]
            record = {}  # 单个图像的记录
            # 分别查找标记为255和非255的点的坐标
            coords_255 = np.transpose(np.array(np.where(labels_numpy == 255)), (1, 0))
            coords_n255 = np.transpose(np.array(np.where(labels_numpy != 255)), (1, 0))
            if coords_255.shape[0] > 0:
                if coords_255.shape[0] < int(num/2):
                    record["point_coords"] = coords_255
                    record["point_labels"] = np.full(coords_255.shape[0], 1)
                    num_n255 = num - coords_255.shape[0]
                    nums_all = coords_n255.shape[0]
                    # indices = np.linspace(0, nums_all-1, num_n255, dtype=int)
                    # coords_n255 = coords_n255[indices, :]、
                    coords_n255 = sample_midpoint_points(coords_n255, nums_all, num_n255)

                    record["point_coords"] = np.concatenate((record["point_coords"], coords_n255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.zeros(coords_n255.shape[0])), axis=0)
                
                elif coords_n255.shape[0] < int(num/2):
                    record["point_coords"] = coords_n255
                    record["point_labels"] = np.full(coords_n255.shape[0], 0)
                    num_255 = num - coords_n255.shape[0]

                    nums_all = coords_255.shape[0]
                    # indices = np.linspace(0, nums_all-1, num_255, dtype=int)
                    # coords_255 = coords_255[indices, :]
                    coords_255 = sample_midpoint_points(coords_255, nums_all, num_255)

                    record["point_coords"] = np.concatenate((record["point_coords"], coords_255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.ones(coords_255.shape[0])), axis=0)

                else:
                    # 将数据转换为cuDF DataFrame，cuML通常与cuDF一起使用以实现GPU加速
                    nums_all_n255 = coords_n255.shape[0]
                    nums_all_255 = coords_255.shape[0]
                    # indices_n255 = np.linspace(0, nums_all_n255-1, num//2, dtype=int)
                    # indices_255 = np.linspace(0, nums_all_255-1, num//2, dtype=int)
                    # coords_n255 = coords_n255[indices_n255, :]
                    # coords_255 = coords_255[indices_255, :]
                    coords_n255 = sample_midpoint_points(coords_n255, nums_all_n255, num//2)
                    coords_255 = sample_midpoint_points(coords_255, nums_all_255, num//2)
                    record["point_coords"] = coords_255
                    record["point_labels"] = np.full(coords_255.shape[0], 1)
                    record["point_coords"] = np.concatenate((record["point_coords"], coords_n255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.zeros(coords_n255.shape[0])), axis=0)
            else:   
                record["point_coords"] = np.array([labels.size(1)//2, labels.size(2)//2])[np.newaxis, :]
                record["point_labels"] = np.array([1])

            prompt.append(record)
    else:
        for i in range(labels.size(0)):
            if isinstance(labels, torch.Tensor):
                labels_numpy = labels[i].detach().cpu().numpy()
            else:
                labels_numpy = labels[i]
            record = {}  # 单个图像的记录
            # 分别查找标记为255和非255的点的坐标
            coords_255 = np.transpose(np.array(np.where(labels_numpy == 255)), (1, 0))
            coords_n255 = np.transpose(np.array(np.where(labels_numpy != 255)), (1, 0))
            if coords_255.shape[0] > 0:
                if coords_255.shape[0] < int(num/2):
                    record["point_coords"] = coords_255
                    record["point_labels"] = np.full(coords_255.shape[0], 1)
                    num_n255 = num - coords_255.shape[0]
                    coords_n255 = cudf.DataFrame(coords_n255.astype('float32'))
                    kmeans = cuml.KMeans(n_clusters=num_n255)
                    kmeans.fit(coords_n255)
                    coords_n255 = kmeans.cluster_centers_
                    coords_n255 = coords_n255.to_pandas().values
                    coords_n255 = np.round(coords_n255)
                    record["point_coords"] = np.concatenate((record["point_coords"], coords_n255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.zeros(coords_n255.shape[0])), axis=0)
                
                elif coords_n255.shape[0] < int(num/2):
                    record["point_coords"] = coords_n255
                    record["point_labels"] = np.full(coords_n255.shape[0], 0)
                    num_255 = num - coords_n255.shape[0]
                    coords_255 = cudf.DataFrame(coords_255.astype('float32'))
                    kmeans = cuml.KMeans(n_clusters=num_255)
                    kmeans.fit(coords_255)
                    coords_255 = kmeans.cluster_centers_
                    coords_255 = coords_255.to_pandas().values
                    coords_255 = np.round(coords_255)
                    record["point_coords"] = np.concatenate((record["point_coords"], coords_255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.ones(coords_255.shape[0])), axis=0)

                else:
                    # 将数据转换为cuDF DataFrame，cuML通常与cuDF一起使用以实现GPU加速
                    coords_255 = cudf.DataFrame(coords_255.astype('float32'))
                    kmeans = cuml.KMeans(n_clusters=int(num/2))
                    kmeans.fit(coords_255)
                    coords_255 = kmeans.cluster_centers_
                    coords_255 = coords_255.to_pandas().values
                    coords_255 = np.round(coords_255)

                    record["point_coords"] = coords_255
                    record["point_labels"] = np.full(coords_255.shape[0], 1)
                    coords_n255 = cudf.DataFrame(coords_n255.astype('float32'))
                    kmeans = cuml.KMeans(n_clusters=int(num/2))
                    kmeans.fit(coords_n255)
                    coords_n255 = kmeans.cluster_centers_
                    coords_n255 = coords_n255.to_pandas().values
                    coords_n255 = np.round(coords_n255)
                    record["point_coords"] = np.concatenate((record["point_coords"], coords_n255), axis=0)
                    record["point_labels"] = np.concatenate((record["point_labels"], np.zeros(coords_n255.shape[0])), axis=0)
            else:   
                record["point_coords"] = np.array([labels.size(1)//2, labels.size(2)//2])[np.newaxis, :]
                record["point_labels"] = np.array([1])

            prompt.append(record)
    return prompt

class AutomaticWeightedLoss(torch.nn.Module):
    """Automatically weighted multi-task loss.
    Params:
        num: int, the number of losses.
    Examples:
        loss1 = torch.tensor(1.0)
        loss2 = torch.tensor(2.0)
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num))  # Log variance parameters

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # Precision is the inverse of variance
            weighted_loss = precision * loss + self.log_vars[i]  # Notice the change here
            loss_sum += weighted_loss
        return loss_sum / len(losses)
    
# def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
#     model.train()
#     criterion.train()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

#     print_freq = 10
#     cur_iteration=0
#     while True:
#         for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
#             for k,v in samples.items():
#                 if isinstance(samples[k],torch.Tensor):
#                     samples[k]=v.to(device)
#             cur_iteration+=1
#             for i, param_group in enumerate(optimizer.param_groups):
#                 param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

#             img=samples['images']
#             lbl=samples['labels']
#             pred = model(img)
#             loss_dict = criterion.get_loss(pred,lbl)
#             losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
#             optimizer.zero_grad()
#             losses.backward()
#             optimizer.step()

#             metric_logger.update(**loss_dict)
#             metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#             if cur_iteration>=warmup_iteration:
#                 print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
#                 return cur_iteration
#         metric_logger.synchronize_between_processes()
    
if __name__ == '__main__':
    nump = np.array([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16]])
    nump = nump.reshape(8,2)
    a = sample_random_points(nump, 8, 3)
