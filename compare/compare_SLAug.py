# coding:utf-8
import argparse
import os
import random
import sys
import time
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
sys.path.append(".")
from util.my_dataset import slaug_dataset
from util.util import calculate_accuracy, calculate_index, intersect_and_union, f_score, prepare_training
from tqdm import tqdm
from tensorboardX import SummaryWriter
import model
from monai.losses import DiceLoss
import torch.nn.functional as F
import numpy as np
import cv2
CUDA_LAUNCH_BLOCKING=1

class SetCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss=torch.nn.CrossEntropyLoss()
        self.dice_loss=DiceLoss(to_onehot_y=True,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)
        self.weight_dict={'ce_loss':1, 'dice_loss':0}

    def get_loss(self, pred, gt):
        if len(gt.size())==4 and gt.size(1)==1:
            gt=gt[:,0]

        if type(pred) is not list:
            _ce=self.ce_loss(pred,gt)
            _dc=self.dice_loss(pred,gt.unsqueeze(1))
            return {'ce_loss': _ce,'dice_loss':_dc}
        else:
            ce=0
            dc=0
            for p in pred:
                ce+=self.ce_loss(p,gt)
                dc+=self.dice_loss(p,gt.unsqueeze(1))
            return {'ce_loss': ce, 'dice_loss':dc}

def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device='cuda'):
    '''
    generate bspline 2D kernel matrix.
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation
    :param asTensor:
    :param dtype: data type
    :param use_gpu: bool
    :return:
    '''
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(i * padding).tolist()) / ((sigma[0] * sigma[1]))

    if asTensor:
        return kernel.to(dtype=dtype, device=device)
    else:
        return kernel.numpy()

def get_bspline_kernel(spacing=[32, 32], order=3):
    '''
    :param order init: bspline order, default to 3
    :param spacing tuple of int: spacing between control points along h and w.
    :return:  kernel matrix
    '''
    _kernel = bspline_kernel_2d(spacing, order=order, asTensor=True)
    _padding = (np.array(_kernel.size()[2:]) - 1) / 2
    _padding = _padding.astype(dtype=int).tolist()
    return _kernel, _padding

def rescale_intensity(data, new_min=0, new_max=1, group=4, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    new_data = (data - old_min + eps) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data

def get_SBF_map(gradient, grid_size):
    b, c, h, w = gradient.size()
    bs_kernel, bs_pad = get_bspline_kernel(spacing=[h // grid_size, h // grid_size], order=2)
    bs_kernel = bs_kernel.to(gradient.device)

    #Smooth the saliency map
    saliency = F.adaptive_avg_pool2d(gradient, grid_size)
    saliency = F.conv_transpose2d(saliency, bs_kernel, padding=bs_pad, stride=h // grid_size)
    saliency = F.interpolate(saliency, size=(h, w), mode='bilinear', align_corners=True)

    #Normalize the saliency map
    saliency = rescale_intensity(saliency)
    return saliency

def get_args_parser():
    project_name='compare/add/slaug/'
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch_from', default=1, type=int)
    parser.add_argument('--epoch_max', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_class', default=6, type=int)
    parser.add_argument('--n_channels', default=4, type=int)

    # * Optimizer parameters
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_start', default=3e-4, type=int)
    parser.add_argument('--lr_decay', default=0.97, type=float)
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--step_size', default=80, type=float)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--lr_min', default=1.0e-7, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    # * Dataset parameters
    parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experimentadd/train/img/', type=str)
    parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experimentadd/train/label/', type=str)
    parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experimentadd/val_e_backup/img/', type=str)
    parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experimentadd/val_e_backup/label/', type=str)

    # parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experiment2021/train/img/', type=str)
    # parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experiment2021/train/label/', type=str)
    # parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experiment2021/val/img/', type=str)
    # parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experiment2021/val/label/', type=str)

    parser.add_argument('--map_seffix', default='.npy', type=str)
    parser.add_argument('--label_seffix', default='.npy', type=str)
    parser.add_argument('--augmentation_methods', default=[], type=list)

   # * Project parameters
    parser.add_argument('--project_name', default=project_name, type=str)
    parser.add_argument('--model_name', default='SLAug', type=str)

    # * SAM parameters
    parser.add_argument('--model_type', default='efficientunetb2', type=str)

    # * Path
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--model_dir', default='weights/', type=str)
    parser.add_argument('--tensorboard_log_dir', default=f'weights/{project_name}/tensorboard/', type=str)
    parser.add_argument('--threshold', default=0.8, type=float)

    # * if mutistep, parameters such as lr inherit from the optimizer above 
    parser.add_argument('--mutioptim', default='torch.optim.Adam', type=str)
    parser.add_argument('--rho', default=0.05, type=str)
    parser.add_argument('--adaptive', default=False, type=bool)

    parser.add_argument('--concat_input', default=True, type=bool)
    parser.add_argument('--grid_size', default=3, type=int)

    parser.add_argument('--gla_weight', type=float, default=0.1,
                    help='weight for consistency loss, e.g. js loss')
    parser.add_argument('--lla_weight', type=float, default=0.1,
                    help='weight for consistency loss, e.g. js loss')

    
    return parser

def train(epo, model, train_loader, optimizer, args):
    for param_group in optimizer.param_groups:
        lr_this_epo = param_group['lr']
    
    loss_avg = 0.
    acc_avg = 0.
    start_t = t = time.time()
    model = model.cuda(args.gpu)
    model.train()  
    criterion = SetCriterion()
    for it, (images_org, GLA_images, LLA_images, labels, num) in enumerate(train_loader):
        if args.gpu >= 0:
            images_org = images_org.cuda(args.gpu)
            images_org = images_org.float()
            GLA_images = GLA_images.cuda(args.gpu)
            GLA_images = GLA_images.float()
            LLA_images = LLA_images.cuda(args.gpu)
            LLA_images = LLA_images.float()
            labels = labels.cuda(args.gpu)
            labels = labels.long()

        B, C, H, W = images_org.shape
        images = torch.autograd.Variable(images_org, requires_grad=True)
        GLA = torch.autograd.Variable(GLA_images, requires_grad=True)
        optimizer.zero_grad()    

        logits = model(images);logits_GLA = model(GLA)
        loss_dict = criterion.get_loss(logits, labels)
        losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys() if k in criterion.weight_dict)
        losses_GLA = criterion.get_loss(logits_GLA, labels)
        losses_GLA = sum(losses_GLA[k] * criterion.weight_dict[k] for k in losses_GLA.keys() if k in criterion.weight_dict)
        losses = losses + args.gla_weight*losses_GLA
        loss_all = losses.item() + args.gla_weight*losses_GLA.item()
        losses.backward()
        
        gradient = torch.sqrt(torch.mean(images.grad ** 2, dim=1, keepdim=True)).detach()
        gradient_GLA = torch.sqrt(torch.mean(GLA.grad ** 2, dim=1, keepdim=True)).detach()
        saliency=get_SBF_map(gradient, args.grid_size) + 0.1*get_SBF_map(gradient_GLA, args.grid_size)

        mixed_img = (images.detach() + args.gla_weight*GLA_images.detach()) * saliency + LLA_images * (1 - saliency)
        aug_var = torch.autograd.Variable(mixed_img, requires_grad=True)
        aug_logits = model(aug_var)
        aug_loss_dict = criterion.get_loss(aug_logits, labels)
        aug_losses = args.lla_weight*sum(aug_loss_dict[k] * criterion.weight_dict[k] for k in aug_loss_dict.keys() if k in criterion.weight_dict)
        loss_all = loss_all + 0.1*aug_losses.item()
        aug_losses.backward()

        optimizer.step()
        acc,_ = calculate_accuracy(logits, labels, args.n_class)
        loss_avg += float(loss_all)
        acc_avg += float(acc)

        cur_t = time.time()
        if cur_t - t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                  % (
                  epo, args.epoch_max, it + 1, train_loader.n_iter, (it + 1) * args.batch_size / (cur_t - start_t), float(loss_all),
                  float(acc)))
            t += 5

    content = '| epo:%s/%s \nlr:%.4f train_loss_avg:%.4f train_acc_avg:%.4f ' \
              % (epo, args.epoch_max, lr_this_epo, loss_avg / train_loader.n_iter, acc_avg / train_loader.n_iter)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    return format(acc_avg / train_loader.n_iter, '.4f'), format(loss_avg / train_loader.n_iter, '.4f')


def validation(epo, model, val_loader, args):
    loss_avg = 0.
    acc_avg = 0.
    start_t = time.time()
    model.eval()
    ce_criterion = torch.nn.CrossEntropyLoss()

    total_area_intersect = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_union = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_label = torch.zeros((args.n_class,), dtype=torch.float64)

    total_area_intersect = total_area_intersect.cuda(args.gpu)
    total_area_union = total_area_union.cuda(args.gpu)
    total_area_pred_label = total_area_pred_label.cuda(args.gpu)
    total_area_label = total_area_label.cuda(args.gpu)

    with torch.no_grad():
        confusionmat = torch.zeros([args.n_class, args.n_class])
        confusionmat = confusionmat.to(args.gpu)
        for it, (images, labels, num) in enumerate(val_loader):
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                images = images.float()
                labels = labels.cuda(args.gpu)
                labels = labels.long()
            logits1 = model(images)
            
            loss1 = ce_criterion(logits1, labels)
            loss = loss1
            acc, confusionmat_tmp = calculate_accuracy(logits1, labels, args.n_class)
            confusionmat = confusionmat + confusionmat_tmp
            for i in range(logits1.shape[0]):
                it_logit = logits1[i]
                it_label = labels[i]
                area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
                    args.n_class, it_logit, it_label)
                total_area_intersect += area_intersect
                total_area_union += area_union
                total_area_pred_label += area_pred_label
                total_area_label += area_label

            loss_avg += float(loss)
            acc_avg += float(acc)

            cur_t = time.time()
            print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                  % (epo, args.epoch_max, it + 1, val_loader.n_iter, (it + 1) * args.batch_size / (cur_t - start_t), float(loss),
                     float(acc)))
    _, _, _, _, OA, _, _, mIoU = calculate_index(confusionmat)
    iou = total_area_intersect / total_area_union
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    beta = 1
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    dice = 2 * total_area_intersect / (
            total_area_pred_label + total_area_label)
    acc = total_area_intersect / total_area_label
    mtx0 = '|************************validation****************************\n'
    mtx1 = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' \
           % (loss_avg / val_loader.n_iter, acc_avg / val_loader.n_iter)
    mtx2 = '|OA:' + str((OA*100).round(2)) + '\n'
    mtx22 = '|mIoU:' + str((mIoU*100).round(2)) + '\n'
    mtx3 = '|IoU' + str(iou.cpu().numpy()) + '\n'
    # mtx4 = '|Acc' + str(acc.cpu().numpy()) + '\n'
    mtx5 = '|Fscore' + str(f_value.cpu().numpy()) + '\n'
    mtx6 = '|Precision' + str(precision.cpu().numpy()) + '\n'
    mtx7 = '|Recall' + str(recall.cpu().numpy()) + '\n'
    mtx8 = '|Dice' + str(dice.cpu().numpy()) + '\n'

    print(mtx0, mtx1, mtx2, mtx3, mtx22, mtx5, mtx6, mtx7, mtx8)

    with open(log_file, 'a') as appender:
        appender.write(mtx0)
        appender.write(mtx1)
        appender.write(mtx2)
        appender.write(mtx22)
        appender.write(mtx3)
        appender.write(mtx5)
        appender.write(mtx6)
        appender.write(mtx7)
        appender.write(mtx8)
        appender.write('\n')
    return format(acc_avg / val_loader.n_iter, '.4f'), format(loss_avg / val_loader.n_iter, '.4f'), format(mIoU, '.4f')


def main(args):
    model_name_all = 'model.' + args.model_type
    Model = eval(model_name_all)(args=args)
    optimizer, lr_scheduler = prepare_training(args, Model)

    if args.gpu >= 0: Model.cuda(args.gpu)

    # Model.load_state_dict(torch.load('./compare/SLAug_epo23_tacc0.9300_vacc0.9070_vmiou0.6793.pth', map_location='cuda:'+str(args.gpu)))

    train_dataset = slaug_dataset(
        map_dir=args.train_map, 
        map_seffix=args.map_seffix, 
        label_dir=args.train_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=True
        )

    val_dataset = slaug_dataset(
        map_dir=args.val_map, 
        map_seffix=args.label_seffix, 
        label_dir=args.val_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=False
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter = len(val_loader)
    # 制作伪标签
    # pseudo_generate(val_loader, args)
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)
    stop_flag = False
    current_vmiou = 0
    for epo in tqdm(range(args.epoch_from, args.epoch_max + 1)):
        lr_scheduler.step()
        print('\n| epo #%s begin...' % epo)

        t_acc, t_loss = train(epo, Model, train_loader, optimizer, args)
        v_acc, v_loss, v_miou = validation(epo, Model, val_loader, args)

        # record the score to tensorboard
        writer.add_scalars('train_acc', {args.project_name: float(t_acc)}, epo)
        writer.add_scalars('train_loss', {args.project_name: float(t_loss)}, epo)
        writer.add_scalars('val_acc', {args.project_name: float(v_acc)}, epo)
        writer.add_scalars('val_loss', {args.project_name: float(v_loss)}, epo)

        torch.save(Model.state_dict(), checkpoint_model_file)

        if float(v_miou) <= current_vmiou:
            continue
        current_vmiou = max(current_vmiou, float(v_miou))

        print('| saving check point model file... ', end='')

        checkpoint_epoch_name = model_dir_path + args.model_name + '_epo' + str(epo) + '_tacc' + str(t_acc) + '_vacc' + str(
            v_acc) + '_vmiou' + str(v_miou) +'.pth'
        torch.save(Model.state_dict(), checkpoint_epoch_name)

        print('done!')
        if stop_flag == True:
            break
    writer.close()
    os.rename(checkpoint_model_file, final_model_file)

def pred_pic(args):
    model_name_all = 'model.' + args.model_type
    Model = eval(model_name_all)(args=args)
    model_dir = '/zbssd/yuyu/code/Domain_torch/weights/compare/add/slaug/SLAug_epo34_tacc0.9846_vacc0.8415_vmiou0.6119.pth'
    Model.load_state_dict(torch.load(model_dir, map_location='cuda:0'))

    if args.gpu >= 0: Model.cuda(args.gpu)

    val_dataset = slaug_dataset(
        map_dir=args.val_map, 
        map_seffix=args.label_seffix, 
        label_dir=args.val_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=False
        )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader.n_iter = len(val_loader)
    loss_avg = 0.
    acc_avg = 0.
    start_t = time.time()
    Model.eval()
    ce_criterion = torch.nn.CrossEntropyLoss()

    total_area_intersect = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_union = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((args.n_class,), dtype=torch.float64)
    total_area_label = torch.zeros((args.n_class,), dtype=torch.float64)

    total_area_intersect = total_area_intersect.cuda(args.gpu)
    total_area_union = total_area_union.cuda(args.gpu)
    total_area_pred_label = total_area_pred_label.cuda(args.gpu)
    total_area_label = total_area_label.cuda(args.gpu)

    # map_H, map_W = 22183, 18838
    # map_H, map_W = 22698, 18928
    map_H, map_W = 7312, 7712
    picture = np.zeros((map_H, map_W), dtype=int)
    # color_map = {
    #         1: (255, 255, 255),  # 类别1的颜色，白色
    #         2: (0, 0, 255),       # 类别2的颜色，红色
    #         3: (0, 255, 0),       # 类别3的颜色，绿色
    #         4: (255, 197, 0),     # 类别4的颜色 淡蓝色
    #         5: (0, 255, 255),     # 类别5的颜色，黄色
    #         6: (255, 0, 255),     # 类别6的颜色，紫色
    #         7: (255, 255, 0),     # 类别7的颜色，青色
    #         8: (255, 0, 0)        # 类别8的颜色，蓝色
    #     }
    color_map = {
            1: (255, 255, 255),  # 类别1的颜色，白色
            2: (0, 255, 255),     # 类别5的颜色，黄色
            3: (0, 255, 0),       # 类别3的颜色，绿色
            4: (255, 255, 0),     # 类别7的颜色，青色
            5: (0, 0, 255),       # 类别2的颜色，红色
            6: (255, 0, 0)        # 类别8的颜色，蓝色
        }
    X = []
    index_all = []
    segmented_image = np.zeros((map_H, map_W, 3), dtype=np.uint8)
    start = time.time()
    with torch.no_grad():
        labels_array = np.arange(args.n_class)
        confusionmat = np.zeros([args.n_class, args.n_class])
        for it, (images, labels, num) in enumerate(val_loader):
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                images = images.float()
                labels = labels.cuda(args.gpu)
                labels = labels.long()
            logits = Model(images)
            
            # loss = ce_criterion(logits, labels)
            # zero_mask = (images == 0).all(dim=1)
            # non_zero_mask = ~zero_mask 

            # confusionmat_tmp = confusion_matrix(
            #     labels[non_zero_mask].cpu().numpy().reshape(-1), 
            #     logits.argmax(1)[non_zero_mask].cpu().numpy().reshape(-1), 
            #     labels=labels_array
            # )
            # acc, _ = calculate_accuracy(logits, labels, args.n_class)
            # confusionmat = confusionmat + confusionmat_tmp
            # for i in range(logits.shape[0]):
            #     it_logit = logits[i]
            #     it_label = labels[i]
            #     area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            #         args.n_class, it_logit, it_label)
            #     total_area_intersect += area_intersect
            #     total_area_union += area_union
            #     total_area_pred_label += area_pred_label
            #     total_area_label += area_label

            # loss_avg += float(loss)
            # acc_avg += float(acc)

            # cur_t = time.time()
            # pred_result = logits.argmax(1)
            # pred_result = pred_result.cpu().numpy()

            # for i in range(pred_result.shape[0]):
            #     pred_result_batch = pred_result[i]
            #     X.append(pred_result_batch)
            #     index_all.append(index[i].cpu().numpy())
    end = time.time()
    running_time = end-start
    PA, UA, F1, mean_F1, OA, Kappa, IoU, mIoU = calculate_index(confusionmat)
    num_X = 0
    for x, y in index_all:
        picture[x:x + 512, y:y + 512] = X[num_X]
        num_X += 1
    picture = picture + 1
    for label, color in color_map.items():
        segmented_image[picture == label] = color
    segmented_image = segmented_image[256:map_H-256,256:map_W-256,:]
    cv2.imwrite("/zbssd/yuyu/code/Domain_torch/picture_4/add/e/slauge.png", segmented_image)

    iou = total_area_intersect / total_area_union
    precision = total_area_intersect / total_area_pred_label
    recall = total_area_intersect / total_area_label
    beta = 1
    f_value = torch.tensor(
        [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
    dice = 2 * total_area_intersect / (
            total_area_pred_label + total_area_label)
    acc = total_area_intersect / total_area_label
    mtx0 = '|************************validation****************************\n'
    mtx1 = '| val_loss_avg:%.4f val_acc_avg:%.4f\n' \
           % (loss_avg / val_loader.n_iter, acc_avg / val_loader.n_iter)
    mtx2 = '|OA:' + str((OA*100).round(2)) + '\n'
    mtx22 = '|mIoU:' + str((mIoU*100).round(2)) + '\n'
    mtx3 = '|IoU' + str(iou.cpu().numpy()) + '\n'
    # mtx4 = '|Acc' + str(acc.cpu().numpy()) + '\n'
    mtx5 = '|Fscore' + str(f_value.cpu().numpy()) + '\n'
    mtx6 = '|Precision' + str(precision.cpu().numpy()) + '\n'
    mtx7 = '|Recall' + str(recall.cpu().numpy()) + '\n'
    mtx8 = '|Dice' + str(dice.cpu().numpy()) + '\n'

    print(mtx0, mtx1, mtx2, mtx3, mtx22, mtx5, mtx6, mtx7, mtx8)

    with open(log_file, 'a') as appender:
        appender.write(mtx0)
        appender.write(mtx1)
        appender.write(mtx2)
        appender.write(mtx22)
        appender.write(mtx3)
        appender.write(mtx5)
        appender.write(mtx6)
        appender.write(mtx7)
        appender.write(mtx8)
        appender.write('\n')
    return format(acc_avg / val_loader.n_iter, '.4f'), format(loss_avg / val_loader.n_iter, '.4f'), format(mIoU, '.4f')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    model_dir_path = os.path.join(args.model_dir, args.project_name + '/')
    os.makedirs(model_dir_path, exist_ok=True)
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)

    checkpoint_model_file = os.path.join(model_dir_path, 'tmp.pth')

    final_model_file = os.path.join(model_dir_path, 'final.pth')
    log_file = os.path.join(model_dir_path, 'log.txt')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % model_dir_path)

    # main(args)
    pred_pic(args)
