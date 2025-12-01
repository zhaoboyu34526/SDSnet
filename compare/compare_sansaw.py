# coding:utf-8
import argparse
import os
import random
import sys
import time
import cuml
import cudf
from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
sys.path.append(".")
from util.my_dataset import huanghe_dataset
from util.util import calculate_accuracy, calculate_index, intersect_and_union, f_score, prepare_training
from tqdm import tqdm
from tensorboardX import SummaryWriter
import model
from model import sam_seg_model_registry, sam_feat_seg_model_registry
from loss.lossfunction import CrossEntropy, FocalLoss
from loss.lovasz_loss import lovasz_softmax
import torch.nn.functional as F
import numpy as np
import cv2

def get_args_parser():
    project_name='test'
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch_from', default=1, type=int)
    parser.add_argument('--epoch_max', default=40, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_class', default=8, type=int)
    parser.add_argument('--n_channels', default=4, type=int)

    # * Optimizer parameters
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_start', default=1e-4, type=int)
    parser.add_argument('--lr_decay', default=0.97, type=float)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--step_size', default=80, type=float)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr_min', default=1.0e-7, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    # * Dataset parameters
    # parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experiment2021/train/img/', type=str)
    # parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experiment2021/train/label/', type=str)
    # parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experiment2021/val/img/', type=str)
    # parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experiment2021/val/label/', type=str)

    parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experiment2022/train/img/', type=str)
    parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experiment2022/train/label/', type=str)
    parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experiment2022/val/img/', type=str)
    parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experiment2022/val/label/', type=str)
    # parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experimentadd/val_e_backup/img/', type=str)
    # parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experimentadd/val_e_backup/label/', type=str)
    parser.add_argument('--map_seffix', default='.npy', type=str)
    parser.add_argument('--label_seffix', default='.npy', type=str)
    parser.add_argument('--augmentation_methods', default=[], type=list)

   # * Project parameters
    parser.add_argument('--project_name', default=project_name, type=str)
    parser.add_argument('--model_name', default='SANSAW', type=str)

    # * SAM parameters
    parser.add_argument('--model_type', default='SANSAW101', type=str)

    # * Path
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--model_dir', default='weights/', type=str)
    parser.add_argument('--tensorboard_log_dir', default=f'weights/{project_name}/tensorboard/', type=str)
    parser.add_argument('--threshold', default=0.8, type=float)

    # * if mutistep, parameters such as lr inherit from the optimizer above 
    parser.add_argument('--mutioptim', default='torch.optim.Adam', type=str)
    parser.add_argument('--rho', default=0.05, type=str)
    parser.add_argument('--adaptive', default=False, type=bool)

    parser.add_argument('--selected_classes', default=[0,1,2,3,4],
                        help="poly_power")
    
    return parser

def get_one_hot( label, N):
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(N).to(label.device)
    ones = ones.index_select(0, label)
    size.append(N)
    ones = ones.view(*size)
    ones = ones.transpose(2, 3)
    ones = ones.transpose(1, 2)
    return ones

def train(epo, model, train_loader, optimizer, args):
    for param_group in optimizer.param_groups:
        lr_this_epo = param_group['lr']
    
    loss_avg = 0.
    acc_avg = 0.
    start_t = t = time.time()
    model = model.cuda(args.gpu)
    model.train()  

    ce_criterion = torch.nn.CrossEntropyLoss()
    for it, (images, labels, num) in enumerate(train_loader):
        if args.gpu >= 0:
            images = images.cuda(args.gpu)
            images = images.float()
            labels = labels.cuda(args.gpu)
            labels = labels.long()
        B, C, H, W = images.shape
        optimizer.zero_grad()               

        pred = model(images)

        if isinstance(pred, tuple):
            pred_2 = pred[1]
            pred_lay2 = pred[2]
            pred_lay1 = pred[3]
            mid_lay2_ori = pred[4]
            mid_lay2_ined = pred[5]
            mid_lay1_ori = pred[6]
            mid_lay1_ined = pred[7]
            mid_lay2_iwed = pred[8]
            mid_lay1_iwed = pred[9]
            pred = pred[0]

        a_ = torch.ones(labels.size()[0],labels.size()[1],labels.size()[2],dtype=torch.long)*args.n_class
        a_ = a_.cuda(args.gpu)
        y_ = torch.where(labels==-1, a_, labels).cuda(args.gpu)
        gt_one_hot = get_one_hot(y_, args.n_class+1).cuda(args.gpu)

        outs_lay2 = []
        for i in args.selected_classes:
            mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
            mask = F.interpolate(mask, size=mid_lay2_ori.size()[2:],mode='nearest')
            out = mid_lay2_ori * mask
            out = model.SAN_stage_2.IN(out)
            outs_lay2.append(out)
        mid_lay2_label = sum(outs_lay2)
        mid_lay2_label = model.SAN_stage_2.relu(mid_lay2_label)
        
        outs_lay1 = []
        for i in args.selected_classes:
            mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
            mask = F.interpolate(mask, size=mid_lay1_ori.size()[2:], mode='nearest')
            out = mid_lay1_ori * mask
            out = model.SAN_stage_1.IN(out)
            # out = nn.InstanceNorm2d(512, affine=True)(out)
            outs_lay1.append(out)
        mid_lay1_label = sum(outs_lay1)
        mid_lay1_label = model.SAN_stage_1.relu(mid_lay1_label)

                    # loss
        loss_main = ce_criterion(pred, labels)
        loss_lay2 = 0.1 * ce_criterion(pred_lay2, labels)
        loss_lay1 = 0.1 * ce_criterion(pred_lay1, labels)
        loss_in_lay2 = 0.1 * F.smooth_l1_loss(mid_lay2_ined, mid_lay2_label)
        loss_in_lay1 = 0.1 * F.smooth_l1_loss(mid_lay1_ined, mid_lay1_label)
        loss_iw_lay2 = 0.1 * mid_lay2_iwed
        loss_iw_lay1 = 0.1 * mid_lay1_iwed
        total_loss = loss_main + loss_lay2 + loss_lay1 + loss_in_lay2 + loss_in_lay1 + loss_iw_lay2 + loss_iw_lay1

        total_loss.backward()
        optimizer.step()
        acc,_ = calculate_accuracy(pred, labels, args.n_class)
        
        acc_avg += float(acc)

        cur_t = time.time()
        if cur_t - t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f, acc: %.4f' \
                  % (
                  epo, args.epoch_max, it + 1, train_loader.n_iter, (it + 1) * args.batch_size / (cur_t - start_t), float(total_loss),
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
            logits1, _, _, _, _, _, _, _, _, _ = model(images)
            
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
    Model = eval(model_name_all)(num_classes=args.n_class, args=args)
    optimizer, lr_scheduler = prepare_training(args, Model)

    if args.gpu >= 0: Model.cuda(args.gpu)

    train_dataset = huanghe_dataset(
        map_dir=args.train_map, 
        map_seffix=args.map_seffix, 
        label_dir=args.train_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=True
        )

    val_dataset = huanghe_dataset(
        map_dir=args.val_map, 
        map_seffix=args.label_seffix, 
        label_dir=args.val_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=True
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
    Model = eval(model_name_all)(num_classes=args.n_class, args=args)
    # model_dir = '/zbssd/yuyu/code/Domain_torch/weights/compareadd/sansaw/compare_sansaw_1/SANSAW_epo11_tacc0.9252_vacc0.7281_vmiou0.4683.pth'
    
    # Model.load_state_dict(torch.load(model_dir))
    if args.gpu >= 0: Model.cuda(args.gpu)

    val_dataset = huanghe_dataset(
        map_dir=args.val_map, 
        map_seffix=args.label_seffix, 
        label_dir=args.val_label, 
        label_seffix=args.label_seffix, 
        is_index=False, 
        is_train=True,
        transform=[]
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
            logits, _, _, _, _, _, _, _, _, _ = Model(images)
            
            loss = ce_criterion(logits, labels)
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
    cv2.imwrite("/zbssd/yuyu/code/Domain_torch/picture/sansawadd.png", segmented_image)

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
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
