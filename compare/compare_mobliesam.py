# coding:utf-8
import argparse
import os
import time
import sys
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
import numpy as np
import cv2
def get_args_parser():
    project_name='test'
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=6, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * accum_iter * # gpus')
    parser.add_argument('--epoch_from', default=1, type=int)
    parser.add_argument('--epoch_max', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--n_class', default=8, type=int)
    parser.add_argument('--n_channels', default=4, type=int)

    # * Optimizer parameters
    parser.add_argument('--lr_scheduler', default='poly', type=str)
    parser.add_argument('--lr_start', default=1e-4, type=int)
    parser.add_argument('--lr_decay', default=0.97, type=float)
    parser.add_argument('--weight_decay', default=0.0003, type=float)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--lr_min', default=1.0e-7, type=int)
    # SGD
    parser.add_argument('--momentum', default=0.98, type=float)
    # Adam & AdamW
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    
    # * Dataset parameters
    # parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experimentadd/train/img/', type=str)
    # parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experimentadd/train/label/', type=str)
    # parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experimentadd/val_b/img/', type=str)
    # parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experimentadd/val_b/label/', type=str)
    parser.add_argument('--train_map', default=r'/zbssd/yuyu/code/data512/experimentadd/train/img/', type=str)
    parser.add_argument('--train_label', default=r'/zbssd/yuyu/code/data512/experimentadd/train/label/', type=str)
    parser.add_argument('--val_map', default=r'/zbssd/yuyu/code/data512/experimentadd/val/img/', type=str)
    parser.add_argument('--val_label', default=r'/zbssd/yuyu/code/data512/experimentadd/val/label/', type=str)
    parser.add_argument('--map_seffix', default='.npy', type=str)
    parser.add_argument('--label_seffix', default='.npy', type=str)
    parser.add_argument('--augmentation_methods', default=[], type=list)

    # * Project parameters
    parser.add_argument('--project_name', default=project_name, type=str)
    parser.add_argument('--model_name', default='SAM', type=str)

    # * SAM parameters
    parser.add_argument('--model_type', default='vit_t', type=str)

    # * Path
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--model_dir', default='weights/', type=str)
    parser.add_argument('--tensorboard_log_dir', default=f'weights/{project_name}/tensorboard/', type=str)
    
    return parser

def train(epo, model, train_loader, optimizer, args):
    # lr_this_epo = args.lr_start * args.lr_decay ** ((epo - 1) / 10 + 1)
    for param_group in optimizer.param_groups:
        lr_this_epo = param_group['lr']

    loss_avg = 0.
    acc_avg = 0.
    start_t = t = time.time()
    model = model.cuda(args.gpu)
    model.train()
    
    ce_criterion = torch.nn.CrossEntropyLoss()
    for it, (images, labels, names) in enumerate(train_loader):
        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        if args.gpu >= 0:
            images = images.cuda(args.gpu)
            images = images.float()
            labels = labels.cuda(args.gpu)
            labels = labels.long()
        optimizer.zero_grad()
        logits1 = model(images)
        loss1 = ce_criterion(logits1, labels)
        # loss2 = ce_criterion(logits2, labels)
        loss_all = loss1
        loss_all.backward()
        optimizer.step()

        acc,_ = calculate_accuracy(logits1, labels, args.n_class)
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
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images)
            labels = Variable(labels)
            if args.gpu >= 0:
                images = images.cuda(args.gpu)
                images = images.float()
                labels = labels.cuda(args.gpu)
                labels = labels.long()
            logits = model(images)
            loss1 = ce_criterion(logits, labels)
            loss = loss1
            acc, confusionmat_tmp = calculate_accuracy(logits, labels, args.n_class)
            confusionmat = confusionmat + confusionmat_tmp
            for i in range(logits.shape[0]):
                it_logit = logits[i]
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
                
    if args.model_type=='vit_h':
        model_checkpoint = './model/sam/weight/sam_vit_h_4b8939.pth'
    elif args.model_type == 'vit_l':
        model_checkpoint = './model/sam/weight/sam_vit_l_0b3195.pth'
    elif args.model_type == 'vit_b':
        model_checkpoint = './model/sam/weight/sam_vit_b_01ec64.pth'
    elif args.model_type == 'vit_t':
        model_checkpoint = './model/sam/weight/mobile_sam.pt'
    
    Model = sam_seg_model_registry[args.model_type](in_chans=args.n_channels, num_classes=args.n_class, checkpoint=model_checkpoint)
    optimizer, lr_scheduler = prepare_training(args, Model)

    if args.gpu >= 0: Model.cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            Model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_dataset = huanghe_dataset(map_dir=args.train_map, map_seffix=args.map_seffix, label_dir=args.train_label, label_seffix=args.label_seffix, is_index=False, is_train=True)

    val_dataset = huanghe_dataset(map_dir=args.val_map, map_seffix=args.label_seffix, label_dir=args.val_label, label_seffix=args.label_seffix, is_index=False, is_train=True)

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
    if args.model_type == 'vit_t':
        model_checkpoint = './model/sam/weight/mobile_sam.pt'
    Model = sam_seg_model_registry[args.model_type](in_chans=args.n_channels, num_classes=args.n_class, checkpoint=None)
    model_dir = '/zbssd/yuyu/code/Domain_torch/weights/compare2021/mobliesam/mobliesam_biaop/SAM_epo81_tacc0.9664_vacc0.9147_vmiou0.7228.pth'
    # model_dir = '/mnt/backup/zby/code/Domain_torch/weights/1e-4+0.95+poly_down_test2/SAM_epo11_tacc0.8849_vacc0.8223_vmiou0.5632.pth'

    with open(model_dir, "rb") as f:
        state_dict = torch.load(f)

    loaded_keys = {}
    for k in state_dict.keys():
        if k in Model.state_dict().keys() and 'prompt_encoder' not in k:
            loaded_keys[k] = state_dict[k]
    Model.load_state_dict(loaded_keys, strict=False)
    print("loaded keys:", loaded_keys.keys())
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
    map_H, map_W = 22698, 18928
    picture = np.zeros((map_H, map_W), dtype=int)
    color_map = {
            1: (255, 255, 255),  # 类别1的颜色，白色
            2: (0, 0, 255),       # 类别2的颜色，红色
            3: (0, 255, 0),       # 类别3的颜色，绿色
            4: (255, 197, 0),     # 类别4的颜色 淡蓝色
            5: (0, 255, 255),     # 类别5的颜色，黄色
            6: (255, 0, 255),     # 类别6的颜色，紫色
            7: (255, 255, 0),     # 类别7的颜色，青色
            8: (255, 0, 0)        # 类别8的颜色，蓝色
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
    cv2.imwrite("/zbssd/yuyu/code/Domain_torch/picture/mobliesam2021.png", segmented_image)

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
