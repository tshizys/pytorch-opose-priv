import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import random
import time
import shutil
from torch.autograd import Variable
from scipy.misc import imresize, imsave
from utils import Logger, dataset, transforms, measure_model
from config import cfg, cfg_from_file, cfg_from_list
from models.openpose import pose_estimation

parser = argparse.ArgumentParser(description='PyTorch Model Training')
parser.add_argument('--gpu', default=[0,1], nargs='+', type=int,
                    dest='gpu', help='the gpu used')
parser.add_argument('--pretrained', default=None,type=str,
                    dest='pretrained', help='the path of pretrained model')
parser.add_argument('--root', default=None, type=str,
                    dest='root', help='the root of images')
parser.add_argument('--train_dir', nargs='+', type=str,
                    dest='train_dir', help='the path of train file')
parser.add_argument('--val_dir', default=None, nargs='+', type=str,
                    dest='val_dir', help='the path of val file')
parser.add_argument('--num_classes', default=1000, type=int,
                    dest='num_classes', help='num_classes (default: 1000)')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional cfg.OPOSE file',
                    default='./cfgs/opose/vgg19s8_cpm6s.yml', type=str)
parser.add_argument('--set', dest='set_cfgs',
                    help='set cfg.OPOSE keys', default=None,
                    nargs=argparse.REMAINDER);
args = parser.parse_args()
print('==> Called with args:')
print(args)
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
print('==> Using config:')
print(cfg.OPOSE)
# Use CUDA
gpu = []
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_ids
print cfg.gpu_ids.split(',')
gpu.append(i for i in cfg.gpu_ids.split(','))
print gpu
USE_CUDA = torch.cuda.is_available()
# Random seed
if cfg.rng_seed is None:
    cfg.rng_seed = random.randint(1, 10000)
random.seed(cfg.rng_seed)
torch.manual_seed(cfg.rng_seed)
# Global param
NUM_GPUS = len(cfg.gpu_ids.split(','))
BATCH_SIZE = NUM_GPUS * cfg.OPOSE.batch_size_per_gpu
HEAT_LOSS_WEIGHT = (cfg.OPOSE.input_size // cfg.OPOSE.stride) * (cfg.OPOSE.input_size // cfg.OPOSE.stride) * cfg.OPOSE.num_points / 2.0  # for convenient to compare with origin code
VEC_LOSS_WEIGHT = (cfg.OPOSE.input_size // cfg.OPOSE.stride) * (cfg.OPOSE.input_size // cfg.OPOSE.stride) * cfg.OPOSE.num_limbs  # for convenient to compare with origin code
BEST_LOSS = cfg.OPOSE.best_loss  # best test loss


class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    if epoch in schedule:
        cfg.OPOSE.base_lr *= cfg.OPOSE.gamma
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = cfg.OPOSE.base_lr
    return cfg.OPOSE.base_lr

def save_checkpoint(model, optimizer, cur_valloss, epoch):
    global BEST_LOSS
    suffix_latest = 'latest.pth.tar'
    suffix_best = 'best.pth.tar'
    print('==> Saving checkpoints...')
    dict_model = model.state_dict()
    state_model = {'epoch': epoch + 1, 'state_dict': dict_model, 'loss': cur_valloss, 'best_loss': BEST_LOSS,
                     'optimizer': optimizer.state_dict()}
    torch.save(state_model, cfg.OPOSE.ckpt + 'latest.pth.tar')
    if cur_loss < BEST_LOSS:
        # update BEST_LOSS
        BEST_LOSS = cur_loss
        shutil.copyfile('{}/{}'.format(cfg.OPOSE.ckpt, suffix_latest),
                        '{}/{}'.format(cfg.OPOSE.ckpt, suffix_best))

def get_parameters(model, config, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': cfg.OPOSE.base_lr},
            {'params': lr_2, 'lr': cfg.OPOSE.base_lr * 2.},
            {'params': lr_4, 'lr': cfg.OPOSE.base_lr * 4.},
            {'params': lr_8, 'lr': cfg.OPOSE.base_lr * 8.}]

    return params, [1., 2., 4., 8.]

# train one epoch
def train(trainloader, model, criterion, optimizer, epoch, use_cuda=True):
    # switch to train mode
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(12)]
    end = time.time()
    for batch_idx, (input, heatmap, vecmap, mask) in enumerate(trainloader):
        data_time.update(time.time() - end)

        heatmap = heatmap.cuda(async=True)
        vecmap = vecmap.cuda(async=True)
        mask = mask.cuda(async=True)
        input_var = torch.autograd.Variable(input) #1x3x48x48
        heatmap_var = torch.autograd.Variable(heatmap) #1x19x46x46
        vecmap_var = torch.autograd.Variable(vecmap) #1x38x46x46
        mask_var = torch.autograd.Variable(mask) #1x1x46x46
        # print input_var.size(),heatmap_var.size(),vecmap_var.size(),mask_var.size() 
        vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)
        loss1_1 = criterion(vec1, vecmap_var) * VEC_LOSS_WEIGHT
        loss1_2 = criterion(heat1, heatmap_var) * HEAT_LOSS_WEIGHT
        loss2_1 = criterion(vec2, vecmap_var) * VEC_LOSS_WEIGHT
        loss2_2 = criterion(heat2, heatmap_var) * HEAT_LOSS_WEIGHT
        loss3_1 = criterion(vec3, vecmap_var) * VEC_LOSS_WEIGHT
        loss3_2 = criterion(heat3, heatmap_var) * HEAT_LOSS_WEIGHT
        loss4_1 = criterion(vec4, vecmap_var) * VEC_LOSS_WEIGHT
        loss4_2 = criterion(heat4, heatmap_var) * HEAT_LOSS_WEIGHT
        loss5_1 = criterion(vec5, vecmap_var) * VEC_LOSS_WEIGHT
        loss5_2 = criterion(heat5, heatmap_var) * HEAT_LOSS_WEIGHT
        loss6_1 = criterion(vec6, vecmap_var) * VEC_LOSS_WEIGHT
        loss6_2 = criterion(heat6, heatmap_var) * HEAT_LOSS_WEIGHT
        
        loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2
        for cnt, l in enumerate([loss1_1, loss1_2, loss2_1, loss2_2, loss3_1, loss3_2, loss4_1, loss4_2, loss5_1, loss5_2, loss6_1, loss6_2]):
            losses_list[cnt].update(l.data[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.data[0], input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_idx+1) % cfg.OPOSE.display == 0:

            print('Train Iteration: {0}\t' 'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t''Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                  'Learning rate = {2}\n'
                  'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(batch_idx+1, cfg.OPOSE.display, cfg.OPOSE.base_lr, batch_time=batch_time, data_time=data_time, loss=losses))
            for cnt in range(0,12,2):
                print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                    'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1, loss1=losses_list[cnt], loss2=losses_list[cnt + 1]))
            print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())

            batch_time.reset()
            data_time.reset()
            losses.reset()
            for cnt in range(12):
                losses_list[cnt].reset()
    return losses.avg

def test(valloader, model, criterion, optimizer, epoch, use_cuda=True):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_list = [AverageMeter() for i in range(12)]
    end = time.time()
    for j, (input, heatmap, vecmap, mask) in enumerate(valloader):

        heatmap = heatmap.cuda(async=True)
        vecmap = vecmap.cuda(async=True)
        mask = mask.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        heatmap_var = torch.autograd.Variable(heatmap, volatile=True)
        vecmap_var = torch.autograd.Variable(vecmap, volatile=True)
        mask_var = torch.autograd.Variable(mask, volatile=True)

        vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)
        loss1_1 = criterion(vec1, vecmap_var) * VEC_LOSS_WEIGHT
        loss1_2 = criterion(heat1, heatmap_var) * HEAT_LOSS_WEIGHT
        loss2_1 = criterion(vec2, vecmap_var) * VEC_LOSS_WEIGHT
        loss2_2 = criterion(heat2, heatmap_var) * HEAT_LOSS_WEIGHT
        loss3_1 = criterion(vec3, vecmap_var) * VEC_LOSS_WEIGHT
        loss3_2 = criterion(heat3, heatmap_var) * HEAT_LOSS_WEIGHT
        loss4_1 = criterion(vec4, vecmap_var) * VEC_LOSS_WEIGHT
        loss4_2 = criterion(heat4, heatmap_var) * HEAT_LOSS_WEIGHT
        loss5_1 = criterion(vec5, vecmap_var) * VEC_LOSS_WEIGHT
        loss5_2 = criterion(heat5, heatmap_var) * HEAT_LOSS_WEIGHT
        loss6_1 = criterion(vec6, vecmap_var) * VEC_LOSS_WEIGHT
        loss6_2 = criterion(heat6, heatmap_var) * HEAT_LOSS_WEIGHT
        
        loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2
        # record loss
        losses.update(loss.data[0], input.size(0))
        for cnt, l in enumerate([loss1_1, loss1_2, loss2_1, loss2_2, loss3_1, loss3_2, loss4_1, loss4_2, loss5_1, loss5_2, loss6_1, loss6_2]):
            losses_list[cnt].update(l.data[0], input.size(0))
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    print(
        'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
        'Loss {loss.avg:.8f}\n'.format(
        batch_time=batch_time, loss=losses))
    for cnt in range(0,12,2):
        print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
            'Loss{1}_2 = {loss2.val:.8f} (ave = {loss2.avg:.8f})'.format(cnt / 2 + 1, cnt / 2 + 1, loss1=losses_list[cnt], loss2=losses_list[cnt + 1]))
    print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())

    batch_time.reset()
    losses.reset()
    for cnt in range(12):
        losses_list[cnt].reset()
    return losses.avg


def main():
    global BEST_LOSS
    cudnn.benchmark = True
    start_epoch = cfg.OPOSE.start_epoch  # start from epoch 0 or last checkpoint epoch
    # Create ckpt & vis folder
    if not os.path.isdir(cfg.OPOSE.ckpt):
        os.makedirs(cfg.OPOSE.ckpt)
    if not os.path.exists(os.path.join(cfg.OPOSE.ckpt, 'vis')):
        os.makedirs(os.path.join(cfg.OPOSE.ckpt, 'vis'))
    if args.cfg_file is not None and not cfg.OPOSE.evaluate:
        shutil.copyfile(args.cfg_file, os.path.join(cfg.OPOSE.ckpt, args.cfg_file.split('/')[-1]))
    model = pose_estimation.PoseModel(num_point=19, num_vector=19, pretrained=True)
    # # Calculate FLOPs & Param
    # n_flops, n_convops, n_params = measure_model(model, cfg.OPOSE.input_size, cfg.OPOSE.input_size)
    criterion = nn.MSELoss().cuda()
    # Dataset and Loader
    train_dataset = dataset.CocoOpenposeData(cfg, cfg.OPOSE.data_root, cfg.OPOSE.info_root, 'train2017',
                                        transformer=transforms.Compose([transforms.RandomResized(),
                                        transforms.RandomRotate(40),
                                        transforms.RandomCrop(368),
                                        transforms.RandomHorizontalFlip(),
                                        ]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                               num_workers=int(cfg.workers), pin_memory=True)
    if cfg.OPOSE.validate or cfg.OPOSE.evaluate:
        val_dataset = dataset.CocoOpenposeData(cfg, cfg.OPOSE.data_root, cfg.OPOSE.info_root, 'val2017',
                                       transformer=transforms.Compose([transforms.RandomResized(cfg.OPOSE.input_size)]))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                 num_workers=int(cfg.workers), pin_memory=True)
    # Load nets into gpu
    if NUM_GPUS > 1:
        model = torch.nn.DataParallel(model, device_ids= args.gpu).cuda()
    # Set up optimizers
    params, multiple = get_parameters(model, cfg, False)
    
    optimizer = torch.optim.SGD(params, cfg.OPOSE.base_lr, momentum=cfg.OPOSE.momentum,
                                weight_decay=cfg.OPOSE.weight_decay)

    # Resume training
    title = 'Pytorch-OPOSE-{}-{}'.format(cfg.OPOSE.arch_encoder, cfg.OPOSE.arch_decoder)
    if cfg.OPOSE.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint '{}'".format(cfg.OPOSE.resume))
        assert os.path.isfile(cfg.OPOSE.resume), 'Error: no checkpoint directory found!'
        ckpt = torch.load(cfg.OPOSE.resume)
        BEST_LOSS = ckpt['best_loss']
        start_epoch = ckpt['epoch']
        try:
            model[0].module.load_state_dict(ckpt['state_dict'])
        except:
            model[0].load_state_dict(ckpt['state_dict'])

        optimizer.load_state_dict(ckpt['optimizer'])
        logger = Logger(os.path.join(cfg.OPOSE.ckpt, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(cfg.OPOSE.ckpt, 'log.txt'), title=title)
        logger.set_names(['epoch','Learning Rate', 'Train Loss', 'Valid Loss'])

    # Train and val
    for epoch in range(start_epoch, cfg.OPOSE.epochs):
        print('\nEpoch: [{}/{}] | LR: {:.8f} '.format(epoch + 1, cfg.OPOSE.epochs, cfg.OPOSE.base_lr))
        train_loss = train(train_loader, model, criterion, optimizer, epoch, USE_CUDA)
        if cfg.OPOSE.validate:
            test_loss = test(val_loader, model, criterion, optimizer, epoch, USE_CUDA)
        else:
            test_loss = 0.0, 0.0

        # Append logger file
        logger.append([epoch, cfg.OPOSE.base_lr, train_loss, test_loss])
        # Save model
        save_checkpoint(model, optimizer, test_loss, epoch)
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)
        # Draw curve
        try:
            draw_curve('model', cfg.OPOSE.ckpt)
            print('==> Success saving log curve...')
        except:
            print('==> Saving log curve error...')

    logger.close()
    try:
        savefig(os.path.join(cfg.OPOSE.ckpt, 'log.eps'))
        shutil.copyfile(os.path.join(cfg.OPOSE.ckpt, 'log.txt'), os.path.join(cfg.OPOSE.ckpt, 'log{}.txt'.format(
            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
    except:
        print('Copy log error.')
    print('==> Training Done!')
    print('==> Best acc: {:.4f}%'.format(BEST_LOSS))


if __name__ == '__main__':
    main()
