import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import ResNet , TFNet
from data import get_training_set, get_test_set
from TFNetV21 import TFNet as TFNet2
import random
import os
import time

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=96, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', default = True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--dataset', type=str, default='I:/个人文档/西北大学/图像融合/论文/TFNet/tfnet_pytorch-master/data')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--net", type=str, choices={'resnet','tfnet'}, default = 'tfnet')
parser.add_argument("--log", type=str, default="I:/个人文档/西北大学/图像融合/论文/TFNet/tfnet_pytorch-master/log/")
parser.add_argument('--device',type = int, default=0, help = 'GPU device. Default = 0')
opt = parser.parse_args()

def main():
    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    print(opt.dataset)
    print('===> Loading datasets')
    train_set = get_training_set(opt.dataset, opt.device)
    test_set = get_test_set(opt.dataset, opt.device)

    training_data_loader = Data.DataLoader(
        train_set, 
        num_workers=opt.threads, 
        batch_size=opt.batchSize, 
        shuffle=True)
    

    test_data_loader = Data.DataLoader(
        test_set, 
        num_workers=opt.threads, 
        batch_size=opt.testBatchSize, 
        shuffle=False)

    print("===> Building model")
    if (opt.net == 'resnet'):
        model = ResNet()
    else:
        model = TFNet2()
    criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")

    # optimizer set: SDG or Adam
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.5, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.5))

    print("===> Training")
    t = time.strftime("%Y%m%d%H%M")
    train_log = open(os.path.join(opt.log, "%s_%s_train.log") %
                     (opt.net, t), "w")
    test_log = open(os.path.join(opt.log, "%s_%s_test.log") %
                    (opt.net, t), "w")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer,
              model, criterion, epoch, train_log)
        if epoch % 10 == 0:
            test(test_data_loader, model, criterion, epoch, test_log)
            save_checkpoint(model, epoch, t)
    train_log.close()
    test_log.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr


def train(training_data_loader, optimizer, model, criterion, epoch, train_log):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print ("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        pan_patches, lr_hsi_patches, hsi_gt_patches = Variable(batch[0]), Variable(batch[1]),Variable(batch[2], requires_grad=False)
        lr_hsi_patches = torch.nn.functional.interpolate(lr_hsi_patches, size = [lr_hsi_patches.shape[2] * 4,lr_hsi_patches.shape[3] * 4], mode='bilinear', align_corners=True)

        if opt.cuda:
            pan_patches = pan_patches.cuda()
            lr_hsi_patches = lr_hsi_patches.cuda()
            hsi_gt_patches = hsi_gt_patches.cuda()
        
        if(opt.net=="resnet"):
            output = model(pan_patches, lr_hsi_patches)
        else:
            output = model(pan_patches, lr_hsi_patches)
        loss = criterion(output, hsi_gt_patches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_log.write("{} {:.10f}\n".format((epoch-1)*len(training_data_loader)+iteration, loss.item()))
        if iteration%10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader),
                                                                loss.item()))

def test(test_data_loader, model, criterion, epoch, test_log):
    avg_l1 = 0
    model.eval()
    for index,batch in enumerate(test_data_loader):
        pan_patches, lr_hsi_patches, hsi_gt_patches = Variable(batch[0],volatile=True), Variable(batch[1],volatile=True), Variable(batch[2],volatile=True)
        # 如果输入图像需要进行上采样，则使用这个代码对输入的光谱图像进行上采样，使得RGB/PAN图像与MS图像保持一致
        lr_hsi_patches = torch.nn.functional.interpolate(lr_hsi_patches, size = [lr_hsi_patches.shape[2] * 4,lr_hsi_patches.shape[3] * 4], mode='bilinear', align_corners=True)

        if opt.cuda:
            pan_patches = pan_patches.cuda()
            lr_hsi_patches = lr_hsi_patches.cuda()
            hsi_gt_patches = hsi_gt_patches.cuda()

        if(opt.net=="resnet"):
            output = model(pan_patches, lr_hsi_patches)
        else:
            output = model(pan_patches, lr_hsi_patches)
        loss = criterion(output, hsi_gt_patches)
        avg_l1 += loss.item()
    del (pan_patches, lr_hsi_patches, hsi_gt_patches, output)
    test_log.write("{} {:.10f}\n".format((epoch-1), avg_l1 / len(test_data_loader)))
    print("===>Epoch{} Avg. L1: {:.4f} ".format(epoch, avg_l1 / len(test_data_loader)))

def save_checkpoint(model, epoch, t):
    model_out_path = "model/{}_{}/model_epoch_{}.pth".format(opt.net,t,epoch)
    state = {"epoch": epoch, "model": model}

    if not os.path.exists("model/{}_{}".format(opt.net, t)):
        os.makedirs("model/{}_{}".format(opt.net, t))

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()