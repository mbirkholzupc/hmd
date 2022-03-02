from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
import datetime
import pickle
from data_loader import dataloader_anchor_btfm
from utility import take_notes
from utility import save_to_img
from utility import save_batch_tensors
from network import anchor_net_vgg16
from renderer import SMPLRenderer
from torch.utils.tensorboard import SummaryWriter

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--sil', type = bool, 
                    help = 'switch to silhouette version', default = False)
parser.add_argument('--workers', type = int, 
                    help = 'number of data loading workers', default = 16)
parser.add_argument('--batchSize', type = int, default = 20, 
                    help = 'input batch size')
parser.add_argument('--imageSize', type = int, default = [224, 224], 
                    help = 'the height / width of the input image to network')
parser.add_argument('--nepoch', type = int, default = 12, 
                    help = 'number of epochs to train for')
parser.add_argument('--niter', type = int, default = 20000, 
                    help = 'number of iterations to train for')
parser.add_argument('--lr', type = float, default = 0.0001, 
                    help = 'learning rate, default=0.0001')
parser.add_argument('--beta1', type = float, default = 0.9, 
                    help = 'beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = True,
                    help = 'enables cuda')
parser.add_argument('--ngpu', type = int, default = 1, 
                    help = 'number of GPUs to use')
parser.add_argument('--finetune', default = '', 
                    help = "path to net (to continue training)")
parser.add_argument('--outf', default = '../model/snapshots/', 
                    help = 'folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type = int, default = 1234, 
                    help = 'manual seed')
parser.add_argument('--testInterval', type = int, default = 5000, 
                    help = 'test interval')
parser.add_argument('--prvInterval', type = int, default = 1, 
                    help = 'preview interval')
parser.add_argument('--shlInterval', type = int, default = 1, 
                    help = 'show loss interval')
parser.add_argument('--saveModelIter', type = int, default = 20000, 
                    help = 'save model interval')
opt = parser.parse_args()
print(opt)

# get current time
c_time = datetime.datetime.now()
time_string = "%s-%02d:%02d:%02d" % (c_time.date(), c_time.hour, c_time.minute, c_time.second)

# make directory for output
if opt.sil is False:
    opt.outf = opt.outf[:-1] + "_a_" + time_string + "/"
else:
    opt.outf = opt.outf[:-1] + "_a_sil_" + time_string + "/"
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
    print("New directory for output is built: "+opt.outf)

# generate random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# write log file
f_log = open(opt.outf+"log.txt",'w')
f_log.write("time: %s\r\n" % time_string)
for arg in vars(opt):
    f_log.write("%s: %s\r\n" % (arg, getattr(opt, arg)))

# remind to use cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# prepare for cuda
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)

# get dataset class
dataset = dataloader_anchor_btfm(train = True,
                            sil_ver = opt.sil,
                            manual_seed = opt.manualSeed,
                            transform = transforms.Compose([
                                #transforms.RandomCrop(224, padding=15),
                                #transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ])
                           )

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=opt.batchSize, 
                                         shuffle=True, 
                                         num_workers=int(opt.workers))

# transfer model to device (GPU or CPU)
net_anchor = anchor_net_vgg16(in_channels = dataset[0][0].shape[0],
                              init_weights = True,
                             ).train().to(device)

if opt.finetune != '':
    net_anchor.load_state_dict(torch.load(opt.finetune))

# define the optimizer and criterion
optimizer = optim.Adam(net_anchor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99), weight_decay=0.0005)
criterion = nn.MSELoss()

# prepare for loss saving
batch_num = dataloader.dataset.num/dataloader.batch_size
take_notes("===Loss===|=Epoch=|==Iter==", opt.outf+"loss.txt", create_file = True)

# Tensorboard
writer=SummaryWriter(opt.outf+'/tensorboard')

# start training
for epoch in range(opt.nepoch):
    loss_acc=0
    loss_samples=0
    for i, data in enumerate(dataloader, 0):
        
        # load data
        input_arr = data[0].to(device).float()
        tgt_para = data[1].to(device).float()
        
        # forward and backward propagate
        optimizer.zero_grad()
        pred_para = net_anchor(input_arr)
        pred_para[tgt_para==0] = 0 # mask out zero pixel
        loss = criterion(pred_para, tgt_para)
        loss_acc += float(loss)
        loss_samples += data[1].shape[0]

        loss.backward()
        optimizer.step()
        
        ## show loss, save loss
        #if i%opt.shlInterval==0:
        #    print("step: %d/%d, loss: %f"\
        #          % (i+epoch*batch_num, opt.niter, loss))
        #    take_notes("\r\n%10f %7d %8d"\
        #               % (loss, epoch, i), opt.outf+"loss.txt")
        ## show loss, save loss
        if i%opt.shlInterval==0:
            print("epoch: %d/%d, step: %d/%d, loss: %f"\
                  % (epoch, opt.nepoch, i, opt.niter, loss))
            take_notes("\r\n%10f %7d %8d"\
                       % (loss, epoch, i), opt.outf+"loss.txt")
        
        # save parameters for lasllt epoch
        if (epoch == opt.nepoch-1) and (i == opt.niter-1):
            torch.save(net_anchor.state_dict(), 
                       "%spretrained_anchor.pth" % opt.outf)
            break

        if (i == opt.niter-1):
            break

        # save parameters for each epoch
        #if i%opt.saveModelIter == 0:
        #    torch.save(net_anchor.state_dict(), 
        #               "%sanchor_epoch_%d_%d.pth" % (opt.outf, epoch, i))

    # At end of epoch
    avg_loss = loss_acc/loss_samples
    writer.add_scalar('Loss/train', avg_loss, epoch)
    print('Avg loss: ' + str(avg_loss))
    
writer.flush()
writer.close()

print("Done, final model saved to %spretrained_anchor.pth" % opt.outf)
