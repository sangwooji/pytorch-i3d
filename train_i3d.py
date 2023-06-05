import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='2'
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-init_lr', type=float, help='initial lr')
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)

args = parser.parse_args()


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from dataset import UCF101Dataset

# Change batch size 40 --> 16 --> 8
def run(init_lr=0.1, max_steps=4e3, mode='rgb', train_split='charades/charades.json', batch_size=8*5, save_model=''):
    scale_factor = 5
    batch_size = int(batch_size / scale_factor)
    max_steps = max_steps * scale_factor

    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                       videotransforms.RandomHorizontalFlip(),
    # ])
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset =  UCF101Dataset(
        dataset_path="data/UCF-101-frames",
        split_path="data/ucfTrainTestlist",
        split_number=1,
        input_shape=(3,224,224),
        sequence_length=64,
        training=True,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset =  UCF101Dataset(
        dataset_path="data/UCF-101-frames",
        split_path="data/ucfTrainTestlist",
        split_number=1,
        input_shape=(3,224,224),
        sequence_length=64,
        training=False,
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        print ("Load pre-trained Flow")
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        print ("Load pre-trained RGB")
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(101)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
#     i3d = nn.DataParallel(i3d)

    lr = init_lr
    # optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [2000, 3000])


    num_steps_per_update = 4 * scale_factor # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print (f'Step {steps}/{max_steps}')
        print ('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            total = correct = 0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            for epoch, data in enumerate(dataloaders[phase]):
                num_iter += 1
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
#                 tot_loc_loss += loc_loss.data[0]
                tot_loc_loss += loc_loss.item()

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
#                 tot_cls_loss += cls_loss.data[0]
                tot_cls_loss += cls_loss.item()
    
                _, pred = torch.max(per_frame_logits, dim=2)[0].max(1)
                _, gt = labels[:,:,0].max(1)
                total += pred.shape[0]
                correct += pred.eq(gt).sum().item()

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print (lr_sched.get_lr(), '%d Loc Loss: %.4f Cls Loss: %.4f Tot Loss: %.4f acc: %.3f' % (steps, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10, correct/total))
                    if steps % (500) == 0:
                        print ('%s Loc Loss: %.4f Cls Loss: %.4f Tot Loss: %.4f' % (phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(i3d.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print ('%s Loc Loss: %.4f Cls Loss: %.4f Tot Loss: %.4f acc: %.3f' % (phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter, correct/total))

                torch.save(i3d.state_dict(), save_model+str(epoch).zfill(6)+'.pt')


if __name__ == '__main__':
    # need to add argparse
    run(init_lr=args.init_lr, mode=args.mode, save_model=args.save_model)
