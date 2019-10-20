import os
import uuid
import torch
import torch.nn as nn
import models
import argparse
import torchvision
import tensorboardX
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *
from ss import rotation, JigsawGenerator

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Selfie')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default="CUB")
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of steps of selfie')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-method', default='step', type=str,
                    help='method of learning rate')
parser.add_argument('--lr-params', default=[], dest='lr_params',nargs='*',type=float,
                    action='append', help='params of lr method')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=3e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
                    help='store checkpoint in every epoch')
parser.add_argument('--evaluation', action="store_true")
parser.add_argument('--resume', action="store_true")

parser.add_argument('--load-weights', default=None, type=str)
parser.add_argument('--task', type=str, default=uuid.uuid1())
parser.add_argument('--with-rotation', action="store_true")
parser.add_argument('--with-jigsaw', action="store_true")
parser.add_argument('--with-selfie', action="store_true")
parser.add_argument('--ignore-classification', action="store_true")
parser.add_argument('--seperate-layer4', action="store_true")
parser.add_argument('--rotation-aug', action="store_true")

parser.add_argument('--self-ensemble', action="store_true")

def main():
    global args, best_prec1, summary_writer, jigsaw

    jigsaw = JigsawGenerator(30)
    args = parser.parse_args()
    summary_writer = tensorboardX.SummaryWriter(os.path.join('logs', args.task))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    print(args)

    if args.dataset == 'CUB':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            #transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        num_classes = 200
        train_dataset = datasets.ImageFolder(traindir, train_transforms)
        val_dataset = datasets.ImageFolder(valdir, val_transforms)
    elif args.dataset == 'cifar':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(args.data, True, train_transforms, download = True)
        val_dataset   = datasets.CIFAR10(args.data, False, val_transforms, download = True)
        num_classes = 10
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last = True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last = True)

    if not args.self_ensemble:
        model = models.Model(args, num_classes)
    else:
        model = models.SelfEnsembleModel(args, 3)

    criterion = nn.CrossEntropyLoss().cuda()
    if args.gpu is None:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 150], 0.7)
    best_prec1 = 0

    if not os.path.exists(os.path.join('models', str(args.task))):
        os.mkdir(os.path.join('models', str(args.task)))

    for epoch in range(args.start_epoch, args.epochs):
        if not args.self_ensemble:
            trainObj, top1 = train(train_loader, model, criterion, optimizer, scheduler, epoch)
            valObj, prec1 = val(val_loader, model, criterion)
        else:
            trainObj, top1 = ensemble_train(train_loader, model, criterion, optimizer, scheduler, epoch)
            valObj, prec1 = ensemble_val(val_loader, model, criterion)
        summary_writer.add_scalar("train_loss", trainObj, epoch)
        summary_writer.add_scalar("test_loss", valObj, epoch)
        summary_writer.add_scalar("train_acc", top1, epoch)
        summary_writer.add_scalar("test_acc", prec1, epoch)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            torch.save(
                {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                }, os.path.join('models', str(args.task), 'checkpoint.pth.tar'))
            torch.save(
                {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                }, os.path.join('models', str(args.task), 'model_best.pth.tar'))
        else:
            torch.save(
                {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                }, os.path.join('models', str(args.task), 'checkpoint.pth.tar'))

def ensemble_train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global args
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for index, (input, target) in enumerate(train_loader):
        input = input.cuda(args.gpu)
        target = target.cuda(args.gpu)
            
        output = model([input] * 3)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.shape[0])
        top1.update(prec1[0].item(), input.shape[0])

        if index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, index, len(train_loader), loss=losses, top1=top1))
    scheduler.step()
    return losses.avg, top1.avg

def ensemble_val(val_loader, model, criterion):
    global args
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for index, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu)
            target = target.cuda(args.gpu)

            output = model([input] * 3)
            loss = criterion(output, target)

            prec1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), input.shape[0])
            top1.update(prec1[0].item(), input.shape[0])

            if index % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       index, len(val_loader), loss=losses, top1=top1))

    return losses.avg, top1.avg

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    global args
    losses = AverageMeter()
    top1rotation = AverageMeter()
    top1jigsaw = AverageMeter()
    top1selfie = AverageMeter()
    model.train()
    for index, (input, target) in enumerate(train_loader):
        input = input.cuda(args.gpu)
        target = target.cuda(args.gpu)

        jigsaw_stacked = None
        rotation_input = None
        selfie_input = None
        if args.with_jigsaw:
            if args.dataset == 'CUB':
                splited_list = split_image(input, 112)
            elif args.dataset == 'cifar':
                splited_list = split_image(input, 8)
            splited_list = [i.unsqueeze(1) for i in splited_list]
            jigsaw_stacked = torch.cat(splited_list, 1)
            jigsaw_stacked, jigsaw_target = jigsaw(jigsaw_stacked)
            jigsaw_stacked = combine_image(jigsaw_stacked, 4)
       

        if args.with_rotation:
            rotation_input = input
            rotation_input, rotation_target = rotation(input)


        if args.with_selfie:
            t, v, batches = get_index(input)            
            selfie_input = (batches, v, t)


        if args.rotation_aug:
            input = rotation_input
            
        output, rotation_output, jigsaw_output, selfie_output = model(input, rotation_input, jigsaw_stacked, selfie_input)
        if not args.ignore_classification:
            loss = criterion(output, target)
        else:
            loss = 0

        if args.with_rotation:
            loss = loss + criterion(rotation_output, rotation_target)
        if args.with_jigsaw:
            loss = loss + criterion(jigsaw_output, jigsaw_target)

        if args.with_selfie:
            patch_loss = 0
            output_encoder, features = selfie_output
            for i in range(len(t)):
                activate = output_encoder[:, i, :].unsqueeze(1)
                pre = torch.bmm(activate, features)
                logit = nn.functional.softmax(pre, 2).view(-1, len(t))
                temptarget = torch.ones(logit.shape[0]).cuda(args.gpu) * i
                temptarget = temptarget.long()
                loss_ = criterion(logit, temptarget)
                prec_selfie = accuracy(logit, temptarget, topk=(1,))
                top1selfie.update(prec_selfie[0].item(), input.shape[0])
                patch_loss += loss_

            loss = loss + patch_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec_rotation = accuracy(rotation_output, rotation_target, topk=(1,))
        prec_jigsaw = accuracy(jigsaw_output, jigsaw_target, topk=(1,))
        losses.update(loss.item(), input.shape[0])
        top1rotation.update(prec_rotation[0].item(), input.shape[0])
        top1jigsaw.update(prec_jigsaw[0].item(), input.shape[0])
        if index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PrecRotation@1 {top1rotation.val:.3f} ({top1rotation.avg:.3f})\t'
                  'PrecJigsaw@1 {top1jigsaw.val:.3f} ({top1jigsaw.avg:.3f})\t'
                  'PrecSelfie@1 {top1selfie.val:.3f} ({top1selfie.avg:.3f})\t'.format(
                       epoch, index, len(train_loader), loss=losses, top1rotation=top1rotation, top1jigsaw=top1jigsaw,top1selfie=top1selfie))

    scheduler.step()
    return losses.avg, (top1rotation.avg + top1jigsaw.avg) / 2

def val(val_loader, model, criterion):
    global args
    losses = AverageMeter()
    top1rotation = AverageMeter()
    top1jigsaw = AverageMeter()
    top1selfie = AverageMeter()
    model.eval()
    with torch.no_grad():
        for index, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu)
            target = target.cuda(args.gpu)
            jigsaw_stacked = None
            rotation_input = None
            selfie_input = None
            if args.with_jigsaw:
                if args.dataset == 'CUB':
                    splited_list = split_image(input, 112)
                elif args.dataset == 'cifar':
                    splited_list = split_image(input, 8)
                splited_list = [i.unsqueeze(1) for i in splited_list]
                jigsaw_stacked = torch.cat(splited_list, 1)
                jigsaw_stacked, jigsaw_target = jigsaw(jigsaw_stacked)
                jigsaw_stacked = combine_image(jigsaw_stacked, 4)

            target = target.cuda(args.gpu)

            if args.with_rotation:
                rotation_input = input
                rotation_input, rotation_target = rotation(input)

            if args.rotation_aug:
                input = rotation_input

            if args.with_selfie:
                t, v, batches = get_index(input)            
                selfie_input = (batches, v, t)


            output, rotation_output, jigsaw_output, selfie_output = model(input, rotation_input, jigsaw_stacked, selfie_input)
            if not args.ignore_classification:
                loss = criterion(output, target)
            else:
                loss = 0   

            if args.with_rotation:
                loss = loss + criterion(rotation_output, rotation_target)
            if args.with_jigsaw:
                loss = loss + criterion(jigsaw_output, jigsaw_target)

            if args.with_selfie:
                patch_loss = 0
                output_encoder, features = selfie_output
                for i in range(len(t)):
                    activate = output_encoder[:, i, :].unsqueeze(1)
                    pre = torch.bmm(activate, features)
                    logit = nn.functional.softmax(pre, 2).view(-1, len(t))
                    temptarget = torch.ones(logit.shape[0]).cuda(args.gpu) * i
                    temptarget = temptarget.long()
                    loss_ = criterion(logit, temptarget)
                    prec_selfie = accuracy(logit, temptarget, topk=(1,))
                    top1selfie.update(prec_selfie[0].item(), input.shape[0])
                    patch_loss += loss_

                loss = loss + patch_loss

            prec_rotation = accuracy(rotation_output, rotation_target, topk=(1,))
            prec_jigsaw = accuracy(jigsaw_output, jigsaw_target, topk=(1,))
            losses.update(loss.item(), input.shape[0])
            top1rotation.update(prec_rotation[0].item(), input.shape[0])
            top1jigsaw.update(prec_jigsaw[0].item(), input.shape[0])
            if index % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PrecRotation@1 {top1rotation.val:.3f} ({top1rotation.avg:.3f})\t'
                      'PrecJigsaw@1 {top1jigsaw.val:.3f} ({top1jigsaw.avg:.3f})\t'
                      'PrecSelfie@1 {top1selfie.val:.3f} ({top1selfie.avg:.3f})\t'.format(
                       index, len(val_loader), loss=losses, top1rotation=top1rotation, top1jigsaw=top1jigsaw, top1selfie=top1selfie))

    return losses.avg, (top1rotation.avg + top1jigsaw.avg) / 2

def accuracy(output, target, topk=(1,)):
    #print(output.shape)
    #print(target.shape)
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()