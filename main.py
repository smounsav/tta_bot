"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent and EATA code.
"""
from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
from pycm import *

import math
from typing import ValuesView

from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

import torch    
import torch.nn.functional as F

import tent
import eata
import sar
from sam import SAM
import timm
import delta
import moba

import models.Res as Resnet
from models.network import ResBase, feat_bootleneck, feat_classifier # for modified ResNet101
from robustbench.utils import load_model # for WRN2810
from utee import selector # for SVHN to digits scenarios

from models.resnet_cifar import resnet26_cifar

def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            # images, target = dl[0], dl[1]
            images, target = dl[0], dl[1] - 1 # svhn
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            if args.corruption == 'rendition':
                output = model(images, rendition_mask=val_loader.dataset.imagenet_r_mask)
            else:
                output = model(images)
            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # measure accuracy on all dataset
            # if (i == 0) or (i % 40 ==0):
            #     new_val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=2048, shuffle=False,
            #                                         num_workers=args.workers, pin_memory=False)
            #     correct = 0
            #     total = 0
            #     model.eval()
            #     with torch.no_grad():
            #         for image, label in new_val_loader:
            #             image = image.to('cuda', non_blocking=True)
            #             label = label.to('cuda', non_blocking=True)
            #             out = model.forward_only(image)
            #             _, predicted = out.max(1)
            #             total += label.size(0)
            #             correct += (predicted == label).sum().item()
            #     print(i,' : ',100 * correct / total)


            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
    return top1.avg, top5.avg



def get_args():

    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=64, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar, moba')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm or resnet101')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # Common model parameters
    parser.add_argument('--optim_type', default='sgd', type=str, help='sgd, adam')
    parser.add_argument('--optim_lr', default=0.00025, type=float)
    parser.add_argument('--optim_momentum', default=0.9, type=float)
    parser.add_argument('--optim_wd', default=0, type=float)
    parser.add_argument('--old_prior', default=0.95, type=float)

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(10)*0.4, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=int, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')

    # Moba parameters
    parser.add_argument('--moob_time_decay', default=0.99, type=float, help='time decay parameter for moob method')
    parser.add_argument('--temperature', default=10.0, type=float, help='temperature scaling parameter')
    parser.add_argument('--no_class_rebalancing', action=argparse.BooleanOptionalAction, help='if class rebalancing should be applied')
    parser.add_argument('--buffer_size', default=16, type=int, help='sample buffer size')

    # Delta parameters
    parser.add_argument('--loss_type', default='entropy', type=str, help='entropy, cross_entropy, poly_loss')
    parser.add_argument('--ent_w', action=argparse.BooleanOptionalAction, help='if sample selection is used')
    parser.add_argument('--norm_type', default='rn', type=str, help='bn_training, rn')
    parser.add_argument('--dot', default=0.95, type=float)
    parser.add_argument('--class_num', default='1000', type=int)

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

    if not os.path.exists(args.output): # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)


    args.logger_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-{}-{}-level{}-seed{}.txt".format(args.method, args.model, args.level, args.seed)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 
        
    
    # common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    common_corruptions =['frost']
    # common_corruptions = ['gaussian_noise', 'contrast']
    # common_corruptions = ['speckle_noise', 'spatter','gaussian_blur','saturate']

    if args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'no_adapt', 'eata', 'sar', 'delta', 'moba']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
    
    if args.exp_type == 'bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")

    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")


    acc1s, acc5s = [], []
    ir = args.imbalance_ratio
    if args.corruption in ['rendition' , 'visda-c', 'sketch' ]:
        common_corruptions = [args.corruption]
    for corrupt in common_corruptions:
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs

        if args.method in ['tent', 'eata', 'sar', 'no_adapt', 'delta', 'moba']:
            if args.corruption != 'mix_shifts':
                val_dataset, val_loader = prepare_test_data(args)
                if (args.corruption != 'visda-c') & ('CIFAR-10-C' not in args.data_corruption) & ('CIFAR-100-C' not in args.data_corruption) & ('MNIST' not in args.data_corruption) & ('MNIST-M' not in args.data_corruption) & ('USPS' not in args.data_corruption) & ('SVHN' not in args.data_corruption):
                    val_dataset.switch_mode(True, False)
        else:
            assert False, NotImplementedError
        # construt new dataset with online imbalanced label distribution shifts, see Section 4.3 for details
        # note that this operation does not support mix-domain-shifts exps
        if args.exp_type == 'label_shifts':
            logger.info(f"imbalance ratio is {ir}")
            if args.seed == 2021:
                indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
            else:
                indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000, ir)
            logger.info(f"label_shifts_indices_path is {indices_path}")
            indices = np.load(indices_path)
            val_dataset.set_specific_subset(indices.astype(int).tolist())
        
        # build model for adaptation
        if args.method in ['tent', 'eata', 'sar', 'no_adapt', 'delta', 'moba']:
            if args.model == "resnet50_gn_timm":
                net = timm.create_model('resnet50_gn', pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "vitbase_timm":
                net = timm.create_model('vit_base_patch16_224', pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model == "resnet50_bn_torch":
                net = Resnet.__dict__['resnet50'](pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "resnet101":
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                netF = ResBase(res_name='resnet101')
                netB = feat_bootleneck(type='bn',
                                            feature_dim=netF.in_features,
                                            bottleneck_dim=256)
                netC = feat_classifier(type='wn',
                                            class_num=args.class_num,
                                            bottleneck_dim=256)                  
                netF.load_state_dict(torch.load('source_F.pt')) # Model used in Wang et al. NeurIPS21
                netB.load_state_dict(torch.load('source_B.pt')) # Model used in Wang et al. NeurIPS21
                netC.load_state_dict(torch.load('source_C.pt')) # Model used in Wang et al. NeurIPS21

                net = nn.Sequential(netF, netB, netC)
            elif args.model == "resnet26":                
                    net = resnet26_cifar()     # Pretrained model
                    net.load_state_dict(torch.load(os.path.join('/home/saypra/Projects/ttada/state_dicts/','cifar10_resnet26_nonorm.pth')))
                    net = net.to('cuda')
                    # args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                    args.lr = 5e-3  * (args.test_batch_size / 64)
                    # args.lr = 2
            elif args.model == 'WRN2810':
                net = load_model(model_name='Standard',
                            dataset='cifar10', threat_model='corruptions')
                args.lr = 5e-3  * (args.test_batch_size / 64)                    
            elif args.model == 'WRN402':
                net = load_model(model_name='Hendrycks2020AugMix_WRN',
                            dataset='cifar10', threat_model='corruptions')
            elif args.model == 'WRN402_100':
                net = load_model(model_name='Hendrycks2020AugMix_WRN',
                            dataset='cifar100', threat_model='corruptions')                
                args.lr = 5e-3  * (args.test_batch_size / 64)
            elif args.model == 'MNIST':
                net, _, _ = selector.select('mnist')
                args.lr = 1e-3  * (args.test_batch_size / 64)     
            elif args.model == 'SVHN':
                net, _, _ = selector.select('svhn')
                args.lr = 1e-3  * (args.test_batch_size / 64)                
            else:
                assert False, NotImplementedError
            net = net.cuda()
        else:
            assert False, NotImplementedError

        if args.test_batch_size == 1 and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")

        logger.info(args)

        if args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = tent.Tent(net, optimizer, temperature=args.temperature)

            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adapttion accuracy of Tent is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "no_adapt":
            tented_model = net
            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. Original Accuracy (no adapt) is top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "eata":
            # compute fisher informatrix
            args.corruption = 'original'
            fisher_dataset, fisher_loader = prepare_test_data(args)
            fisher_dataset.set_dataset_size(args.fisher_size)
            fisher_dataset.switch_mode(True, False)

            net = eata.configure_model(net)
            params, param_names = eata.collect_params(net)
            # fishers = None
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    targets = targets.cuda(args.gpu, non_blocking=True)
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            logger.info("compute fisher matrices finished")
            del ewc_optimizer

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

            top1, top5 = validate(val_loader, adapt_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. After EATA Adapt: Accuracy: top1: {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
            end = time.time()
            for i, dl in enumerate(val_loader):
                # images, target = dl[0], dl[1]
                images, target = dl[0], dl[1] - 1 # svhn
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                if args.corruption == 'rendition':
                    output = adapt_model(images, rendition_mask=val_loader.dataset.imagenet_r_mask)
                else:
                    output = adapt_model(images)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(f"Result under {args.corruption}. The adaptation accuracy of SAR is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")

        elif args.method == "delta":
            adapted_model = delta.DELTA(args, net)

            top1, top5 = validate(val_loader, adapted_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of Delta is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")
        
        elif args.method == "moba":
            net = moba.configure_model(net)
            params, param_names = moba.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = moba.MOBA(net, optimizer, moob_factor=args.moob_time_decay, temperature=args.temperature, class_rebalancing=not args.no_class_rebalancing, buffer_size=args.buffer_size, loss_type=args.loss_type, class_num=args.class_num)

            top1, top5 = validate(val_loader, tented_model, None, args, mode='eval')
            logger.info(f"Result under {args.corruption}. The adaptation accuracy of Moba is top1 {top1:.5f} and top5: {top5:.5f}")

            acc1s.append(top1.item())
            acc5s.append(top5.item())

            logger.info(f"acc1s are {acc1s}")
            logger.info(f"acc5s are {acc5s}")            

        else:
            assert False, NotImplementedError
