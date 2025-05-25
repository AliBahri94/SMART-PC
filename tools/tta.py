from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
from utils.rotnet_utils import rotate_batch
import utils.tent_shot as tent_shot_utils
import utils.t3a as t3a_utils
from utils.misc import *
from torchvision import transforms
from datasets import data_transforms
import time
import numpy as np
import torch
from pytorch3d.ops import knn_points
import copy

import os
import matplotlib.pyplot as plt
from scipy.stats import norm

level = [5]



train_transforms_random = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
    ]
)


train_transforms_random_jitter = transforms.Compose(
    [
        data_transforms.PointcloudJitter(),
    ]
)


train_transforms_random_h_flip = transforms.Compose(
    [
        data_transforms.RandomHorizontalFlip(),
    ]
)

train_transforms_random_scale_trans = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

train_transforms_random_trans = transforms.Compose(
    [
        data_transforms.PointcloudTranslate(),
    ]
)


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = os.path.join(root, f'{args.dataset_name}_c')

        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'partnet':
        if args.corruption != 'clean':
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}_{args.severity}')
        else:
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}')

        inference_dataset = tta_datasets.PartNormalDataset(root=root, npoints=config.npoints, split='test',
                                                           normal_channel=config.normal, debug=args.debug)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)
    elif args.dataset_name == 'scanobject':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    else:
        raise NotImplementedError(f'TTA for {args.tta} is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader


def load_base_model(args, config, logger, load_part_seg=False):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts, load_part_seg)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
            args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def eval_source(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    method = config.model.transformer_config.method

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name in ['scanobject', 'scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    if (method == "MATE"):   
                        logits = base_model.module.classification_only(points, only_unmasked=False)
                    elif (method == "SMART_PC_P" or method == "SMART_PC_N"):  
                        logits = base_model.module.classification_SMART(points, only_unmasked=False)      

                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())    

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:', os.path.join('source_only_results/', f'{logtime}_results.txt'))


def eval_source_rotnet(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results_rotnet/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'scanobject':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:',
                          os.path.join('source_only_results_rotnet/', f'{logtime}_results.txt'))


def tta_rotnet(args, config, train_writer=None):
    dataset_name = args.dataset_name

    assert dataset_name is not None
    assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.batch_size_tta = 48
    args.batch_size = 1
    args.disable_bn_adaptation = True

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_rotnet_results/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []

            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'scanobject':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # making a batch
                    points = [points for _ in range(args.batch_size_tta)]
                    points = torch.squeeze(torch.vstack(points))
                    pts_rot, label_rot = rotate_batch(points)
                    pts_rot, label_rot = pts_rot.cuda(), label_rot.cuda()
                    loss = base_model(0, pts_rot, 0, label_rot, tta=True)  # get out only rotnet loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Rot Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)
                logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    # intermediate results
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('tta_rotnet_results/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def tta_tent(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)
    args.severity = 5
    f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_tent_results/')
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            logits = tent_shot_utils.forward_and_adapt_tent(points, adapted_model, optimizer)
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()


def tta_t3a(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    ext, cls = t3a_utils.get_cls_ext(base_model)

    adapted_model = t3a_utils.T3A(args, ext, cls, config)

    args.severity = 5
    f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_t3a_results/')
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            logits = adapted_model(points)
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()


def tta_shot(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    base_model = load_base_model(args, config, logger)
    adapted_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)
    f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_shot_results/')
    f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
    f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
    f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
    args.severity = 5
    for corr_id, args.corruption in enumerate(corruptions):
        tta_loader = load_tta_dataset(args, config)
        test_pred = []
        test_label = []
        for idx, (data, labels) in enumerate(tta_loader):
            adapted_model.zero_grad()
            points = data.cuda()
            labels = labels.cuda()
            # points = [points for _ in range(args.batch_size_tta)]
            points = misc.fps(points, npoints)
            logits = tent_shot_utils.forward_and_adapt_shot(points, adapted_model, optimizer)
            target = labels.view(-1)
            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())
            if idx % 50 == 0:
                # intermediate results
                test_pred_ = torch.cat(test_pred, dim=0)
                test_label_ = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                    test_label_ = dist_utils.gather_tensor(test_label_, args)

                acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                          logger=logger)
        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                  logger=logger)
        f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
        f_write.flush()
    f_write.close()
    if train_writer is not None:
        train_writer.close()

def batch_statistical_outlier_removal(point_clouds, lengths, K=20, std_ratio=2.0):
    """
    Remove statistical outliers from the input point clouds in batch mode.
    Args:
        point_clouds (torch.Tensor): The input point clouds (shape: B x N x 3).
        lengths (torch.Tensor): The lengths of the point clouds (shape: B).
        K (int, optional): The number of neighbors to consider. Defaults to 20.
        std_ratio (float, optional): The standard deviation ratio. Defaults to 2.0.
    Returns:
        torch.Tensor: The filtered point clouds (padded to max length in batch).
        torch.Tensor: The updated lengths of the filtered point clouds.
    """
    # Compute the mean and standard deviation of the distances to the K nearest neighbors
    dists, _, _ = knn_points(
        point_clouds, point_clouds, K=K
    )
    dists = dists[:, :, 1:].mean(-1)  # Exclude the self-distance (0)
    mean_dists = dists.mean(dim= -1)  # Mean distance for each point
    std_dists = dists.std(dim= -1)  # Standard deviation for each point
    # Filter out points based on distance threshold
    mask = (
        dists > (mean_dists[:, None] + (std_ratio * std_dists[:, None]))
    ) # Mask indicating points within the threshold

    # s_point = point_clouds[~mask][:, -1:]
    # noise = ((0.001 - 0.0005) * torch.rand(mask.shape[0], mask.shape[1], 3) + 0.0005).cuda()       
    repeated_points = point_clouds[:, -1:].repeat(1, 1024, 1)

    point_clouds[mask] = repeated_points[mask]     
    # point_clouds = point_clouds * mask[..., None]    
    # point_clouds[mask] = 0

    return point_clouds


def random_scale_one_axis(point_cloud, scale_min=0.8, scale_max=1.2):
    """
    Randomly scale exactly one axis (x, y, or z) in each point cloud in the batch.
    
    Args:
        point_cloud (torch.Tensor): Input tensor of shape (B, N, 3).
        scale_min (float): Minimum scale factor (inclusive).
        scale_max (float): Maximum scale factor (inclusive).
        
    Returns:
        torch.Tensor: Scaled point cloud of shape (B, N, 3).
    """
    # point_cloud shape: (B, N, 3)
    B, N, C = point_cloud.shape
    assert C == 3, "Last dimension must be 3 for (x, y, z)."
    
    # Choose a random axis [0, 1, or 2] for each item in the batch
    axes = torch.randint(low=0, high=3, size=(B,))
    
    # Generate random scale factors for each item in the batch
    scales = torch.empty(B).uniform_(scale_min, scale_max)
    
    # Clone the input so we don't modify it in-place
    scaled_pc = point_cloud.clone()
    
    # Scale the selected axis for each item in the batch
    for i in range(B):
        axis = axes[i]
        scale = scales[i]
        scaled_pc[i, :, axis] *= scale
    
    return scaled_pc

def plot_bn_gaussians(running_mean, running_var, save_path='bn_gaussians.png', num_curves=5):
    """
    Plots and saves Gaussian curves based on BatchNorm running statistics.

    Parameters:
    - running_mean: numpy array of shape (C,)
    - running_var: numpy array of shape (C,)
    - save_path: file path to save the plot
    - num_curves: number of channels to plot (default: 5)
    """
    assert running_mean.shape == running_var.shape, "Mean and variance must have same shape"

    C = running_mean.shape[0]
    num_curves = min(num_curves, C)  # avoid indexing errors

    plt.figure(figsize=(8, 4))

    # x = np.linspace(-0.3, 0.3, 200)                 ### for first later
    # x = np.linspace(-9.0, 9.0, 200)                 ### for second later    
    x = np.linspace(-7.0, 7.0, 200)                 ### for second later       

    for i in range(num_curves):
        mu = running_mean[i]
        sigma = np.sqrt(running_var[i])
        y = norm.pdf(x, mu, sigma)
        plt.plot(x, y, label=f'Channel {i} (μ={mu:.2f}, σ={sigma:.2f})')

    plt.title('Gaussian Curves of BatchNorm Channels')
    plt.xlabel('Activation value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'Plot saved to {save_path}')

def tta(args, config, train_writer=None):
    dataset_name = args.dataset_name
    npoints = config.npoints
    logger = get_logger(args.log_name)

    method = config.model.transformer_config.method
    only_bn_update = config.model.transformer_config.only_bn_update
    all_params_update = config.model.transformer_config.all_params_update
    only_bn_ln_update = config.model.transformer_config.only_bn_ln_update
    iteration = config.model.transformer_config.iteration
    repeat_data_MATE = config.model.transformer_config.repeat_data_MATE
    repeat_data_SMART_PC = config.model.transformer_config.repeat_data_SMART_PC
    repeat_data_with_rotation_SMART_PC = config.model.transformer_config.repeat_data_with_rotation_SMART_PC
    time_cal = config.model.transformer_config.time_cal
    alg_update = config.model.transformer_config.alg_update
    scale_aug = config.model.transformer_config.scale_aug
    repeat_data_with_jitter_SMART_PC = config.model.transformer_config.repeat_data_with_jitter_SMART_PC
    repeat_data_with_h_flip_SMART_PC = config.model.transformer_config.repeat_data_with_h_flip_SMART_PC
    repeat_data_with_scale_trans_SMART_PC = config.model.transformer_config.repeat_data_with_scale_trans_SMART_PC
    repeat_data_with_trans_SMART_PC = config.model.transformer_config.repeat_data_with_trans_SMART_PC
    visualization = config.model.transformer_config.visualization
    visualization_step = config.model.transformer_config.visualization_step
    visualization_saved_addr = config.model.transformer_config.visualization_saved_addr

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
                f_write.write(f'Corruption LEVEL: {args.severity}' + '\n\n')

            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []
            if args.online:
                without_backpropagation = False   
                base_model = load_base_model(args, config, logger)
                if (alg_update != "tent"):
                    if (all_params_update):
                        optimizer = builder.build_opti_sche(base_model, config)[0] 
                    elif (only_bn_update):    
                        optimizer = builder.build_opti_sche_bn(base_model, config)[0]
                    elif (only_bn_ln_update):    
                        optimizer = builder.build_opti_sche_bn_ln(base_model, config)[0]     
                    else:
                        without_backpropagation = True    

                else:
                    base_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)      

                # args.grad_steps = 1
                args.grad_steps = iteration 

            base_model_overall = load_base_model(args, config, logger)  

            time_list = [] 

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    without_backpropagation = False
                    # base_model = load_base_model(args, config, logger)     
                    base_model = copy.deepcopy(base_model_overall)
                    if (alg_update != "tent"):
                        if (all_params_update):
                            optimizer = builder.build_opti_sche(base_model, config)[0]      
                        elif (only_bn_update):    
                            optimizer = builder.build_opti_sche_bn(base_model, config)[0]
                        elif (only_bn_ln_update):    
                            optimizer = builder.build_opti_sche_bn_ln(base_model, config)[0]   
                        else:
                            without_backpropagation = True      

                    else:
                        base_model, optimizer = tent_shot_utils.setup_tent_shot(args, model=base_model)                      

                base_model.zero_grad()     
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:  
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    if (config.model.transformer_config.outlayer_removal):  
                        points = batch_statistical_outlier_removal(points, lengths= 1024, K=5, std_ratio=1.5)    

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:

                        if (repeat_data_MATE or repeat_data_SMART_PC or repeat_data_with_rotation_SMART_PC):
                            points = [points for _ in range(args.batch_size_tta)]        
                            points = torch.squeeze(torch.vstack(points))   

                            if (repeat_data_with_rotation_SMART_PC):
                                points = train_transforms_random(points)    

                            if (repeat_data_with_jitter_SMART_PC): 
                                points = train_transforms_random_jitter(points)    

                            if (repeat_data_with_h_flip_SMART_PC): 
                                points = train_transforms_random_h_flip(points)      

                            if (repeat_data_with_scale_trans_SMART_PC): 
                                points = train_transforms_random_scale_trans(points)       

                            if (repeat_data_with_trans_SMART_PC): 
                                points = train_transforms_random_trans(points)       
                                  

                            if (scale_aug):
                                points = random_scale_one_axis(points, 0.5, 2.5)            

                        if (time_cal):
                            time_list = []
                            for i_time in range (20): 
                                start_time = time.time()
                                loss_recon, loss_p_consistency, loss_regularize = base_model(points)    
                                loss = loss_recon + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
                                # loss = loss.mean()
                                # loss.backward()
                                # optimizer.step()
                                # base_model.zero_grad()
                                # optimizer.zero_grad()  
                                end_time = time.time()

                                time_list.append(end_time - start_time)

                            print("total time: ", np.mean(time_list))    
                        else:
                            start_time = time.time()   
                            loss_recon, loss_p_consistency, loss_regularize = base_model(points)   
                            end_time = time.time()   
                            time_list.append(end_time - start_time)   
                            loss = loss_recon + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
                            # loss = torch.min(loss_recon, torch.tensor(0.07).cuda()) + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
                            loss = loss.mean()
                            if (without_backpropagation == False):       
                                loss.backward()   
                                optimizer.step()
                                base_model.zero_grad()            
                                optimizer.zero_grad()  

                            # end_time = time.time()   
                            # time_list.append(end_time - start_time)               


                    else:
                        continue

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                if (visualization == True):
                    if ((idx % visualization_step) == 0):
                        base_model.eval()
                        # running_mean = base_model.module.MAE_encoder.encoder.first_conv[1].running_mean
                        # running_var = base_model.module.MAE_encoder.encoder.first_conv[1].running_var

                        # running_mean = base_model.module.MAE_encoder.encoder.second_conv[1].running_mean
                        # running_var = base_model.module.MAE_encoder.encoder.second_conv[1].running_var

                        # running_mean = base_model.module.class_head[1].running_mean
                        # running_var = base_model.module.class_head[1].running_var   

                        if not os.path.exists(visualization_saved_addr + args.corruption):
                            os.makedirs(visualization_saved_addr + args.corruption)
                        plot_bn_gaussians(running_mean.cpu().detach().numpy(), running_var.cpu().detach().numpy(), save_path= visualization_saved_addr + args.corruption + f'/bn_layer2__batch_{idx}.png')
                        base_model.train()

                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)   

                if (method == "MATE"):   
                    logits = base_model.module.classification_only(points, only_unmasked=False)

                elif (method == "SMART_PC_N"):    
                    logits = base_model.module.classification_SMART(points, only_unmasked=False)      

                elif (method == "SMART_PC_P"):    
                    logits = base_model.module.classification_SMART_2(points, only_unmasked=False)          

                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:    
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',   
                              logger=logger)

                    acc_avg.append(acc.cpu())

            ###############vis   
            # base_model.eval()     
            # logits = base_model.module.classification_SMART(points, only_unmasked=False)   
            #################
            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            print("#############################################################################")   
            print("FPS: ", np.mean(time_list[10:]))
            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',  
                      logger=logger)
            # torch.save({'base_model': base_model.module.state_dict() if args.distributed else base_model.state_dict(),
            #                     'optimizer': optimizer.state_dict()}, "./best_model_background.pth")
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def tta_dua(args, config, train_writer=None):
    dataset_name = args.dataset_name
    # assert args.tta
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.disable_bn_adaptation = False

    args.batch_size_tta = 48
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            test_pred = []
            test_label = []
            base_model = load_base_model(args, config, logger)

            for idx, (data, labels) in enumerate(tta_loader):
                base_model.train()

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject', 'scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        _ = base_model.module.classification_only(points,
                                                                  only_unmasked=True)  # only a forward pass through the encoder with BN in train mode
                        # loss=0
                    else:
                        continue

                    # print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},',
                    #           logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model.module.classification_only(points, only_unmasked=False)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 100 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def tta_partseg(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_partnet):
            root = config.root_partseg

            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if args.corruption != 'clean':
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}_{args.severity}')
            else:
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_part_seg/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_partnet}' + '\n\n')

            TEST_DATASET = tta_datasets.PartNormalDatasetSeg(root=root, npoints=config.npoint, split='test',
                                                             normal_channel=config.normal, debug=args.debug)
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_partnet) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()


def tta_shapenet(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_h5):
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_shape_net/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_h5}' + '\n\n')

            TEST_DATASET = tta_datasets.ShapeNetC(args,
                                                  root='./data/shapenet_c')
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_h5) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()
