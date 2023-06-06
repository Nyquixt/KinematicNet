import os
import shutil
import argparse
from datetime import datetime
from collections import defaultdict
import pickle

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.kinematic_net_vol_direct import KinematicNetDirect

from utils import cfg
from dataset import roofing
from dataset import roofing_utils_direct as dataset_utils
from utils.geometry import rotation_matrix_to_quaternion, mean_angle_error_pavllo, quat2mat, compute_rotation_matrix_from_euler, compute_rotation_matrix_from_ortho6d
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="./logs", help="Path, where logs will be stored")
    parser.add_argument("--verbose", action='store_true', help="0=no output, 1=verbose")

    args = parser.parse_args()
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = roofing.RoofingMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            labels_path=config.dataset.train.labels_path,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True,
            drop_last=True
        )

    # val
    val_dataset = roofing.RoofingMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        labels_path=config.dataset.train.labels_path,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler

def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    return experiment_dir


def one_epoch(model, config, criterion, opt, dataloader, device, epoch, is_train=True, experiment_dir=None, verbose=False):
    if is_train:
        model.train()
    else:
        model.eval()

    results = defaultdict(list)
    running_loss = 0.0
    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        if verbose:
            with tqdm(dataloader, unit='batch') as tepoch:
                for batch in tepoch:
                    if is_train:
                        tepoch.set_description(f"Epoch {epoch + 1}/ Train/")
                    else:
                        tepoch.set_description(f"Epoch {epoch + 1}/ Val/")
                    
                    if batch is None:
                        print("Found None batch")
                        continue

                    images_batch, proj_matricies_batch, rotations_batch = dataset_utils.prepare_batch(batch, device, config) # process batch
                    rotations_pred = model(images_batch, proj_matricies_batch, batch) # forward pass

                    # calculate loss
                    loss = criterion(rotations_pred, rotations_batch)
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
                    if is_train: # calculate gradients and update network's weights
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                    # save answers for evaluation
                    if not is_train:
                        # rot mat
                        b, j, _ = rotations_pred.size()
                        if rotations_pred.shape[2] == 4: # quaternion
                            rotations_pred = quat2mat(rotations_pred.view(-1, 4))
                            rotations_batch = quat2mat(rotations_batch.view(-1, 4))
                        elif rotations_pred.shape[2] == 3: # euler
                            rotations_pred = compute_rotation_matrix_from_euler(rotations_pred.view(-1, 3))
                            rotations_batch = compute_rotation_matrix_from_euler(rotations_batch.view(-1, 3))
                        elif rotations_pred.shape[2] == 6: # 6d
                            rotations_pred = compute_rotation_matrix_from_ortho6d(rotations_pred.view(-1, 6))
                            rotations_batch = compute_rotation_matrix_from_ortho6d(rotations_batch.view(-1, 6))

                        rotations_pred = rotations_pred.view(b, j, 3, 3)
                        rotations_batch = rotations_batch.view(b, j, 3, 3)

                        results['rotations_pred'].append(rotations_pred.detach().cpu().numpy()) # B, J , 3, 3
                        results['rotations_gt'].append(rotations_batch.detach().cpu().numpy())
        else:
            for batch in dataloader:
                if batch is None:
                    print("Found None batch")
                    continue

                images_batch, proj_matricies_batch, rotations_batch = dataset_utils.prepare_batch(batch, device, config) # process batch
                rotations_pred = model(images_batch, proj_matricies_batch, batch) # forward pass

                # calculate loss
                loss = criterion(rotations_pred, rotations_batch)
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                if is_train: # calculate gradients and update network's weights
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # save answers for evaluation
                if not is_train:
                    # rot mat
                    b, j, _ = rotations_pred.size()
                    if rotations_pred.shape[2] == 4: # quaternion
                        rotations_pred = quat2mat(rotations_pred.view(-1, 4))
                        rotations_batch = quat2mat(rotations_batch.view(-1, 4))
                    elif rotations_pred.shape[2] == 3: # euler
                        rotations_pred = compute_rotation_matrix_from_euler(rotations_pred.view(-1, 3))
                        rotations_batch = compute_rotation_matrix_from_euler(rotations_batch.view(-1, 3))
                    elif rotations_pred.shape[2] == 6: # 6d
                        rotations_pred = compute_rotation_matrix_from_ortho6d(rotations_pred.view(-1, 6))
                        rotations_batch = compute_rotation_matrix_from_ortho6d(rotations_batch.view(-1, 6))

                    rotations_pred = rotations_pred.view(b, j, 3, 3)
                    rotations_batch = rotations_batch.view(b, j, 3, 3)

                    results['rotations_pred'].append(rotations_pred.detach().cpu().numpy()) # B, J , 3, 3
                    results['rotations_gt'].append(rotations_batch.detach().cpu().numpy())

    if not is_train: # save for plotting
        results['rotations_pred'] = np.concatenate(results['rotations_pred'], axis=0)
        results['rotations_gt'] = np.concatenate(results['rotations_gt'], axis=0)

        checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
        os.makedirs(checkpoint_dir, exist_ok=True)

        # dump results
        with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
            pickle.dump(results, fout)

        # calculate euler angle error for all angles and then average them
        preds = torch.from_numpy(results['rotations_pred'])
        gt = torch.from_numpy(results['rotations_gt'])
        n_joints = preds.shape[1]

        preds = preds.view(-1, 3, 3)
        gt = gt.view(-1, 3, 3)

        # homogeneous transformation before converting to quaternion

        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                                device=preds.device).reshape(1, 3, 1).expand(preds.shape[0], -1, -1)
        quat_preds = torch.cat([preds, hom], dim=-1)
        quat_gt = torch.cat([gt, hom], dim=-1)

        quat_preds = rotation_matrix_to_quaternion(quat_preds).view(-1, n_joints, 4)
        quat_gt = rotation_matrix_to_quaternion(quat_gt).view(-1, n_joints, 4)
    
        avg_joint_error = mean_angle_error_pavllo(quat_preds, quat_gt, n_joints=n_joints)
        return (running_loss / len(dataloader)), avg_joint_error # val
    
    return (running_loss / len(dataloader)) # train

def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    device = torch.device('cuda')

    # config
    config = cfg.load_config(args.config)
    model = KinematicNetDirect(config).to(device) # define model

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights for the whole model")

    # criterion
    if config.opt.criterion == "l2":
        criterion = nn.MSELoss()
    else:
        raise Exception("Loss not implemented")

    # optimizer
    opt = None
    if not args.eval:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    lr = config.opt.lr

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, _ = setup_dataloaders(config)

    # experiment
    experiment_dir = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # log file
    with open(os.path.join(experiment_dir, 'log.txt'), 'w') as logfile:
        logfile.write("Logging\n\n")

    # multi-gpu
    # model = nn.DataParallel(model)
    best_perf = float("inf")
    best_epoch = 0
    torch.autograd.set_detect_anomaly(True)
    if not args.eval:
        # train loop
        for epoch in range(config.opt.n_epochs):
            lr = adjust_learning_rate(opt, epoch, lr, config.opt.schedule, config.opt.gamma)

            train_avg_loss = one_epoch(model, config, criterion, opt, train_dataloader, device, epoch, is_train=True, experiment_dir=experiment_dir, verbose=args.verbose)
            val_avg_loss, err = one_epoch(model, config, criterion, opt, val_dataloader, device, epoch, is_train=False, experiment_dir=experiment_dir, verbose=args.verbose)
            
            # log file
            with open(os.path.join(experiment_dir, 'log.txt'), 'a+') as logfile:
                logfile.write(f'Epoch {epoch + 1} - Train Loss: {train_avg_loss} - Val Loss: {val_avg_loss} - Error: {err}\n')

            print(f'Epoch {epoch + 1} - Train Loss: {train_avg_loss} - Val Loss: {val_avg_loss} - Error: {err}')
            
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            if val_avg_loss < best_perf:
                best_epoch = epoch + 1
                best_perf = val_avg_loss
                print("=> Saving best torch model")
                torch.save(model.state_dict(), os.path.join(experiment_dir, "best.pth"))

        # log file
        with open(os.path.join(experiment_dir, 'log.txt'), 'a+') as logfile:
            logfile.write(f'Best epoch: {best_epoch}')

    else:
        if args.eval_dataset == 'train':
            one_epoch(model, config, criterion, opt, train_dataloader, device, 0, is_train=False, experiment_dir=experiment_dir, verbose=args.verbose)
        else:
            one_epoch(model, config, criterion, opt, val_dataloader, device, 0, is_train=False, experiment_dir=experiment_dir, verbose=args.verbose)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
