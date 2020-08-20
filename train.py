import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import rnn, clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

import os
import sys
import pdb
import json
import argparse
from argparse import Namespace
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from dataset import MyDataset
from models import MyModel
from losses import my_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # system setting
    parser.add_argument('--exp_dir', default=os.getcwd(), type=str)
    parser.add_argument('--exp_name', default='MyExperiment', type=str)
    parser.add_argument('--data_dir', default='/path/to/data/', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--add_graph', action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    # training specifics
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--clip_grad_norm_val', default=0.0, type=float)
    parser.add_argument('--grad_accumulate_batches', default=1, type=int)
    parser.add_argument('--log_grad_norm', action='store_true')
    parser.add_argument('--resume_dir', default='', type=str)
    parser.add_argument('--use_swa', action='store_true')

    # model hyperparameters
    # TODO: write some model hparams here

    args = parser.parse_args()
    
    # add hyperparameters
    ckpt_path = os.path.join(args.exp_dir, args.exp_name, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
        os.makedirs(ckpt_path.replace('ckpt', 'logs'))
        with open(os.path.join(ckpt_path, 'hparams.json'), 'w') as f:
            json.dump(vars(args), f)
    else:
        print(f'Experiment {args.exp_name} already exists.')
        sys.exit()
    writer = SummaryWriter(os.path.join(args.exp_dir, args.exp_name, 'logs'))
    writer.add_hparams(vars(args), dict())

    # seed
    if args.seed:
        fix_seed(args.seed)

    # device
    device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')

    # create loaders
    train_dataloader = DataLoader(
        MyDataset(data_dir=args.data_dir, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_dataloader = DataLoader(
        MyDataset(data_dir=args.data_dir, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    # create model
    if not args.resume_dir:
        # TODO: Initiate model here
        # net = MyModel()
    else:
        try:
            with open(os.path.join(args.resume_dir, 'hparams.json'), 'r') as f:
                hparams = json.load(f)
        except FileNotFoundError:
            print('Cannot find "hparams.json".')
            sys.exit()

        hparams['resume_dir'] = args.resume_dir
        args = Namespace(**hparams)
        # TODO: Initiate model here
        # net = MyModel()
        model_path = os.path.join(args.resume_dir, 'model_best.ckpt')
        print(f'Resume model from {model_path} ...')
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(device)

    # optimization
    # TODO: Choose an optimizer
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=args.learning_rate)
    scheduler = None
    if args.use_swa:
        steps_per_epoch = len(train_dataloader) // args.batch_size
        optimizer = SWA(optimizer, swa_start=20 * steps_per_epoch, swa_freq=steps_per_epoch)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer.optimizer, mode="max", patience=5, factor=0.5)

    if args.resume_dir:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # best_pesq = checkpoint['pesq']
        best_loss = checkpoint['loss']
    else:
        start_epoch = 0
        best_loss = 1e8
        # best_pesq = 0.0
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # add graph to tensorboard
    if args.add_graph:
        # TODO: Create a dummy input for your model
        # dummy = torch.randn(16, 1, args.hop_length * 16).to(device)
        writer.add_graph(net, dummy)

    # iteration start
    criterion = my_loss
    for epoch in range(start_epoch, start_epoch + args.num_epochs, 1):
        # ------------- training ------------- 
        net.train()
        pbar = tqdm(train_dataloader)
        pbar.set_description(f'Epoch {epoch + 1}')
        total_loss = 0.0
        if args.log_grad_norm:
            total_norm = 0.0
        net.zero_grad()
        for i, batch in enumerate(pbar):
            x, y = map(lambda x: x.to(device), batch)
            y_hat = net(x)
            loss = criterion(y_hat, y)
            loss /= args.grad_accumulate_batches
            loss.backward()

            # gradient clipping
            if args.clip_grad_norm_val > 0.0:
                clip_grad_norm_(net.parameters(), args.clip_grad_norm_val)

            # log metrics
            pbar_dict = OrderedDict({
                'loss': loss.item(),
            })
            pbar.set_postfix(pbar_dict)

            total_loss += loss.item()
            if (i + 1) % args.log_interval == 0:
                step = epoch * len(train_dataloader) + i
                writer.add_scalar('Loss/train', total_loss / args.log_interval, step)
                total_loss = 0.0

                # log gradient norm
                if args.log_grad_norm:
                    for p in net.parameters():
                        norm = p.grad.data.norm(2)
                        total_norm += norm.item() ** 2
                    norm = total_norm ** 0.5
                    writer.add_scalar('Gradient 2-Norm/train', norm, step)
                    total_norm = 0.0

            # accumulate gradients
            if (i + 1) % args.grad_accumulate_batches == 0:
                optimizer.step()
                net.zero_grad()

        # ------------- validation -------------
        pbar = tqdm(test_dataloader)
        pbar.set_description('Validation')
        total_loss = 0.0
        # total_pesq = 0.0
        num_test_data = len(test_dataloader)
        with torch.no_grad():
            net.eval()
            for i, batch in enumerate(pbar):
                x, y = map(lambda x: x.to(device), batch)
                y_hat = net(x)
                loss = criterion(y_hat, y)
                # pesq_score = evaluate(e, c, l, fn=cal_pesq)
                pbar_dict = OrderedDict({
                    'val_loss': loss.item(),
                    # 'val_pesq': pesq_score.item(),
                })
                pbar.set_postfix(pbar_dict)

                total_loss += loss.item()
                # total_pesq += pesq_score.item()

            if scheduler is not None:
                scheduler.step(total_pesq / num_test_data)

            writer.add_scalar('Loss/valid', total_loss / num_test_data, epoch)
            # writer.add_scalar('PESQ/valid', total_pesq / num_test_data, epoch)

            # checkpointing
            curr_loss = total_loss / num_test_data
            if  curr_loss < best_loss:
                best_loss = curr_loss
                save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                print(f'Saving checkpoint to {save_path}')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss / num_test_data,
                }, save_path)
            # curr_pesq = total_pesq / num_test_data
            # if  curr_pesq > best_pesq:
                # best_pesq = curr_pesq
                # save_path = os.path.join(ckpt_path, 'model_best.ckpt')
                # print(f'Saving checkpoint to {save_path}')
                # torch.save({
                    # 'epoch': epoch,
                    # 'model_state_dict': net.state_dict(), 
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': total_loss / num_test_data,
                    # 'pesq': total_pesq / num_test_data
                # }, save_path)

    writer.flush()
    writer.close()
