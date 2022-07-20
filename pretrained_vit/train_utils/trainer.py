import os
import sys

import wandb
import numpy as np
import torch
from torchvision.utils import save_image

from .misc_utils import AverageMeter, accuracy, count_params_single
from .dist_utils import reduce_tensor, distribute_bn
from .mix import cutmix_data, mixup_data, mixup_criterion


class Trainer():
    def __init__(self, args, model, criterion, optimizer, lr_scheduler,
                 train_loader, val_loader, test_loader):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.saved = False

    def train(self):
        self.best_acc = 0
        self.best_epoch = 0
        self.max_memory = 0
        self.no_params = 0
        self.lr_scheduler.step(0)

        for epoch in range(self.args.epochs):
            self.epoch = epoch

            if self.args.distributed or self.args.ra > 1:
                self.train_loader.sampler.set_epoch(self.epoch)

            train_acc, train_loss = self.train_epoch()

            if self.args.local_rank == 0:
                val_acc, val_loss = self.validate_epoch()
                self.epoch_end_routine(train_acc, train_loss, val_acc, val_loss)

        if self.args.local_rank == 0:
            self.train_end_routine(val_acc)

        return self.best_acc, self.best_epoch, self.max_memory, self.no_params

    def prepare_batch(self, batch):
        images, targets = batch
        if type(images) == list:
            images = torch.stack(images, dim=1)
        if self.args.distributed:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            images = images.to(self.args.device, non_blocking=True)
            targets = targets.to(self.args.device, non_blocking=True)
        return images, targets

    def predict(self, images, targets, train=True):
        if self.args.cm or self.args.mu:
            r = np.random.rand(1)
            if r < self.args.mix_prob and train:
                images, y_a, y_b, lam = self.prepare_mix(images, targets)

        if self.args.save_images > 0 and (self.curr_iter % self.args.save_images) == 0:
            samples = (images.reshape(
                -1, 3, self.args.image_size, self.args.image_size).data + 1) / 2.0
            save_image(samples,
                       os.path.join(self.args.results_dir, f'{self.curr_iter}.png'),
                       nrow=int(np.sqrt(samples.shape[0])))
        elif self.args.save_images > 0 and self.epoch == 0 and not train and not self.saved:
            samples = (images.reshape(
                -1, 3, self.args.image_size, self.args.image_size).data + 1) / 2.0
            save_image(samples,
                       os.path.join(self.args.results_dir, f'test.png'),
                       nrow=int(np.sqrt(samples.shape[0])))
            self.saved = True

        output = self.model(images)

        if (self.args.cm or self.args.mu) and r < self.args.mix_prob and train:
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            loss = self.criterion(output, targets)

        return output, loss

    def prepare_mix(self, images, targets):
        # cutmix and mixup
        if self.args.cm and self.args.mu:
            switching_prob = np.random.rand(1)
            # Cutmix
            if switching_prob < 0.5:
                slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, self.args)
                images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
            # Mixup
            else:
                images, y_a, y_b, lam = mixup_data(images, targets, self.args)
        # cutmix only
        elif self.args.cm:
            slicing_idx, y_a, y_b, lam, sliced = cutmix_data(images, targets, self.args)
            images[:, :, slicing_idx[0]:slicing_idx[2], slicing_idx[1]:slicing_idx[3]] = sliced
        # mixup only
        elif self.args.mu:
            images, y_a, y_b, lam = mixup_data(images, targets, self.args)
        return images, y_a, y_b, lam

    def train_epoch(self):
        """vanilla training"""
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.curr_iter = self.epoch * len(self.train_loader)

        for idx, batch in enumerate(self.train_loader):
            images, targets = self.prepare_batch(batch)
            output, loss = self.predict(images, targets, train=True)

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))

            # ===================backward=====================
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.curr_iter += 1
            self.lr_scheduler.step_update(num_updates=self.curr_iter)

            # ===================meters=====================
            torch.cuda.synchronize()

            if self.args.distributed:
                reduced_loss = reduce_tensor(loss.data, self.args.world_size)
                acc1 = reduce_tensor(acc1, self.args.world_size)
                acc5 = reduce_tensor(acc5, self.args.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # print info
            if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                lr_curr = self.optimizer.param_groups[0]['lr']
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'LR: {4}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        self.epoch, self.args.epochs, idx, len(self.train_loader), lr_curr,
                        loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        if self.args.distributed:
            distribute_bn(self.model, self.args.world_size, True)

        self.lr_scheduler.step(self.epoch + 1)

        return top1.avg, losses.avg

    def validate_epoch(self):
        """validation"""
        # switch to evaluate mode
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                images, targets = self.prepare_batch(batch)
                output, loss = self.predict(images, targets, train=False)

                acc1, acc5 = accuracy(output, targets, topk=(1, 5))

                torch.cuda.synchronize()

                reduced_loss = loss.data

                losses.update(reduced_loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                    print('Val: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              idx, len(self.val_loader),
                              loss=losses, top1=top1, top5=top5))

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg, losses.avg

    def epoch_end_routine(self, train_acc, train_loss, val_acc, val_loss):
        lr_curr = self.optimizer.param_groups[0]['lr']
        print("Training...Epoch: {} | LR: {}".format(self.epoch, lr_curr))
        log_dic = {'epoch': self.epoch, 'lr': lr_curr,
                   'train_acc': train_acc, 'train_loss': train_loss,
                   'val_acc': val_acc, 'val_loss': val_loss}
        wandb.log(log_dic)

        # save the best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = self.epoch
            self.save_model(self.best_epoch, val_acc, mode='best')
        # regular saving
        if (self.epoch + 1) % self.args.save_freq == 0:
            self.save_model(self.epoch, val_acc, mode='epoch')
        # VRAM memory consumption
        curr_max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        if curr_max_memory > self.max_memory:
            self.max_memory = curr_max_memory

    def train_end_routine(self, val_acc):
        # save last
        self.save_model(self.epoch, val_acc, mode='last')
        # summary stats
        self.no_params = count_params_single(self.model)

    def save_model(self, epoch, acc, mode):
        state = {
            'config': self.args,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'accuracy': acc,
            'optimizer': self.optimizer.state_dict(),
        }

        if mode == 'best':
            save_file = os.path.join(self.args.results_dir, f'{self.args.model_name}_best.pth')
            print('Saving the best model!')
            torch.save(state, save_file)
        elif mode == 'epoch':
            save_file = os.path.join(self.args.results_dir, f'ckpt_epoch_{epoch}.pth')
            print('==> Saving each {} epochs...'.format(self.args.save_freq))
            torch.save(state, save_file)
        elif mode == 'last':
            save_file = os.path.join(self.args.results_dir, f'{self.args.model_name}_last.pth')
            print('Saving last epoch')
            torch.save(state, save_file)
