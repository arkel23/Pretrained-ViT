import os
import time
import random

import wandb
from torchsummary import summary
from timm.optim import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from pretrained_vit.data_utils.build_dataloaders import build_dataloaders
from pretrained_vit.model_utils.build_model import build_model
from pretrained_vit.other_utils.build_args import parse_train_args
from pretrained_vit.train_utils.misc_utils import summary_stats, set_random_seed
from pretrained_vit.train_utils.scheduler import build_scheduler
from pretrained_vit.train_utils.trainer import Trainer


def adjust_args_general(args):
    args.run_name = '{}_{}_{}'.format(
        args.dataset_name, args.model_name, args.serial
    )
    args.results_dir = os.path.join(args.results_dir, args.run_name)


def build_environment(args):
    if args.serial is None:
        args.serial = random.randint(0, 1000000)
    # Set device and random seed
    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model and criterion
    model = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    if args.ls:
        criterion = LabelSmoothingCrossEntropy(args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # loss and optimizer
    optimizer = create_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, train_loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    adjust_args_general(args)
    os.makedirs(args.results_dir, exist_ok=True)

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader


def main():
    time_start = time.time()

    args = parse_train_args(assert_args=True)

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = build_environment(args)

    trainer = Trainer(args, model, criterion, optimizer, lr_scheduler,
                      train_loader, val_loader, test_loader)

    if args.local_rank == 0:
        wandb.init(config=args, project=args.project_name)
        wandb.run.name = args.run_name
        if not args.distributed and hasattr(model, 'cfg'):
            summary(model, input_size=next(iter(train_loader))[0].shape[1:])
            print(model.cfg)
        else:
            print(model)
        print(args)

    best_acc, best_epoch, max_memory, no_params = trainer.train()

    # summary stats
    if args.local_rank == 0:
        time_total = time.time() - time_start
        summary_stats(args.epochs, time_total, best_acc, best_epoch, max_memory, no_params)
        wandb.finish()


if __name__ == '__main__':
    main()
