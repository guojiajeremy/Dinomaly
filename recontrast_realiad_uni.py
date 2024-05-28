# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.resnet import wide_resnet50_2
from models.de_resnet import de_wide_resnet50_2
from models.uad import ReContrast
from dataset import MVTecDataset, RealIADDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, global_cosine_hm, global_cosine_hm_percent, \
    WarmCosineScheduler, replace_layers
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list):
    setup_seed(1)

    total_iters = 30000
    batch_size = 16
    image_size = 256
    crop_size = 256

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for i, item in enumerate(item_list):
        root_path = '/data/disk8T2/guoj/Real-IAD'
        # root_path = args.data_path

        train_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                    phase='train')
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        # train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = RealIADDataset(root=root_path, category=item, transform=data_transform, gt_transform=gt_transform,
                                   phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)

    encoder, bn = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)

    replace_layers(decoder, nn.ReLU, nn.GELU())

    encoder = encoder.to(device)
    bn = bn.to(device)
    decoder = decoder.to(device)
    encoder_freeze = copy.deepcopy(encoder)

    model = ReContrast(encoder=encoder, encoder_freeze=encoder_freeze, bottleneck=bn, decoder=decoder)

    optimizer = StableAdamW([{'params': decoder.parameters()}, {'params': bn.parameters()},
                             {'params': encoder.parameters(), 'lr': 1e-5}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_iters * 0.8)], gamma=0.2)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train(encoder_bn_train=False)

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)
            # loss = global_cosine(en, de)

            alpha_final = 1
            alpha = min(-3 + (alpha_final - -3) * it / 1000, alpha_final)
            loss = global_cosine_hm(en[:3], de[:3], alpha=alpha, factor=0.) / 2 + \
                   global_cosine_hm(en[3:], de[3:], alpha=alpha, factor=0.) / 2

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % total_iters == 0:
                # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train(encoder_bn_train=False)

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []
        # print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='recontrast_realiad_uni_max1_it30k_sams2e31e5_b16_s1')
    args = parser.parse_args()

    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
