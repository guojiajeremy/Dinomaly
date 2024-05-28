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

from models.uad import ViTill, ViTillv2
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, Mlp, bMlp, EfficientBlock
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, replace_layers, global_cosine_hm_percent, WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info

import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


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


def train(item):
    setup_seed(1)
    print_fn(item)
    total_iters = 5000
    batch_size = 16
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_path = '../MPDD/' + item + '/train'
    test_path = '../MPDD/' + item

    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in small, base, large."
    bottleneck = []
    decoder = []

    bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2))
    bottleneck = nn.ModuleList(bottleneck)

    for i in range(8):
        # blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
        #                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.3)
        blk = EfficientBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   fuse_dropout=0.)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.08, b=0.08)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = torch.optim.AdamW([{'params': decoder.parameters()}, {'params': bottleneck.parameters()}],
                                  lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-5, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=3e-3, final_value=3e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    auroc_px_best, auroc_sp_best, ap_px_best, ap_sp_best = 0, 0, 0, 0

    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)
            # loss = global_cosine(en, de)

            p_final = 0.9
            p = min(p_final * it / (total_iters * 0.1), p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 1000 == 0:
                auroc_px, auroc_sp, ap_px, ap_sp, gt_pr = evaluation_batch(model, test_dataloader, device,
                                                                           max_ratio=0.01)
                if auroc_sp >= auroc_sp_best:
                    auroc_px_best, auroc_sp_best, ap_px_best, ap_sp_best = auroc_px, auroc_sp, ap_px, ap_sp
                print_fn('{}: P-Auroc:{:.4f}, I-Auroc:{:.4f},P-AP:{:.4f},I-AP:{:.4f}'.format(item,
                                                                                             auroc_px,
                                                                                             auroc_sp,
                                                                                             ap_px, ap_sp))

                model.train()

            it += 1
            if it == total_iters:
                break
        print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))

    # torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

    return auroc_px, auroc_sp, ap_px, ap_sp, auroc_px_best, auroc_sp_best, ap_px_best, ap_sp_best


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_mpdd_sep_dinov2br_c392_en29_v2bn4dp2_de8_eattn_md_it10k_lr3e3_wd1e5_w1hcosa3e4_hmp09f01w01_b16_s1')
    args = parser.parse_args()

    item_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    result_list = []
    result_list_best = []
    for i, item in enumerate(item_list):
        auroc_px, auroc_sp, ap_px, ap_sp, auroc_px_best, auroc_sp_best, ap_px_best, ap_sp_best = train(item)
        result_list.append([item, auroc_px, auroc_sp, ap_px, ap_sp])
        result_list_best.append([item, auroc_px_best, auroc_sp_best, ap_px_best, ap_sp_best])

    mean_auroc_px = np.mean([result[1] for result in result_list])
    mean_auroc_sp = np.mean([result[2] for result in result_list])
    mean_ap_px = np.mean([result[3] for result in result_list])
    mean_ap_sp = np.mean([result[4] for result in result_list])

    print_fn(result_list)
    print_fn('mean P-Auroc:{:.4f}, I-AUROC:{:.4f}, P-AP:{:.4}, I-AP:{:.4}'.format(mean_auroc_px, mean_auroc_sp,
                                                                                  mean_ap_px, mean_ap_sp))

    best_auroc_px = np.mean([result[1] for result in result_list_best])
    best_auroc_sp = np.mean([result[2] for result in result_list_best])
    best_ap_px = np.mean([result[3] for result in result_list_best])
    best_ap_sp = np.mean([result[4] for result in result_list_best])
    print_fn(result_list_best)
    print_fn('best P-Auroc:{:.4f}, I-AUROC:{:.4f}, P-AP:{:.4}, I-AP:{:.4}'.format(best_auroc_px, best_auroc_sp,
                                                                                  best_ap_px, best_ap_sp))
