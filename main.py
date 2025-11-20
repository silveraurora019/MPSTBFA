# 文件名: main.py (已修改为 MPS-TBFA)

import logging
import torch
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Any

from utils import set_for_logger
from dataloaders import build_dataloader
# 确保使用您之前修复过 bug 的 loss.py，现在包含 TBFA 损失
from loss import DiceLoss, JointLoss, TBFContrastiveLoss
import torch.nn.functional as F
from nets import build_model

# --- 新增：导入 MPS 聚合器 (已修改) 和 TBFA 工具 ---
from aggregator_mps import MPSAggregator
import tbfa_utils


# --- TBFA 全局设置 ---
# 定义 tail bins (例如 q=[0, 0.1] 和 q=[0.1, 0.25])
# (这些是 tbfa_utils.compute_margins_and_bins 返回的索引)
TBFA_TAIL_BINS_INDICES = [0] # 仅关注最差的 10% (假设 QUANTILES[0]=0.1)
TBFA_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9] # 用于 binning

# 特征统计设置
# (选择 'z' 或 'shadow'。'z' 是 bottleneck avg_pool，维度固定)
TBFA_FEATURE_SOURCE = 'shadow' # 'z' or 'shadow'
# (UNet_pro 'shadow' 特征 (来自 decoder1) 的维度是 32)
TBFA_FEATURE_DIM = 32
TBFA_PROJ_DIM = 128 # 投影维度 d' 
# (注意：如果 32D -> 128D 投影维度过高，也可以将 TBFA_PROJ_DIM 改为 32)


@torch.no_grad()
def get_client_stats_and_logits(
    model, 
    dataloader, 
    device, 
    R: torch.Tensor,
    num_classes: int
):
    """
    (替换 get_client_logits_targets)
    
    为 MPS 和 TBFA 提取所需的所有信息：
    1. Logits, Targets (用于 MPS 模型聚合)
    2. Feature Statistics (mu, cov) (用于 TBFA 重心计算 [cite: 452])
    """
    model.eval()
    
    # --- 1. Logits/Targets (用于 MPS) ---
    all_logits_list = []
    all_targets_list = []
    downsample_size = (64, 64) 

    # --- 2. Stats (用于 TBFA) ---
    all_features_list = []
    all_bins_list = []
    all_targets_spatial_list = [] # 保持空间维度
    
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'

    try:
        max_batches = 10 
        batch_count = 0
        
        for x, target in dataloader:
            if batch_count >= max_batches: break
            
            x = x.to(device)
            target = target.to(device) # [B, 1, H, W] or [B, H, W]
            
            # 1. 模型前向传播
            if is_unet_pro:
                out = model(x)
                logits = out[0] # [B, C, H, W]
                z = out[1] # [B, D_z]
                shadow = out[2] # [B, D_s, H, W]
            else:
                out = model(x)
                logits = out
                z = F.adaptive_avg_pool2d(out, 1).view(out.shape[0], -1) # 模拟 z
                shadow = logits # 模拟 shadow
            
            # --- 存储 Logits/Targets (用于 MPS) ---
            logits_down = F.interpolate(logits, size=downsample_size, mode='bilinear', align_corners=False)
            
            if target.dim() == 3:
                target_spatial = target.unsqueeze(1) # (B, 1, H, W)
            else:
                target_spatial = target
                
            target_down = F.interpolate(target_spatial.float(), size=downsample_size, mode='nearest').long().squeeze(1)
            
            logits_flat = logits_down.permute(0, 2, 3, 1).reshape(-1, logits_down.shape[1])
            target_flat = target_down.reshape(-1)
            
            all_logits_list.append(logits_flat.cpu())
            all_targets_list.append(target_flat.cpu())
            
            # --- 存储特征和空间目标 (用于 TBFA) ---
            if TBFA_FEATURE_SOURCE == 'z':
                features = z # [B, D_z]
                # z 已经是 [B, D]，我们需要将其 "广播" 到空间维度
                B, D_z = features.shape
                features_spatial = features.view(B, D_z, 1, 1).expand(-1, -1, target_spatial.shape[2], target_spatial.shape[3])
            else:
                features_spatial = shadow # [B, D_s, H, W]

            # 计算此 batch 的 margins 和 bins [cite: 451]
            _, bins = tbfa_utils.compute_margins_and_bins(
                logits.detach(), target_spatial, TBFA_QUANTILES
            )
            
            all_features_list.append(features_spatial.cpu())
            all_bins_list.append(bins.cpu())
            all_targets_spatial_list.append(target_spatial.cpu())
            
            batch_count += 1
        
        # --- 拼接 MPS 数据 ---
        if len(all_logits_list) > 0:
            full_logits = torch.cat(all_logits_list, dim=0)
            full_targets = torch.cat(all_targets_list, dim=0)
        else:
            logging.warning("客户端验证集为空 (MPS)。")
            full_logits, full_targets = None, None

        # --- 计算 TBFA 统计量 ---
        if len(all_features_list) > 0:
            full_features = torch.cat(all_features_list, dim=0)
            full_targets_spatial = torch.cat(all_targets_spatial_list, dim=0)
            full_bins = torch.cat(all_bins_list, dim=0)
            
            # 提取 (c,b) 的统计量 [cite: 452]
            raw_stats = tbfa_utils.extract_feature_statistics(
                full_features,
                full_targets_spatial,
                full_bins,
                TBFA_TAIL_BINS_INDICES,
                num_classes
            )
            
            # 压缩统计量 [cite: 453]
            compressed_stats = tbfa_utils.compress_statistics(raw_stats, R)
        else:
            logging.warning("客户端验证集为空 (TBFA)。")
            compressed_stats = {}

        return compressed_stats, full_logits, full_targets

    except Exception as e:
         logging.error(f"提取 Stats/Logits/Targets 时出错: {e}")
         return {}, None, None


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    # parser.add_argument('--data_root', type=str, required=False, default="/data/myn/dataset/Prostate", help="Data directory")
    parser.add_argument('--data_root', type=str, required=False, default="/data/myn/dataset/Fundus", help="Data directory")
    parser.add_argument('--dataset', type=str, default='fundus', 
                        help="Dataset type: 'fundus' or 'prostate'")
    
    parser.add_argument('--model', type=str, default='unet_pro', help='Model type (unet_pro required for TBFA features)')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:1', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--experiment', type=str, default='experiment_mps_tbfa-f-10', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    
    # --- 修改：MPS 参数 ---
    parser.add_argument('--mps_gamma', type=float, default=1, help='Gamma for MPS similarity (exp(-gamma * dist))')
    parser.add_argument('--mps_temp', type=float, default=0.1, help='Temperature for softmax weighting in MPS')
    parser.add_argument('--sim_start_round', type=int, default=0, help='Round to start using MPS/TBFA aggregation')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate for the model')

    # --- 新增：TBFA 参数 ---
    parser.add_argument('--tbfa_lambda_ctr', type=float, default=0.1, help='Weight for TBFA contrastive loss [cite: 437]')
    parser.add_argument('--tbfa_temp', type=float, default=0.5, help='Temperature for TBFA contrastive loss [cite: 433]')

    args = parser.parse_args()
    return args

def communication(server_model, models, client_weights):
    with torch.no_grad():
        device = next(server_model.parameters()).device
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)
            
        # 归一化权重
        client_weights = client_weights / (client_weights.sum() + 1e-8)
        
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
    return server_model


def train_tbfa(
    cid, 
    model, 
    dataloader, 
    device, 
    optimizer, 
    epochs, 
    loss_fun_seg,
    # --- 新增 TBFA 参数 ---
    barycenters_last_round: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    R_matrix: torch.Tensor,
    loss_fun_ctr: TBFContrastiveLoss,
    lambda_ctr: float,
    tail_bins_map: Dict[Tuple[int, int], int]
):
    """
    本地训练，包含 L_seg 和 L_ctr
    """
    model.train()
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    # 准备负重心列表 (只需执行一次)
    neg_barycenters = {}
    if barycenters_last_round:
        for (c_pos, b_pos) in tail_bins_map.keys():
            # [修复 1] 检查键是否存在，防止 KeyError
            if (c_pos, b_pos) not in barycenters_last_round:
                continue
                
            neg_barycenters[(c_pos, b_pos)] = [
                barycenters_last_round[cb_neg]['mean'] 
                for cb_neg in barycenters_last_round 
                if cb_neg != (c_pos, b_pos)
            ]

    for epoch in range(epochs):
        train_acc = 0.
        loss_all_seg = 0.
        loss_all_ctr = 0.
        
        if len(dataloader) == 0:
            continue
            
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device) # [B, 1, H, W]
            
            if is_unet_pro:
                output, z, shadow = model(x)
                if TBFA_FEATURE_SOURCE == 'z':
                    features = z # [B, D_z]
                else:
                    features = shadow # [B, D_s, H, W]
            else:
                output = model(x)
                features = None # 不支持 TBFA
                
            optimizer.zero_grad()
            
            # --- 1. 计算 L_seg ---
            loss_seg = loss_fun_seg(output, target)
            loss_all_seg += loss_seg.item()
            train_acc += DiceLoss().dice_coef(output, target).item()
            
            loss_total = loss_seg
            
            # --- 2. 计算 L_ctr ---
            if barycenters_last_round and features is not None:
                # 2a. 获取当前 batch 的 margins 和 bins
                _, bins = tbfa_utils.compute_margins_and_bins(
                    output.detach(), target, TBFA_QUANTILES
                )
                
                loss_ctr_batch = 0.0
                num_tail_groups = 0

                # 2b. 投影特征
                if TBFA_FEATURE_SOURCE == 'z':
                    # features [B, D] -> phi_proj [B, D_proj]
                    phi_proj = features @ R_matrix.T
                else:
                    # features [B, D, H, W] -> phi_proj [B*H*W, D_proj]
                    phi_proj = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]) @ R_matrix.T

                
                for (c, b_idx) in tail_bins_map.keys():
                    
                    # [修复 2] 关键修改：检查重心是否存在
                    if (c, b_idx) not in barycenters_last_round:
                        continue

                    # 2c. 找到 tail bin (c, b_idx) 中的像素/样本
                    if TBFA_FEATURE_SOURCE == 'z':
                        # (需要将 bins[B,H,W] 聚合到样本级别)
                        # 简化：如果样本的 *平均* bin 属于 tail，则使用该样本的 z
                        sample_bins = torch.mean(bins.float(), dim=(1,2))
                        mask = (sample_bins == b_idx) # (这是一个粗略的近似)
                        # (一个更准确的方法是检查 target==c 像素的 bins, 但这更复杂)
                        # (暂时跳过 'z' 的 L_ctr 实现，因为它不是空间特征)
                        pass 
                    
                    else:
                        # (使用 'shadow' 空间特征)
                        if target.dim() == 4:
                            target_flat = target.squeeze(1).reshape(-1) # [B*H*W]
                        else:
                            target_flat = target.reshape(-1)
                        bins_flat = bins.reshape(-1) # [B*H*W]

                        mask = (target_flat == c) & (bins_flat == b_idx)
                        
                        if mask.sum() > 0:
                            phi_tail_proj = phi_proj[mask] # [N_tail, D_proj]
                            
                            # 现在可以安全访问了，因为前面做了检查
                            mu_pos = barycenters_last_round[(c, b_idx)]['mean']
                            
                            # 安全获取负样本（使用 .get 防止潜在错误，尽管上面 neg_barycenters 已过滤）
                            mu_negs = neg_barycenters.get((c, b_idx), [])
                            
                            # 只有存在负样本时才计算对比损失
                            if mu_negs:
                                loss_ctr_cb = loss_fun_ctr(phi_tail_proj, mu_pos, mu_negs)
                                loss_ctr_batch += loss_ctr_cb
                                num_tail_groups += 1

                if num_tail_groups > 0:
                    loss_ctr = (loss_ctr_batch / num_tail_groups) * lambda_ctr
                    loss_total = loss_total + loss_ctr
                    loss_all_ctr += loss_ctr.item()

            loss_total.backward()
            optimizer.step()
        
        if len(dataloader) > 0:
            avg_loss_seg = loss_all_seg / len(dataloader)
            avg_loss_ctr = loss_all_ctr / len(dataloader)
            train_acc = train_acc / len(dataloader)
            logging.info(f'Client: [{cid}] Epoch: [{epoch}] train_loss_seg: {avg_loss_seg:.4f} train_loss_ctr: {avg_loss_ctr:.4f} train_acc: {train_acc:.4f}')

# --- 修改： 'train' 函数 -> 'train_tbfa' ---
# def train_tbfa(
#     cid, 
#     model, 
#     dataloader, 
#     device, 
#     optimizer, 
#     epochs, 
#     loss_fun_seg,
#     # --- 新增 TBFA 参数 ---
#     barycenters_last_round: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
#     R_matrix: torch.Tensor,
#     loss_fun_ctr: TBFContrastiveLoss,
#     lambda_ctr: float,
#     tail_bins_map: Dict[Tuple[int, int], int]
# ):
#     """
#     本地训练，包含 L_seg 和 L_ctr [cite: 485]
#     """
#     model.train()
#     is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
#     # 准备负重心列表 (只需执行一次)
#     neg_barycenters = {}
#     if barycenters_last_round:
#         for (c_pos, b_pos) in tail_bins_map.keys():
#             neg_barycenters[(c_pos, b_pos)] = [
#                 barycenters_last_round[cb_neg]['mean'] 
#                 for cb_neg in barycenters_last_round 
#                 if cb_neg != (c_pos, b_pos)
#             ]

#     for epoch in range(epochs):
#         train_acc = 0.
#         loss_all_seg = 0.
#         loss_all_ctr = 0.
        
#         if len(dataloader) == 0:
#             continue
            
#         for x, target in dataloader:
#             x = x.to(device)
#             target = target.to(device) # [B, 1, H, W]
            
#             if is_unet_pro:
#                 output, z, shadow = model(x)
#                 if TBFA_FEATURE_SOURCE == 'z':
#                     features = z # [B, D_z]
#                 else:
#                     features = shadow # [B, D_s, H, W]
#             else:
#                 output = model(x)
#                 features = None # 不支持 TBFA
                
#             optimizer.zero_grad()
            
#             # --- 1. 计算 L_seg ---
#             loss_seg = loss_fun_seg(output, target)
#             loss_all_seg += loss_seg.item()
#             train_acc += DiceLoss().dice_coef(output, target).item()
            
#             loss_total = loss_seg
            
#             # --- 2. 计算 L_ctr [cite: 436] ---
#             if barycenters_last_round and features is not None:
#                 # 2a. 获取当前 batch 的 margins 和 bins
#                 _, bins = tbfa_utils.compute_margins_and_bins(
#                     output.detach(), target, TBFA_QUANTILES
#                 )
                
#                 loss_ctr_batch = 0.0
#                 num_tail_groups = 0

#                 # 2b. 投影特征
#                 if TBFA_FEATURE_SOURCE == 'z':
#                     # features [B, D] -> phi_proj [B, D_proj]
#                     phi_proj = features @ R_matrix.T
#                 else:
#                     # features [B, D, H, W] -> phi_proj [B*H*W, D_proj]
#                     phi_proj = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]) @ R_matrix.T

                
#                 for (c, b_idx) in tail_bins_map.keys():
                    
#                     # 2c. 找到 tail bin (c, b_idx) 中的像素/样本
#                     if TBFA_FEATURE_SOURCE == 'z':
#                         # (需要将 bins[B,H,W] 聚合到样本级别)
#                         # 简化：如果样本的 *平均* bin 属于 tail，则使用该样本的 z
#                         sample_bins = torch.mean(bins.float(), dim=(1,2))
#                         mask = (sample_bins == b_idx) # (这是一个粗略的近似)
#                         # (一个更准确的方法是检查 target==c 像素的 bins, 但这更复杂)
#                         # (暂时跳过 'z' 的 L_ctr 实现，因为它不是空间特征)
#                         pass 
                    
#                     else:
#                         # (使用 'shadow' 空间特征)
#                         if target.dim() == 4:
#                             target_flat = target.squeeze(1).reshape(-1) # [B*H*W]
#                         else:
#                             target_flat = target.reshape(-1)
#                         bins_flat = bins.reshape(-1) # [B*H*W]

#                         mask = (target_flat == c) & (bins_flat == b_idx)
                        
#                         if mask.sum() > 0:
#                             phi_tail_proj = phi_proj[mask] # [N_tail, D_proj]
#                             mu_pos = barycenters_last_round[(c, b_idx)]['mean']
#                             mu_negs = neg_barycenters[(c, b_idx)]
                            
#                             loss_ctr_cb = loss_fun_ctr(phi_tail_proj, mu_pos, mu_negs)
#                             loss_ctr_batch += loss_ctr_cb
#                             num_tail_groups += 1

#                 if num_tail_groups > 0:
#                     loss_ctr = (loss_ctr_batch / num_tail_groups) * lambda_ctr
#                     loss_total = loss_total + loss_ctr
#                     loss_all_ctr += loss_ctr.item()

#             loss_total.backward()
#             optimizer.step()
        
#         if len(dataloader) > 0:
#             avg_loss_seg = loss_all_seg / len(dataloader)
#             avg_loss_ctr = loss_all_ctr / len(dataloader)
#             train_acc = train_acc / len(dataloader)
#             logging.info(f'Client: [{cid}] Epoch: [{epoch}] train_loss_seg: {avg_loss_seg:.4f} train_loss_ctr: {avg_loss_ctr:.4f} train_acc: {train_acc:.4f}')

def test(model, dataloader, device, loss_func):
    model.eval()
    loss_all = 0
    test_acc = 0
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    if len(dataloader) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
            loss = loss_func(output, target)
            loss_all += loss.item()
            test_acc += DiceLoss().dice_coef(output, target).item()
        
    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)
    return loss, acc

def main(args):
    set_for_logger(args)
    logging.info(args)
    
    if args.model != 'unet_pro' and args.tbfa_lambda_ctr > 0:
        logging.warning(f"TBFA (lambda_ctr > 0) 需要 'unet_pro' 模型来提取特征，但当前模型是 '{args.model}'. 将禁用 TBFA。")
        args.tbfa_lambda_ctr = 0.0
    
    if TBFA_FEATURE_SOURCE == 'z' and args.tbfa_lambda_ctr > 0:
        logging.warning(f"使用 'z' (bottleneck) 特征进行 L_ctr 在当前实现中被跳过。请使用 'shadow' (TBFA_FEATURE_SOURCE='shadow')。")
        # (如上所示，z 特征的 L_ctr 逻辑被跳过了)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # 2. 动态定义客户端列表
    if args.dataset == 'fundus':
        clients = ['site1', 'site2', 'site3', 'site4']
        num_classes = 3 # (来自 nets/__init__.py)
    elif args.dataset == 'prostate':
        clients = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
        num_classes = 2 # (来自 nets/__init__.py)
    else:
        raise ValueError(f"Unknown client list for dataset: {args.dataset}")

    # 3. build dataset
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # 4. build model
    local_models, global_model = build_model(args, clients, device)

    # --- 新增：初始化 MPS Aggregator (已修改) ---
    mps_aggregator = MPSAggregator(
        device=device,
        gamma=args.mps_gamma,
        temperature=args.mps_temp
    )
    logging.info(f"MPS Aggregator initialized. Gamma={args.mps_gamma}, Temp={args.mps_temp}")
    
    # --- 新增：初始化 TBFA 组件 ---
    # (定义我们要跟踪的 (c, b) 组合)
    tail_bins_map = {}
    for c in range(num_classes):
        for b_idx in TBFA_TAIL_BINS_INDICES:
            tail_bins_map[(c, b_idx)] = len(tail_bins_map)
    logging.info(f"TBFA: 跟踪 {len(tail_bins_map)} 个 (class, bin) 组合。")
    
    # (公共随机映射 R) [cite: 445]
    R_matrix = torch.randn(TBFA_PROJ_DIM, TBFA_FEATURE_DIM, device=device)
    all_personalized_barycenters = None # (用于 r-1 -> r 的滞后更新)
    # --- 新增结束 ---

    # --- 修改：损失函数 ---
    loss_fun_seg = JointLoss() 
    loss_fun_ctr = TBFContrastiveLoss(temperature=args.tbfa_temp)
    
    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    best_dice = 0
    best_dice_round = 0
    best_local_dice = []

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):
        logging.info('-------- Commnication Round: %3d --------'%r)

        # --- 修改： 1. 本地训练 (L_seg + L_ctr) ---
        # (使用上一轮 (r-1) 计算的重心)
        for idx, client in enumerate(clients):
            
            client_barycenters = None
            if all_personalized_barycenters:
                client_barycenters = all_personalized_barycenters[idx]
                
            train_tbfa(
                idx, local_models[idx], train_dls[idx], device, optimizer[idx], 
                args.epochs, loss_fun_seg,
                # TBFA params
                client_barycenters,
                R_matrix.to(device),
                loss_fun_ctr,
                args.tbfa_lambda_ctr if r >= args.sim_start_round else 0.0,
                tail_bins_map
            )
            
        temp_locals = copy.deepcopy(local_models)
        
        # --- 修改：聚合逻辑 (MPS + TBFA) ---
        if r >= args.sim_start_round:
            logging.info('Calculating MPS Similarity and TBFA Statistics...')
            
            all_client_stats = []
            all_logits_list = []
            all_targets_list = []

            # 3a. 提取 Logits (用于MPS) 和 Stats (用于TBFA) [cite: 451]
            for idx, client in enumerate(clients):
                stats, logits, targets = get_client_stats_and_logits(
                    temp_locals[idx], val_dls[idx], device, 
                    R_matrix.cpu(), # 统计在 CPU 上计算
                    num_classes
                )
                all_client_stats.append(stats)
                all_logits_list.append(logits)
                all_targets_list.append(targets)
            
            # 3b. (服务器) 计算 MPS 相似度矩阵 S 
            S_matrix = mps_aggregator.compute_similarity_matrix(
                all_logits_list, all_targets_list
            )
            
            # 3c. (服务器) 计算 TBFA 重心 (用于下一轮 r+1) [cite: 469]
            if args.tbfa_lambda_ctr > 0:
                all_personalized_barycenters = tbfa_utils.compute_barycenters(
                    all_client_stats,
                    S_matrix.to('cpu'), # 重心计算在 CPU 上完成
                    len(clients),
                    tail_bins_map
                )
                logging.info(f'TBFA: Calculated {len(all_personalized_barycenters)} personalized barycenters for next round.')
            
            # 3d. (服务器) 计算 MPS 聚合权重
            aggr_weights = mps_aggregator.weights_from_similarity(S_matrix)
            logging.info(f'MPS Weights: {aggr_weights.cpu().numpy()}')
            
            # 3e. (服务器) 执行模型聚合 [cite: 488]
            communication(global_model, temp_locals, aggr_weights)

        else: 
            # 3f. 早期轮次使用 FedAvg
            logging.info('Using standard FedAvg aggregation.')
            communication(global_model, temp_locals, client_weight_tensor)
            all_personalized_barycenters = None # 重置重心
        # --- 修改结束 ---


        # 4. 分发全局模型
        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r % args.test_step == 0:
            # 5. 测试
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun_seg)
                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice) if len(avg_dice) > 0 else 0
            avg_loss_v = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0
            
            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            # 7. 保存最佳模型
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx] if idx < len(best_local_dice) else 0.0))


if __name__ == '__main__':
    args = get_args()
    main(args)