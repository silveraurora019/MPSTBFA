# 文件名: tbfa_utils.py
# -*- coding: utf-8 -*-
"""
MPS-TBFA (Algorithm 1) 的辅助函数 [cite: 443]
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Sequence

@torch.no_grad()
def compute_margins_and_bins(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    quantiles: Sequence[float]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为分割任务计算空间 margin 和对应的分位 bin。
    
    logits: [B, C, H, W]
    targets: [B, 1, H, W] 或 [B, H, W]
    quantiles: 升序排列的分位点 (例如 [0.1, 0.25, 0.5])
    
    返回: (margins [B, H, W], bins [B, H, W])
    bins[p] = k 表示像素 p 的 margin 位于第 k 个分位区间
    """
    
    if targets.dim() == 4:
        targets = targets.squeeze(1) # -> [B, H, W]
    
    B, C, H, W = logits.shape
    
    # 1. 计算 Margins [cite: 407]
    # 展平以便使用 gather
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
    targets_flat = targets.reshape(-1) # [B*H*W]
    
    true_logits = logits_flat.gather(dim=1, index=targets_flat.long().view(-1, 1)).squeeze(1)
    
    mask = torch.ones_like(logits_flat, dtype=torch.bool)
    mask[torch.arange(logits_flat.size(0)), targets_flat.long()] = False
    wrong_logits = logits_flat.masked_fill(~mask, float("-inf"))
    max_wrong_logits, _ = wrong_logits.max(dim=1)
    
    margins_flat = true_logits - max_wrong_logits
    margins = margins_flat.reshape(B, H, W) # [B, H, W]

    # 2. 计算分位点 (Bins)
    if margins.numel() == 0:
        return margins, torch.zeros_like(targets, dtype=torch.long)

    # 计算全局分位点阈值
    q_thresholds = torch.quantile(
        margins_flat.float(), 
        torch.tensor(quantiles, device=margins.device)
    )
    
    # 使用 searchsorted 找到每个 margin 属于哪个 bin
    # bins[p] = 0 if margin < q[0]
    # bins[p] = 1 if q[0] <= margin < q[1]
    # ...
    # bins[p] = K if margin >= q[K-1]
    bins_flat = torch.searchsorted(q_thresholds, margins_flat)
    bins = bins_flat.reshape(B, H, W).long() # [B, H, W]
    
    return margins, bins


@torch.no_grad()
def extract_feature_statistics(
    features: torch.Tensor, 
    targets: torch.Tensor, 
    bins: torch.Tensor, 
    tail_bins_indices: List[int], 
    num_classes: int
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    提取困难子集 (tail bins) 的特征统计量 [cite: 452]。
    
    features: 特征图 (z or shadow) [B, D, H_f, W_f]
    targets: 目标 [B, 1, H, W]
    bins: 空间分位图 [B, H, W]
    tail_bins_indices: 要提取的 bin 索引 (例如 [0, 1])
    num_classes: 类别数 (C)
    
    返回: stats[ (c, b) ] = {'mean': [D], 'cov': [D, D]}
    """
    
    B, D, H_f, W_f = features.shape
    _, _, H_t, W_t = targets.shape

    # 1. 调整特征和目标的尺寸以匹配
    # (假设 target/bins 的 H,W 与 feature 的 H_f, W_f 匹配或可插值)
    if (H_f != H_t) or (W_f != W_t):
        # 将 bins 和 targets 下采样到特征图大小
        targets_resized = F.interpolate(targets.float(), size=(H_f, W_f), mode='nearest').long()
        bins_resized = F.interpolate(bins.float().unsqueeze(1), size=(H_f, W_f), mode='nearest').long().squeeze(1)
    else:
        targets_resized = targets
        bins_resized = bins
        
    if targets_resized.dim() == 4:
         targets_resized = targets_resized.squeeze(1) # [B, H_f, W_f]

    features_permuted = features.permute(0, 2, 3, 1) # [B, H_f, W_f, D]
    
    stats = {}

    for c in range(num_classes):
        for b in tail_bins_indices:
            # 找到类别 c 且在 tail bin b 中的所有像素 [cite: 410]
            mask = (targets_resized == c) & (bins_resized == b) # [B, H_f, W_f]
            
            if mask.sum() > 0:
                # 提取对应的所有特征向量
                phi_cb = features_permuted[mask] # [N_cb, D]
                
                if phi_cb.shape[0] > 1:
                    # 计算均值和协方差 [cite: 412, 413]
                    mean_cb = torch.mean(phi_cb, dim=0)
                    # (使用 torch.cov)
                    cov_cb = torch.cov(phi_cb.T)
                    
                    stats[(c, b)] = {
                        'mean': mean_cb.cpu(),
                        'cov': cov_cb.cpu(),
                        'count': phi_cb.shape[0]
                    }
    
    return stats


@torch.no_grad()
def compress_statistics(
    stats: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    R: torch.Tensor
) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
    """
    使用随机映射 R 压缩统计量 [cite: 453]。
    R: [D_proj, D]
    """
    compressed_stats = {}
    R = R.to(torch.float32)
    
    for (c, b), s_dict in stats.items():
        mu = s_dict['mean'].to(R.device, dtype=torch.float32)
        cov = s_dict['cov'].to(R.device, dtype=torch.float32)
        
        # 隐私压缩 [cite: 453]
        mu_hat = R @ mu
        cov_hat = R @ cov @ R.T
        
        compressed_stats[(c, b)] = {
            'mean': mu_hat.cpu(), # [D_proj]
            'cov': cov_hat.cpu(), # [D_proj, D_proj]
            'count': s_dict['count']
        }
    return compressed_stats


@torch.no_grad()
def compute_barycenters(
    all_clients_stats: List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]],
    mps_similarity_matrix: torch.Tensor,
    num_clients: int,
    tail_bins_map: Dict[Tuple[int, int], int] # (c,b) -> idx
) -> List[Dict[Tuple[int, int], Dict[str, torch.Tensor]]]:
    """
    服务器端：为每个客户端计算个性化的邻域重心 [cite: 418, 469]。
    
    all_clients_stats: List[client_stats]
    mps_similarity_matrix: S [N, N]
    
    返回: List[personalized_barycenters_j]
          personalized_barycenters_j[(c, b)] = {'mean': ..., 'cov': ...}
    """
    
    all_personalized_barycenters = []

    for j in range(num_clients): # 目标客户端 j [cite: 463]
        
        # 1. 计算客户端 j 的归一化邻域权重 alpha [cite: 465]
        S_j = mps_similarity_matrix[j, :] # [N]
        alpha_j = S_j / (S_j.sum() + 1e-8) # [N]
        
        personalized_barycenters_j = {}

        for (c, b) in tail_bins_map.keys():
            
            # 2. 收集所有客户端关于 (c,b) 的统计数据
            mus_i = [] # List[ \hat{\mu}_i ]
            covs_i = [] # List[ \hat{\Sigma}_i ]
            counts_i = []
            valid_client_indices = []

            for i in range(num_clients):
                if (c, b) in all_clients_stats[i]:
                    stats_i_cb = all_clients_stats[i][(c, b)]
                    mus_i.append(stats_i_cb['mean'])
                    covs_i.append(stats_i_cb['cov'])
                    counts_i.append(stats_i_cb['count'])
                    valid_client_indices.append(i)
            
            if not valid_client_indices:
                continue # 没有客户端报告这个 bin

            # 3. 计算个性化均值重心 (Mean Barycenter) [cite: 469]
            # (只在报告了 (c,b) 的客户端子集上加权)
            
            valid_indices_tensor = torch.tensor(valid_client_indices, dtype=torch.long)
            alpha_j_subset = alpha_j[valid_indices_tensor]
            alpha_j_cb = alpha_j_subset / (alpha_j_subset.sum() + 1e-8) # [N_valid]
            
            mus_i_tensor = torch.stack(mus_i).float() # [N_valid, D_proj]
            
            # \overline{\mu}_j = \sum_i \alpha_{j<-i} \hat{\mu}_i [cite: 469]
            mu_bary_j_cb = torch.einsum('i,id->d', alpha_j_cb, mus_i_tensor)
            
            # 4. TODO: 计算协方差重心 (Cov Barycenter) [cite: 471]
            # (Fréchet/Bures 近似计算量大，这里暂时用加权平均代替作为简化)
            covs_i_tensor = torch.stack(covs_i).float() # [N_valid, D_proj, D_proj]
            cov_bary_j_cb = torch.einsum('i,idk->dk', alpha_j_cb, covs_i_tensor)

            personalized_barycenters_j[(c, b)] = {
                'mean': mu_bary_j_cb,
                'cov': cov_bary_j_cb
            }
            
        all_personalized_barycenters.append(personalized_barycenters_j)

    return all_personalized_barycenters