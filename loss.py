# import torchvision.transforms as transforms
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from typing import Tuple



# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0, activation='sigmoid'):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
#         self.activation = activation

#     def dice_coef(self, pred, gt):
#         """计算 Dice 系数 (用于评估，非损失)"""
#         softmax_pred = torch.nn.functional.softmax(pred, dim=1)
#         seg_pred = torch.argmax(softmax_pred, dim=1)
#         all_dice = 0
        
#         if gt.dim() == 4:
#             gt = gt.squeeze(dim=1) # (B, 1, H, W) -> (B, H, W)
            
#         if gt.dim() != 3:
#              # 确保 gt 是 (B, H, W)
#              return 0.0

#         batch_size = gt.shape[0]
#         num_class = softmax_pred.shape[1]

#         # 遍历所有类别 (包括背景)
#         for i in range(num_class):
#             each_pred = torch.zeros_like(seg_pred)
#             each_pred[seg_pred==i] = 1

#             each_gt = torch.zeros_like(gt)
#             each_gt[gt==i] = 1            

        
#             intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
#             union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
#             # 添加平滑项 eps
#             dice = (2. * intersection + 1e-5) / (union + 1e-5)
         
#             all_dice += torch.mean(dice)
 
#         return all_dice * 1.0 / num_class


#     def forward(self, pred, gt):
#         """计算 Dice 损失"""
#         sigmoid_pred = F.softmax(pred,dim=1)

#         batch_size = gt.shape[0]
#         num_class = sigmoid_pred.shape[1]
    
#         # (B, 1, H, W) -> (B, H, W)
#         if gt.dim() == 4:
#             gt = gt.squeeze(dim=1)
            
#         # 创建 one-hot 编码的 gt
#         # 假设 gt 是 (B, H, W) 且类别是 0, 1, ... num_class-1
#         label_one_hot = F.one_hot(gt.long(), num_classes=num_class).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)

#         loss = 0
#         smooth = 1e-5

#         for i in range(num_class):
#             intersect = torch.sum(sigmoid_pred[:, i, ...] * label_one_hot[:, i, ...])
#             z_sum = torch.sum(sigmoid_pred[:, i, ...] )
#             y_sum = torch.sum(label_one_hot[:, i, ...] )
#             loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            
#         # 平均 Dice 系数，然后 1 - Dice
#         loss = 1 - loss / num_class
#         return loss

# class JointLoss(nn.Module):
#     def __init__(self, n_classes=2):
#         super(JointLoss, self).__init__()
#         # CrossEntropyLoss 期望 gt 为 (B, H, W) 且类型为 Long
#         self.ce = nn.CrossEntropyLoss()
#         self.dice = DiceLoss()
#         self.n_classes = n_classes

#     def forward(self, pred, gt):
#         # 确保 gt 格式正确
#         if gt.dim() == 4:
#              gt = gt.squeeze(axis=1) # (B, 1, H, W) -> (B, H, W)
             
#         # 检查 gt 是否包含超出范围的类别
#         if gt.max() >= self.n_classes:
#              gt = torch.clamp(gt, max=self.n_classes - 1)
             
#         ce_loss = self.ce(pred, gt.long())
#         dice_loss = self.dice(pred, gt)
        
#         return (ce_loss + dice_loss) / 2
    

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            dice = (2. *  intersection )/ (union + 1e-5)
         
            all_dice += torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
    
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)
        
        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, gt):
        ce =  self.ce(pred, gt.squeeze(axis=1).long())
        return (ce + self.dice(pred, gt))/2
    

# --- 新增：MPS-TBFA 对比损失 ---
class TBFContrastiveLoss(nn.Module):
    """
    MPS-TBFA 对比蒸馏损失 (Eq. 7) [cite: 434]
    """
    def __init__(self, temperature=0.1):
        super(TBFContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, phi_tail, mu_pos, mu_negs):
        """
        phi_tail: [N_tail, D_proj] - 困难样本的 (投影后) 特征
        mu_pos: [D_proj] - 对应的 (投影后) 正重心
        mu_negs: List[ [D_proj] ] - 负重心列表
        """
        if phi_tail.shape[0] == 0:
            return torch.tensor(0.0, device=phi_tail.device)
            
        mu_pos = mu_pos.to(phi_tail.device)

        # (N_tail, 1)
        pos_sim = self.cosine_similarity(phi_tail, mu_pos.unsqueeze(0)).unsqueeze(-1)

        neg_sims = []
        for mu_n in mu_negs:
            mu_n = mu_n.to(phi_tail.device)
            # (N_tail, 1)
            neg_sim = self.cosine_similarity(phi_tail, mu_n.unsqueeze(0)).unsqueeze(-1)
            neg_sims.append(neg_sim)

        if not neg_sims:
            # 只有正样本 (罕见)
            return -torch.mean(pos_sim / self.temperature)

        # (N_tail, N_neg)
        neg_sims_tensor = torch.cat(neg_sims, dim=1)
        
        # (N_tail, 1 + N_neg)
        all_sims = torch.cat([pos_sim, neg_sims_tensor], dim=1)
        
        # InfoNCE 损失 [cite: 434]
        log_probs = F.log_softmax(all_sims / self.temperature, dim=1)
        
        # 取第 0 列 (正样本)
        loss = -torch.mean(log_probs[:, 0])
        
        return loss


def newton_schulz_matrix_sqrt(A, num_iters=5):
    """
    使用 Newton-Schulz 迭代法计算矩阵平方根 (支持反向传播)。
    A: [D, D] 对称正定矩阵
    """
    batch_size = A.shape[0]
    dim = A.shape[1]
    
    # 归一化以保证收敛
    norm_A = A.norm(p='fro')
    Y = A.div(norm_A)
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    Z = torch.eye(dim, device=A.device, dtype=A.dtype)

    for i in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)
        
    return Y * torch.sqrt(norm_A)

class TBFFeatureAlignmentLoss(nn.Module):
    """
    MPS-TBFA 特征分布对齐损失 (稳定版)
    
    修改说明：
    原论文建议使用 Bures 距离 (||Sigma^1/2 - Sigma_bar^1/2||)，但这涉及矩阵开方的反向传播，
    在样本量少或协方差矩阵病态时极易导致梯度爆炸。
    
    此处改为使用 Frobenius 范数 (欧氏距离) 直接对齐协方差矩阵：
    L_align = ||mu - mu_bar||^2 + lambda_cov * ||Sigma - Sigma_bar||^2_F
    这种方式数值极其稳定，且能达到相似的特征对齐效果。
    """
    def __init__(self, lambda_cov=1.0):
        super(TBFFeatureAlignmentLoss, self).__init__()
        self.lambda_cov = lambda_cov

    def forward(self, phi_tail, mu_bary, cov_bary):
        """
        phi_tail: [N_tail, D_proj] 当前 batch 的困难样本特征
        mu_bary:  [D_proj] 目标均值重心 (no_grad)
        cov_bary: [D_proj, D_proj] 目标协方差重心 (no_grad)
        """
        # 1. 样本检查
        N, D = phi_tail.shape
        if N < 2:
            # 样本太少无法计算有效的协方差，返回 0 梯度
            return torch.tensor(0.0, device=phi_tail.device, requires_grad=True)

        # 2. 计算当前 Batch 的统计量
        # mu: [D]
        mu_curr = torch.mean(phi_tail, dim=0)
        
        # cov: [D, D]
        # 中心化
        phi_centered = phi_tail - mu_curr.unsqueeze(0)
        # 使用无偏估计 (N-1)
        cov_curr = torch.matmul(phi_centered.T, phi_centered) / (N - 1)
        
        # [重要] 简单的数值保护 (防止全0)
        # 虽然欧氏距离不需要矩阵正定，但这有助于防止后续计算异常
        cov_curr = cov_curr + torch.eye(D, device=phi_tail.device) * 1e-6

        # 3. 计算损失
        # (A) 均值对齐: MSE(mu, mu_bar)
        loss_mu = F.mse_loss(mu_curr, mu_bary)

        # (B) 协方差对齐: MSE(Sigma, Sigma_bar)  <--- 关键修改
        # 直接让两个矩阵的元素逼近，避免了 matrix_sqrt
        loss_cov = F.mse_loss(cov_curr, cov_bary)

        # 总损失
        loss = loss_mu + self.lambda_cov * loss_cov
        
        return loss
