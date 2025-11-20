# 文件名: aggregator_mps.py
# -*- coding: utf-8 -*-
import torch
import logging
import mps_similarity as mps

class MPSAggregator:
    """
    MPS 聚合器：
    1. 接收各客户端的 Logits 和 Targets。
    2. 调用 mps_similarity 计算 Margin Profile。
    3. 计算相似度矩阵 S。
    4. 将相似度转换为聚合权重 w。
    """
    def __init__(self, device, gamma=1.0, temperature=0.1):
        self.device = device
        self.gamma = gamma
        self.temperature = temperature
        # 定义分位点，可以根据需要调整
        self.quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)

    def _get_valid_logits_targets(self, logits_list, targets_list):
        """ 内部函数：过滤无效数据 """
        valid_logits = []
        valid_targets = []
        valid_indices = []
        
        for i, (l, t) in enumerate(zip(logits_list, targets_list)):
            if l is not None and t is not None and l.shape[0] > 0:
                valid_logits.append(l.to(self.device))
                valid_targets.append(t.to(self.device))
                valid_indices.append(i)
            else:
                logging.warning(f"发现无效或空的 logits/targets，跳过客户端 {i}。")
        
        return valid_logits, valid_targets, valid_indices

    def compute_similarity_matrix(self, logits_list, targets_list):
        """
        计算 MPS 相似度矩阵 S [M, M]
        :param logits_list: List[Tensor], 原始 M 个客户端的 logits
        :param targets_list: List[Tensor], 原始 M 个客户端的 targets
        :return: S [M, M] (如果客户端 k 无效，S[k,:] 和 S[:,k] 将为 0，对角线为 1)
        """
        M = len(logits_list)
        S_mps_full = torch.eye(M, device=self.device) # 默认返回单位阵

        valid_logits, valid_targets, valid_indices = self._get_valid_logits_targets(
            logits_list, targets_list
        )
        
        M_valid = len(valid_logits)

        if M_valid < 2:
            logging.warning("有效客户端数量少于2，无法计算 MPS 相似度。返回单位矩阵。")
            return S_mps_full

        try:
            # 2. 计算相似度矩阵 S_valid [M_valid, M_valid]
            S_mps_valid = mps.mps_from_logits_targets(
                logits_list=valid_logits,
                targets_list=valid_targets,
                quantiles=self.quantiles,
                gamma=self.gamma
            )
            
            logging.info(f"MPS 相似度矩阵 (有效客户端):\n{S_mps_valid.cpu().numpy()}")

            # 3. 将 S_valid 映射回 S_mps_full [M, M]
            # (这确保了权重和重心计算的索引一致性)
            idx_tensor = torch.tensor(valid_indices, device=self.device)
            S_mps_full[idx_tensor[:, None], idx_tensor] = S_mps_valid
            
            return S_mps_full

        except Exception as e:
            logging.error(f"MPS 相似度矩阵计算过程中出错: {e}")
            return S_mps_full # 返回单位阵作为回退

    def weights_from_similarity(self, S_matrix):
        """
        从相似度矩阵 S [M, M] 计算聚合权重 w [M]。
        """
        # w_i ∝ sum_j S_ij
        w_score = S_matrix.sum(dim=1) # [M]
        
        if self.temperature > 0:
            weights = torch.softmax(w_score / self.temperature, dim=0)
        else:
            if w_score.sum() > 1e-8:
                weights = w_score / w_score.sum()
            else:
                # 极端情况 (例如 S 全为 0)
                weights = torch.ones_like(w_score) / w_score.numel()
                
        return weights


    def compute_weights(self, logits_list, targets_list):
        """
        (保留旧接口) 计算聚合权重 w [M]
        """
        S_matrix = self.compute_similarity_matrix(logits_list, targets_list)
        weights = self.weights_from_similarity(S_matrix)
        return weights