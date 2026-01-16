import torch
from torch import nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, emb_dim=256, num_heads=1):
        super(CrossAttention, self).__init__()
        self.neg_num = 16
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.num_heads = num_heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.head_dim = emb_dim // num_heads

        # Q, K, V 的线性投影
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, query, key, value):
        B, Lq, _ = query.size()
        _, Lk, _ = key.size()

        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.q_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.q_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_head, Lq, Lk]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch_size, num_head, Lq, Lk]

        attn_output = torch.matmul(attn_probs, V)  # [batch_size, num_head, Lq, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Lq, self.emb_dim)  # [batch_size, Lq, emb_dim]

        output = self.out_proj(attn_output)  # [batch_size, Lq, emb_dim]
        return output, attn_probs


class AlignLoss_CrossAttention(nn.Module):
    def __init__(self, emb_dim, sub_dim=256, neg_num=16, num_heads=4):
        super().__init__()
        self.neg_num = neg_num
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.emb_dim = emb_dim

        # Cross-Attention 模块
        self.cross_str_vis = CrossAttention(sub_dim, num_heads)
        self.cross_str_txt = CrossAttention(sub_dim, num_heads)

    def structure_aware_modality_align(self, str_embedding, mod_embedding):
        """
        :param str_embedding:
        :param mod_embedding:
        :return:
        """
        str_embedding = torch.nn.functional.normalize(str_embedding, p=2, dim=-1, eps=1e-5)  # [num_ent, emb_dim]
        mod_embedding = torch.nn.functional.normalize(mod_embedding, p=2, dim=-1, eps=1e-5)  # [num_ent, emb_dim]
        bs, _ = str_embedding.size()
        neg_sample_id = torch.randint(0, bs, [bs, self.svc_neg_num])  # [num_ent, num_sample]
        neg_str_feat = str_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        neg_mod_feat = mod_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        str_samples = torch.cat([str_embedding.unsqueeze(1), neg_str_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        mod_samples = torch.cat([mod_embedding.unsqueeze(1), neg_mod_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        s2m_score = torch.matmul(mod_samples, str_embedding.unsqueeze(2)).squeeze(
            2) / self.svc_temperature  # [num_ent, 1+num_sample]
        m2s_score = torch.matmul(str_samples, mod_embedding.unsqueeze(2)).squeeze(
            2) / self.svc_temperature  # [num_ent, 1+num_sample]
        label = torch.zeros([bs, ], dtype=torch.long).to(str_embedding.device)
        s2v_loss = torch.nn.functional.cross_entropy(s2m_score, label)
        v2s_loss = torch.nn.functional.cross_entropy(m2s_score, label)
        svc_loss = 0.5 * (s2v_loss + v2s_loss)
        return svc_loss

    def forward(self, str_embedding, vis_embedding, txt_embedding):
        # Cross-Attention 对齐
        aligned_str_vis, _ = self.cross_str_vis(str_embedding.unsqueeze(1), vis_embedding.unsqueeze(1),
                                                vis_embedding.unsqueeze(1))
        aligned_str_txt, _ = self.cross_str_txt(str_embedding.unsqueeze(1), txt_embedding.unsqueeze(1),
                                                txt_embedding.unsqueeze(1))

        # squeeze 去掉 sequence 维度
        aligned_str_vis = aligned_str_vis.squeeze(1)
        aligned_str_txt = aligned_str_txt.squeeze(1)

        # 计算两个模态对齐损失
        loss_1 = self.structure_aware_modality_align(aligned_str_vis, vis_embedding)
        loss_2 = self.structure_aware_modality_align(aligned_str_txt, txt_embedding)

        return loss_1 + loss_2
