import os
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Siamese(nn.Module):
    def __init__(
            self,
            num_ent,
            num_rel,
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout=0.1,
            emb_dropout=0.6,
            vis_dropout=0.1,
            txt_dropout=0.1,
            visual_token_index=None,
            text_token_index=None,
            score_function="tucker"
    ):
        super(Siamese, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.data_type = torch.float32
        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/textual.pth")
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)

        self.bert = True
        self.score_function = score_function

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        false_ents = torch.full((self.num_ent, 1), False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim=1)

        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel, 1), False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim=1)

        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1, dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1, dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1, 1, dim_str))

        self.proj_ent_vis = nn.Linear(32, dim_str)
        self.proj_ent_txt = nn.Linear(768, dim_str)

        # TODO
        self.align_model = AlignLoss_CrossAttention()  # 无参数需要更新
        # TODO
        # self.context_vec = nn.Parameter(torch.randn((1, dim_str)))
        # self.act = nn.Softmax(dim=1)
        # self.scale = torch.Tensor([1. / np.sqrt(self.dim_str)]).cuda()
        self.fusion_model = Attention_Fusion(dim_str)
        # self.fusion_model = SSM_Fusion(dim_str)
        # TODO

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)

        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.num_con = 256
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            self.tucker_decoder = TuckER_Decoder(dim_str, dim_str)
        elif self.score_function == "transe":
            self.transE_decoder = TransE_Decoder()
        elif self.score_function == "rotate":
            self.rotate_decoder = RotatE_Decoder(dim_str)
        else:
            pass
        self.init_weights()
        # torch.save(self.visual_token_embedding, open("tokens/visual_token.pth", "wb"))
        # torch.save(self.text_token_embedding, open("tokens/textual_token.pth", "wb"))

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self):

        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent

        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        ent_tkn2 = ent_tkn.squeeze(1)
        ent_seq1 = torch.cat([ent_tkn, rep_ent_str, ], dim=1)
        ent_seq2 = torch.cat([ent_tkn, rep_ent_vis, ], dim=1)
        ent_seq3 = torch.cat([ent_tkn, rep_ent_txt], dim=1)
        str_embdding = self.ent_encoder(ent_seq1)[:, 0]
        vis_embdding = self.ent_encoder(ent_seq2)[:, 0]
        txt_embdding = self.ent_encoder(ent_seq3)[:, 0]

        cands = torch.stack([ent_tkn2, str_embdding, vis_embdding, txt_embdding],
                            dim=1)  # [batch_size, num_modality, str_dim]
        # x = torch.arange(self.num_ent).cuda()

        # TODO
        # context_vec = self.context_vec  # context_vec:[1, str_dim]
        # att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)  # [batch_size, num_modality, 1]
        # att_weights = self.act(att_weights)  # [batch_size, num_modality, 1]
        # ent_embs = torch.sum(att_weights * cands, dim=1)  # [batch_size, str_dim]
        ent_embs = self.fusion(cands)
        # TODO

        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))
        return torch.cat([ent_embs, self.lp_token], dim=0), rep_rel_str.squeeze(dim=1), [str_embdding, vis_embdding,
                                                                                         txt_embdding]

    def score(self, emb_ent, emb_rel, triples):
        mask = (triples == self.num_ent + self.num_rel)  # [batch_size, 3]
        if self.bert == False:
            if self.score_function == "transE":
                scores = self.TransE(emb_ent, emb_rel, triples, mask)
            elif self.score_function == "rotatE":
                scores = self.RotatE(emb_ent, emb_rel, triples, mask)
            else:
                raise NotImplementedError
        else:
            # bert
            h_seq = emb_ent[triples[:, 0] - self.num_rel].unsqueeze(dim=1) + self.pos_head
            r_seq = emb_rel[triples[:, 1] - self.num_ent].unsqueeze(dim=1) + self.pos_rel
            t_seq = emb_ent[triples[:, 2] - self.num_rel].unsqueeze(dim=1) + self.pos_tail

            dec_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)
            output_dec = self.decoder(dec_seq)
            rel_emb = output_dec[:, 1, :]
            ctx_emb = output_dec[triples == self.num_ent + self.num_rel]

            if self.score_function == "rescal":
                scores = self.tucker_decoder(ctx_emb, rel_emb, emb_ent)
            elif self.score_function == "complex":
                scores = self.tucker_decoder(ctx_emb, rel_emb, emb_ent)
            elif self.score_function == "tucker":
                scores = self.tucker_decoder(ctx_emb, rel_emb, emb_ent)
            else:
                raise NotImplementedError
        return scores

    def align(self, emb_list):
        """
        模态对齐
        :param emb_list:
        :return:
        """
        str_embdding, vis_embdding, txt_embdding = emb_list
        align_loss = self.align_model(str_embdding, vis_embdding, txt_embdding)
        return align_loss

    def fusion(self, x):
        return self.fusion_model(x)

    # 方式四. pos_neg_logits_vectorized_topk(选取top_k)
    def pos_neg_logits_vectorized_topk(self, score, label, filter_mask, neg_num=3):
        """
        实体负采样
        :param score: [batch_size, num_entity]
        :param label: [batch_size,]
        :param filter_mask: [batch_size, num_entity]
        :return: pos_logits[batch_size,]  neg_logits[batch_size, neg_num]
        """
        # 1. softmax
        logits = torch.softmax(score, dim=-1)
        # 2. 正样本pos_logit
        pos_logits = logits.gather(1, label.unsqueeze(1)).squeeze(1)
        # 3. 负样本neg_logit
        masked_score = score.clone()
        masked_score[filter_mask] = -float('inf')
        masked_score.scatter_(1, label.unsqueeze(1), -float('inf'))
        # 5. 寻找困难负样本
        _, neg_idx = masked_score.topk(k=neg_num, dim=1)
        neg_logits = logits.gather(1, neg_idx)
        return pos_logits, neg_logits

    def query(self, emb_ent, emb_rel, triples):
        """
        :param emb_ent: [num_ent, str_dim]
        :param emb_rel: [num_rel, str_dim]
        :param triples: [batch_size, 3]
        :return: [batch_size, num_entity]
        """
        h_seq = emb_ent[triples[:, 0] - self.num_rel].unsqueeze(1) + self.pos_head  # [batch_size, 1, str_dim]
        r_seq = emb_rel[triples[:, 1] - self.num_ent].unsqueeze(1) + self.pos_rel  # [batch_size, 1, str_dim]
        t_seq = emb_ent[triples[:, 2] - self.num_rel].unsqueeze(1) + self.pos_tail  # [batch_size, 1, str_dim]
        triple_seq = torch.cat([h_seq, r_seq, t_seq], dim=1)  # [batch_size, 3, str_dim]
        triple_out = self.decoder(triple_seq)  # [batch_size, 3, str_dim]
        query_out = triple_out[triples == self.num_ent + self.num_rel]  # [batch_size, str_dim]
        return query_out


"""
    模态对齐
"""


class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()
        self.neg_num = 16
        self.temperature = 0.02

    def structure_modality_contrastive(self, str_embedding, mod_embedding):
        """
        :param str_embedding: [num_ent, emb_dim]
        :param mod_embedding: [num_ent, emb_dim]
        :return:
        """
        str_embedding = torch.nn.functional.normalize(str_embedding, p=2, dim=-1, eps=1e-5)
        mod_embedding = torch.nn.functional.normalize(mod_embedding, p=2, dim=-1, eps=1e-5)
        bs, _ = str_embedding.size()
        neg_sample_id = torch.randint(0, bs, [bs, self.neg_num])  # [num_ent, num_sample]
        neg_str_feat = str_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        neg_mod_feat = mod_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        str_samples = torch.cat([str_embedding.unsqueeze(1), neg_str_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        mod_samples = torch.cat([mod_embedding.unsqueeze(1), neg_mod_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        s2m_score = torch.matmul(mod_samples, str_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
        m2s_score = torch.matmul(str_samples, mod_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
        label = torch.zeros([bs, ], dtype=torch.long).to(str_embedding.device)  # [num_ent] 第0个是正确实体
        s2v_loss = torch.nn.functional.cross_entropy(s2m_score, label)
        v2s_loss = torch.nn.functional.cross_entropy(m2s_score, label)
        svc_loss = 0.5 * (s2v_loss + v2s_loss)
        return svc_loss

    def forward(self, str_emb, vis_emb, txt_emb):
        loss1 = self.structure_modality_contrastive(str_emb, vis_emb)
        loss2 = self.structure_modality_contrastive(str_emb, txt_emb)
        loss = loss1 + loss2
        return loss


class AlignLoss_Parameter(nn.Module):
    def __init__(self):
        super(AlignLoss_Parameter, self).__init__()
        self.neg_num = 16
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.proj = nn.Linear(256, 1024)

    def structure_modality_contrastive(self, str_embedding, mod_embedding):
        """
        :param str_embedding: [num_ent, emb_dim]
        :param mod_embedding: [num_ent, emb_dim]
        :return:
        """
        str_embedding = torch.nn.functional.normalize(self.proj(str_embedding), p=2, dim=-1, eps=1e-5)
        mod_embedding = torch.nn.functional.normalize(self.proj(mod_embedding), p=2, dim=-1, eps=1e-5)
        bs, _ = str_embedding.size()
        neg_sample_id = torch.randint(0, bs, [bs, self.neg_num])  # [num_ent, num_sample]
        neg_str_feat = str_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        neg_mod_feat = mod_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        str_samples = torch.cat([str_embedding.unsqueeze(1), neg_str_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        mod_samples = torch.cat([mod_embedding.unsqueeze(1), neg_mod_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        s2m_score = torch.matmul(mod_samples, str_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
        m2s_score = torch.matmul(str_samples, mod_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
        label = torch.zeros([bs, ], dtype=torch.long).to(str_embedding.device)  # [num_ent] 第0个是正确实体
        s2v_loss = torch.nn.functional.cross_entropy(s2m_score, label)
        v2s_loss = torch.nn.functional.cross_entropy(m2s_score, label)
        svc_loss = 0.5 * (s2v_loss + v2s_loss)
        return svc_loss

    def forward(self, str_emb, vis_emb, txt_emb):
        loss1 = self.structure_modality_contrastive(str_emb, vis_emb)
        loss2 = self.structure_modality_contrastive(str_emb, txt_emb)
        loss = loss1 + loss2
        return loss


class AlignLoss_Triangle(nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_num = 16
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.proj = nn.Linear(256, 1024)

    def triangle_area(self, x, y, z):
        """
        x, y, z: [B, D] (L2 normalized)
        return: [B]
        """
        u = x - y
        v = x - z
        uu = torch.sum(u * u, dim=-1)
        vv = torch.sum(v * v, dim=-1)
        uv = torch.sum(u * v, dim=-1)
        area = 0.5 * (uu * vv - uv * uv)
        return area

    def forward(self, str_emb, vis_emb, txt_emb):
        # 1. normalize
        str_emb = F.normalize(self.proj(str_emb), dim=-1)
        vis_emb = F.normalize(self.proj(vis_emb), dim=-1)
        txt_emb = F.normalize(self.proj(txt_emb), dim=-1)

        bs, _ = str_emb.size()

        # 2. negative sampling
        neg_ids = torch.randint(0, bs, (bs, self.neg_num), device=str_emb.device)
        neg_str = str_emb[neg_ids]

        # 3. positive / negative triangle area
        pos_area = self.triangle_area(str_emb, vis_emb, txt_emb)  # [B]
        neg_area = self.triangle_area(
            neg_str, vis_emb.unsqueeze(1), txt_emb.unsqueeze(1)
        )  # [B, neg_num]

        # 4. InfoNCE
        logits = torch.cat(
            [-pos_area.unsqueeze(1), -neg_area], dim=1
        ) / self.temperature

        labels = torch.zeros(bs, dtype=torch.long, device=str_emb.device)
        loss = F.cross_entropy(logits, labels)

        return loss


class CrossAttention(nn.Module):
    def __init__(self, emb_dim=256, num_heads=1):
        super(CrossAttention, self).__init__()
        self.neg_num = 16
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.num_heads = num_heads
        self.emb_dim = emb_dim
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
    def __init__(self, emb_dim=256, neg_num=16, num_heads=2):
        super().__init__()
        self.neg_num = neg_num
        self.temperature = nn.Parameter(torch.tensor(0.02))
        self.emb_dim = emb_dim

        # Cross-Attention 模块
        self.cross_str_vis = CrossAttention(emb_dim, num_heads)
        self.cross_str_txt = CrossAttention(emb_dim, num_heads)

    def structure_aware_modality_align(self, str_embedding, mod_embedding):
        """
        :param str_embedding:
        :param mod_embedding:
        :return:
        """
        str_embedding = torch.nn.functional.normalize(str_embedding, p=2, dim=-1, eps=1e-5)  # [num_ent, emb_dim]
        mod_embedding = torch.nn.functional.normalize(mod_embedding, p=2, dim=-1, eps=1e-5)  # [num_ent, emb_dim]
        bs, _ = str_embedding.size()
        neg_sample_id = torch.randint(0, bs, [bs, self.neg_num])  # [num_ent, num_sample]
        neg_str_feat = str_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        neg_mod_feat = mod_embedding[neg_sample_id]  # [num_ent, num_sample, emb_dim]
        str_samples = torch.cat([str_embedding.unsqueeze(1), neg_str_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        mod_samples = torch.cat([mod_embedding.unsqueeze(1), neg_mod_feat], 1)  # [num_ent, 1+num_sample, emb_dim]
        s2m_score = torch.matmul(mod_samples, str_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
        m2s_score = torch.matmul(str_samples, mod_embedding.unsqueeze(2)).squeeze(
            2) / self.temperature  # [num_ent, 1+num_sample]
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
        loss = loss_1 + loss_2
        return loss


"""
    模态融合
"""


class Attention_Fusion(nn.Module):
    """
        注意力机制融合
    """

    def __init__(self, dim, num_heads=1):
        super(Attention_Fusion, self).__init__()
        self.num_heads = num_heads
        self.scale = torch.Tensor([1. / (dim ** 0.5)]).cuda()
        self.context_vec = nn.Parameter(torch.randn((num_heads, dim)))
        self.act = nn.Softmax(dim=1)
        self.out_proj = nn.Linear(num_heads * dim, dim)

    def forward(self, x):
        """
        :param x: [batch_size, num_modality, dim]
        :return: [batch_size, dim]
        """
        head_outputs = []
        for h in range(self.num_heads):
            context_vec = self.context_vec[h:h + 1]  # context_vec:[1, dim]
            att_weights = torch.sum(context_vec * x * self.scale, dim=-1, keepdim=True)  # [batch_size, num_modality, 1]
            att_weights = self.act(att_weights)  # [batch_size, num_modality, 1]
            head_output = torch.sum(att_weights * x, dim=1)  # [batch_size, dim]
            head_outputs.append(head_output)
        multi_head_out = torch.cat(head_outputs, dim=-1)
        if self.num_heads == 1:
            ent_embs = multi_head_out
        else:
            ent_embs = self.out_proj(multi_head_out)
        return ent_embs


class SSM_Fusion(nn.Module):
    """
        状态空间模型融合
    """

    def __init__(self, dim):
        super(SSM_Fusion, self).__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.h = nn.Parameter(torch.zeros(dim))

    def forward(self, x, return_sequence=False):
        B, L, D = x.shape
        h = self.h.unsqueeze(0).expand(B, -1)

        states = []

        for t in range(L):
            xt = x[:, t]
            a, b = self.in_proj(xt).chunk(2, dim=-1)
            a = torch.sigmoid(a)
            b = torch.tanh(b)
            h = a * h + (1 - a) * b
            states.append(h.unsqueeze(1))

        y = torch.cat(states, dim=1)
        y = self.out_proj(y)

        if return_sequence:
            return y
        else:
            return y[:, -1]


"""
    得分函数
"""


class TransE_Decoder(nn.Module):
    """
        TransE 得分函数
    """

    def __init__(self, num_ent, num_rel, margin=1, p_norm=1):
        super(TransE_Decoder, self).__init__()
        self.margin = margin
        self.p_norm = p_norm
        self.num_ent = num_ent
        self.num_rel = num_rel

    def forward(self, emb_ent, emb_rel, triples, mask):
        """
        :param emb_ent: [num_ent, emb_dim]
        :param emb_rel: [num_ent, rel_dim]
        :param triples: [batch_size, 3]
        :param mask: [batch_size, 3]
        :return:
        """
        emb_ent = F.normalize(emb_ent, p=2, dim=-1)
        emb_rel = F.normalize(emb_rel, p=2, dim=-1)

        h = emb_ent[triples[:, 0] - self.num_rel]
        r = emb_rel[triples[:, 1] - self.num_ent]
        t = emb_ent[triples[:, 2] - self.num_rel]

        h_mask, t_mask = mask[:, 0], mask[:, 2]

        query = torch.zeros_like(h)
        if t_mask.any():  # 预测尾实体
            query[t_mask] = h[t_mask] + r[t_mask]
        elif h_mask.any():  # 预测头实体
            query[h_mask] = t[h_mask] - r[h_mask]
        else:
            raise NotImplementedError
        score = torch.matmul(query, emb_ent[:-1].transpose(1, 0))
        return score


class RotatE_Decoder(nn.Module):
    """
        RotatE 得分函数
    """

    def __init__(self, dim, margin=1):
        super(RotatE_Decoder, self).__init__()
        self.margin = margin
        self.dim = dim // 2
        self.dropout = nn.Dropout(0.3)

    def forward(self, emb_ent, emb_rel, triples, mask):
        """
        :param emb_ent: [num_ent, emb_dim]
        :param emb_rel: [num_ent, rel_dim]
        :param triples: [batch_size, 3]
        :param mask: [batch_size, 3]
        :return:
        """
        emb_ent = F.normalize(emb_ent, p=2, dim=-1)
        emb_ent = self.dropout(emb_ent)

        h = emb_ent[triples[:, 0] - self.num_rel]
        r = emb_rel[triples[:, 1] - self.num_ent]
        t = emb_ent[triples[:, 2] - self.num_rel]

        r = F.normalize(r, p=2, dim=-1)

        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)

        r_phase = r[:, :self.dim]
        r_re, r_im = torch.cos(r_phase), torch.sin(r_phase)

        # 尾实体预测
        h_rot_re = h_re * r_re - h_im * r_im
        h_rot_im = h_re * r_im + h_im * r_re
        h_rot = torch.cat([h_rot_re, h_rot_im], dim=-1)
        # 头实体预测
        t_rot_re = t_re * r_re + t_im * r_im
        t_rot_im = t_im * r_re - t_re * r_im
        t_rot = torch.cat([t_rot_re, t_rot_im], dim=-1)

        h_mask, t_mask = mask[:, 0], mask[:, 2]

        query = torch.zeros_like(h)
        if t_mask.any():
            query[t_mask] = h_rot[t_mask]
        if h_mask.any():
            query[h_mask] = t_rot[h_mask]

        scores = torch.cdist(query, emb_ent[:-1], p=2)

        return scores


class RESCAL_Decoder(nn.Module):
    def __init__(self, e_dim, r_dim):
        super(RESCAL_Decoder, self).__init__()
        self.M = nn.Parameter(
            torch.randn(r_dim, e_dim, e_dim)
        )
        nn.init.xavier_uniform_(self.M.data)

        self.bn0 = nn.BatchNorm1d(e_dim)
        self.bn1 = nn.BatchNorm1d(e_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.5)

    def forward(self, ctx_emb, rel_emb, emb_ent):
        """
        :param ctx_emb:
        :param rel_emb:
        :param emb_ent:
        :return:
        """
        x = self.bn0(ctx_emb)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))  # [batch_size, 1, ent_dim]

        r_mat = self.M[rel_emb]  # [batch_size, ent_dim, ent_dim]
        r_mat = self.hidden_drop(r_mat)

        x = torch.bmm(x, r_mat)  # [batch_size, 1, ent_dim]
        x = x.view(-1, x.size(2))  # [batch_size, ent_dim]

        x = self.bn1(x)
        x = self.output_drop(x)

        scores = torch.mm(x, emb_ent[:-1].transpose(1, 0))
        return scores


class ComplEx_Decoder(nn.Module):
    """
    ComplEx 解码器，接口与 TuckER 一致：
    forward(ctx_emb, rel_emb) -> 解码后的 query 向量
    """

    def __init__(self, ent_dim, rel_dim):
        super(ComplEx_Decoder, self).__init__()
        assert ent_dim == rel_dim
        self.dim = ent_dim // 2
        self.input_drop = nn.Dropout(0.3)
        self.output_drop = nn.Dropout(0.4)
        self.bn0 = nn.BatchNorm1d(self.dim)
        self.bn1 = nn.BatchNorm1d(self.dim)

    def forward(self, ctx_emb, rel_emb, emb_ent):
        """
        :param ctx_emb: [batch_size, emb_dim]，实体或上下文向量
        :param rel_emb: [batch_size, emb_dim]，关系向量
        :param emb_ent:
        :return: [num_ent]，解码后的向量
        """
        ctx_emb = F.normalize(ctx_emb, p=2, dim=-1)
        rel_emb = F.normalize(rel_emb, p=2, dim=-1)
        ctx_emb = self.dropout(ctx_emb)

        # 分割实部和虚部
        ctx_re, ctx_im = torch.chunk(ctx_emb, 2, dim=-1)
        r_re, r_im = torch.chunk(rel_emb, 2, dim=-1)

        # ComplEx 解码：h * r（复数乘法）
        out_re = ctx_re * r_re - ctx_im * r_im
        out_im = ctx_re * r_im + ctx_im * r_re

        out = torch.cat([out_re, out_im], dim=-1)

        scores = torch.mm(out, emb_ent[:-1].transpose(1, 0))
        return scores


class TuckER_Decoder(nn.Module):
    """
        TuckER 得分函数
    """

    def __init__(self, e_dim, r_dim):
        super(TuckER_Decoder, self).__init__()
        self.W = nn.Parameter(torch.randn(r_dim, e_dim, e_dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(e_dim)
        self.bn1 = nn.BatchNorm1d(e_dim)
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.5)

    def forward(self, ctx_emb, rel_emb, emb_ent):
        """
        :param ctx_emb: [batch_size, emb_dim]
        :param rel_emb: [batch_size, rel_dim]
        :param emb_ent:
        :return: [batch_size, emb_dim]
        """
        x = self.bn0(ctx_emb)  # [batch_size, emb_dim]
        x = self.input_drop(x)  # [batch_size, emb_dim]
        x = x.view(-1, 1, x.size(1))  # [batch_size, 1, emb_dim]

        r = torch.mm(rel_emb, self.W.view(rel_emb.size(1), -1))  # [batch_size, emb_dim * emb_dim]
        r = r.view(-1, x.size(2), x.size(2))  # [batch_size, emb_dim, emb_dim]
        r = self.hidden_drop(r)  # [batch_size, emb_dim, emb_dim]

        x = torch.bmm(x, r)  # [batch_size, 1, emb_dim]
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.output_drop(x)

        scores = torch.mm(x, emb_ent[:-1].transpose(1, 0))
        return scores
