import os
import time
import argparse
import logging
import random

from datasets import tqdm

import numpy as np

import torch
from torch import nn

from SPARK import Siamese
from dataset import KG
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics, get_topK


@torch.no_grad()
def valid_eval_metric(key="test", val=None):
    model.eval()
    ent_embs, rel_embs, emb_list = model()
    lp_list_rank = []
    for triplet in tqdm(val):
        h, r, t = triplet
        head_score = model.score(ent_embs, rel_embs, torch.tensor(
            [[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[0].detach().cpu().numpy()
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        tail_score = model.score(ent_embs, rel_embs, torch.tensor(
            [[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda())[0].detach().cpu().numpy()
        tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])

        lp_list_rank.append(head_rank)
        lp_list_rank.append(tail_rank)

    lp_list_rank = np.array(lp_list_rank)
    mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
    print(f"Link Prediction on {key} set")
    print(f"MR: {mr}")
    print(f"MRR: {mrr}")
    print(f"Hit10: {hit10}")
    print(f"Hit3: {hit3}")
    print(f"Hit1: {hit1}")


@torch.no_grad()
def valid_eval_metric_topK(key="test", val=None, topK=10):
    model.eval()
    ent_embs, rel_embs, emb_list = model()
    lp_list_rank = []
    for triplet in tqdm(val):
        h, r, t = triplet
        head_score = model.score(ent_embs, rel_embs, torch.tensor(
            [[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[0].detach().cpu().numpy()
        topks, topk_scores = get_topK(head_score, h, kg.filter_dict[(-1, r, t)], topK)
        lp_list_rank.append(topks)

        tail_score = model.score(ent_embs, rel_embs, torch.tensor(
            [[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda())[0].detach().cpu().numpy()
        topks, topk_scores = get_topK(tail_score, t, kg.filter_dict[(h, r, -1)], topK)
        lp_list_rank.append(topks)

    lp_list_rank = np.array(lp_list_rank)
    a = lp_list_rank[:3]
    b = lp_list_rank[-3:]
    return a, b


if __name__ == '__main__':
    torch.cuda.set_device(1)

    OMP_NUM_THREADS = 8
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)
    torch.cuda.empty_cache()

    torch.manual_seed(2025)
    random.seed(2025)
    np.random.seed(2025)

    parser = argparse.ArgumentParser()
    # DB15K
    parser.add_argument('--data', default="DB15K", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--num_epoch', default=3000, type=int)
    parser.add_argument('--valid_epoch', default=10, type=int)
    parser.add_argument('--exp', default='Siamese_align_param')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=4, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--emb_dropout', default=0.9, type=float)
    parser.add_argument('--vis_dropout', default=0.4, type=float)
    parser.add_argument('--txt_dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--max_vis_num', default=3, type=int)
    parser.add_argument('--cont', action='store_true')  # deprecate
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--max_vis_token', default=16, type=int)
    parser.add_argument('--max_txt_token', default=16, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)

    # MKG-W
    # parser.add_argument('--data', default="MKG-W", type=str)
    # parser.add_argument('--lr', default=5e-4, type=float)
    # parser.add_argument('--dim', default=256, type=int)
    # parser.add_argument('--num_epoch', default=3000, type=int)
    # parser.add_argument('--valid_epoch', default=10, type=int)
    # parser.add_argument('--exp', default='Flare')
    # parser.add_argument('--no_write', action='store_true')
    # parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    # parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    # parser.add_argument('--num_layer_dec', default=2, type=int)
    # parser.add_argument('--num_head', default=4, type=int)
    # parser.add_argument('--hidden_dim', default=1024, type=int)
    # parser.add_argument('--dropout', default=0.01, type=float)
    # parser.add_argument('--emb_dropout', default=0.9, type=float)
    # parser.add_argument('--vis_dropout', default=0.4, type=float)
    # parser.add_argument('--txt_dropout', default=0.1, type=float)
    # parser.add_argument('--smoothing', default=0.0, type=float)
    # parser.add_argument('--batch_size', default=2048, type=int)
    # parser.add_argument('--decay', default=0.0, type=float)
    # parser.add_argument('--max_vis_num', default=3, type=int)
    # parser.add_argument('--cont', action='store_true') # deprecate
    # parser.add_argument('--step_size', default=50, type=int)
    # parser.add_argument('--max_vis_token', default=16, type=int)
    # parser.add_argument('--max_txt_token', default=24, type=int)
    # parser.add_argument('--score_function', default="tucker", type=str)
    # parser.add_argument('--mu', default=0, type=float)
    args = parser.parse_args()

    kg = KG(args.data, None, max_vis_len=args.max_vis_num)
    kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=True)
    visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    text_token_index, text_key_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)
    model = Siamese(
        num_ent=kg.num_ent,
        num_rel=kg.num_rel,
        ent_vis_mask=visual_key_mask,
        ent_txt_mask=text_key_mask,
        dim_str=args.dim,
        num_head=args.num_head,
        dim_hid=args.hidden_dim,
        num_layer_enc_ent=args.num_layer_enc_ent,
        num_layer_enc_rel=args.num_layer_enc_rel,
        num_layer_dec=args.num_layer_dec,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        vis_dropout=args.vis_dropout,
        txt_dropout=args.txt_dropout,
        visual_token_index=visual_token_index,
        text_token_index=text_token_index,
        score_function=args.score_function
    ).cuda()
    model.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)
    scheduler.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['scheduler_state_dict'])

    # valid_eval_metric(key="valid", val=kg.valid)
    # valid_eval_metric(key="test", val=kg.test)
    res1 = valid_eval_metric_topK(key="valid", val=kg.valid)
    print(res1)
    res2 = valid_eval_metric_topK(key="test", val=kg.test)
    print(res2)
