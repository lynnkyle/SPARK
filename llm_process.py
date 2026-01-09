import argparse
import os.path
import random
import json
import numpy as np

import torch

from SPARK import Siamese
from dataset import KG
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics, get_rank, get_topK


@torch.no_grad()
def valid_eval_metric(valid_or_test):
    rank_list = []
    ent_embs, rel_embs, emb_list = model()  # [!!!important]不要放在循环内, 导致测试时速度变慢
    for triple in valid_or_test:
        # for triple in tqdm(valid_or_test):
        h, r, t = triple
        head_score = \
            model.score(torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        head_rank = calculate_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        tail_score = \
            model.score(torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda(), ent_embs,
                        rel_embs)[0].detach().cpu().numpy()  # [batch_size, num_entity]
        tail_rank = calculate_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
    rank_list = np.array(rank_list)
    mr, mrr, hit10, hit3, hit1 = metrics(rank_list)
    return mr, mrr, hit10, hit3, hit1


@torch.no_grad()
def save_numpy(args, type, valid_or_test, topK):
    query_list = []
    rank_list = []
    topk_list = []
    topk_score_list = []
    query_embeds = []
    ent_embs, rel_embs, emb_list = model()  # [!!!important]不要放在循环内, 导致测试时速度变慢
    save_dir = f'{args.save_dir}/{type}'
    for triple in valid_or_test:
        # for triple in tqdm(valid_or_test):
        h, r, t = triple
        query_list.append(f'(?, {r}, {t})')
        head_score = \
            model.score(ent_embs, rel_embs,
                        torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[
                0].detach().cpu().numpy()  # [batch_size, num_entity]
        head_rank = get_rank(head_score, h, kg.filter_dict[(-1, r, t)])
        rank_list.append(head_rank)
        topks, topk_scores = get_topK(head_score, h, kg.filter_dict[(-1, r, t)], topK)
        topk_list.append(topks)
        topk_score_list.append(topk_scores)
        query_embeds.append(
            model.query(ent_embs,
                        rel_embs, torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[
                0].detach().cpu().numpy()
        )

        query_list.append(f'({h}, {r}, ?)')
        tail_score = \
            model.score(ent_embs, rel_embs,
                        torch.tensor([[h + kg.num_rel, r + kg.num_ent, kg.num_ent + kg.num_rel]]).cuda())[
                0].detach().cpu().numpy()  # [batch_size, num_entity]
        tail_rank = get_rank(tail_score, t, kg.filter_dict[(h, r, -1)])
        rank_list.append(tail_rank)
        topks, topk_scores = get_topK(tail_score, t, kg.filter_dict[(h, r, -1)], topK)
        topk_list.append(topks)
        topk_score_list.append(topk_scores)
        query_embeds.append(
            model.query(ent_embs,
                        rel_embs, torch.tensor([[kg.num_ent + kg.num_rel, r + kg.num_ent, t + kg.num_rel]]).cuda())[
                0].detach().cpu().numpy()
        )

    rank_list = np.array(rank_list)
    topk_list = np.array(topk_list)
    topk_score_list = np.array(topk_score_list)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'query.json'), 'w') as f:
        json.dump(query_list, f)
    np.save(os.path.join(save_dir, 'ranks.npy'), rank_list)
    np.save(os.path.join(save_dir, 'topks.npy'), topk_list)
    np.save(os.path.join(save_dir, 'topk_scores.npy'), topk_score_list)
    np.save(os.path.join(save_dir, 'entity_embeddings.npy'), ent_embs.cpu().numpy())
    query_embeds = torch.tensor(query_embeds)
    np.save(os.path.join(save_dir, 'query_embeddings.npy'), query_embeds.cpu().numpy())
    return query_list, rank_list, topk_list, topk_score_list, ent_embs, query_embeds


if __name__ == '__main__':
    torch.cuda.set_device(0)

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
    parser.add_argument('--num_epoch', default=5000, type=int)
    parser.add_argument('--valid_epoch', default=10, type=int)
    parser.add_argument('--exp', default='Siamese_neg')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=4, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--emb_dropout', default=0.5, type=float)
    parser.add_argument('--vis_dropout', default=0.4, type=float)
    parser.add_argument('--txt_dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--max_vis_num', default=3, type=int)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--max_vis_token', default=32, type=int)
    parser.add_argument('--max_txt_token', default=32, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)

    parser.add_argument('--save_dir', default='data/DB15K/llm', type=str)
    parser.add_argument('--top_k', default=20, type=int)

    args = parser.parse_args()

    """
        创建数据集
    """
    kg = KG(args.data, None, max_vis_len=args.max_vis_num)
    kg_loader = torch.utils.data.DataLoader(kg, batch_size=args.batch_size, shuffle=True)

    """
        模型要素
    """
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
    # 模型加载
    # param1 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['state_dict']
    model.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 优化器加载
    # param2 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['optimizer']

    optimizer.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['optimizer_state_dict'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # 学习率裁剪器加载
    # param3 = torch.load(f'ckpt/{args.model}/{args.data}/pre_trained.ckpt')['scheduler']
    lr_scheduler.load_state_dict(torch.load(f'ckpt/db15k.ckpt')['scheduler_state_dict'])

    model.eval()
    valid_and_test = kg.valid + kg.test

    for option in [("train", kg.valid), ("test", kg.test)]:
        _, data = option[0], option[1]
        query_list, rank_list, topk_list, topk_score_list, ent_embs, query_embs = save_numpy(args, _, data,
                                                                                             topK=args.top_k)
        print(len(query_list))
        print(len(rank_list))
        print(len(topk_list))
        print(len(topk_score_list))
        print(len(ent_embs))
        print(len(query_embs))
        print("Done!!!")
