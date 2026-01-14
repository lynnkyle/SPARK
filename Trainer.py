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
from utils import calculate_rank, metrics

torch.cuda.set_device(1)

OMP_NUM_THREADS = 8
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(2025)
random.seed(2025)
np.random.seed(2025)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

if __name__ == '__main__':
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

    file_format = ""
    for arg_name in vars(args).keys():
        if arg_name in ["lr", "hidden_dim", "batch_size", "num_epoch", "max_vis_token", "max_txt_token", "num_head",
                        "mu"]:
            file_format += f"{arg_name}_{vars(args)[arg_name]}"
        elif arg_name in ["score_function", "emb_dropout", "vis_dropout", "txt_dropout"]:
            file_format += f"{vars(args)[arg_name]}"

    if not args.no_write:
        os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./logs/{args.exp}/{args.data}", exist_ok=True)
        if not os.path.isfile(f"ckpt/{args.exp}/args.txt"):
            with open(f"ckpt/{args.exp}/args.txt", "w") as f:
                for arg_name in vars(args).keys():
                    if arg_name not in ["data", "exp", "no_write", "num_epoch", "cont", "early_stop"]:
                        f.write(f"{arg_name}\t{type(vars(args)[arg_name])}\n")
    else:
        file_format = None
    file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.data}/{file_format}.log")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.info(f"{os.getpid()}")
    logger.info(args)

    kg = KG(args.data, logger, max_vis_len=args.max_vis_num)
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

    cross_entropy_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    # TODO
    margin_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=0.1)
    # TODO

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)

    last_epoch = 0
    start = time.time()
    logger.info("EPOCH\tLOSS\tTOTAL TIME")
    all_ents = torch.arange(kg.num_ent).cuda()
    all_rels = torch.arange(kg.num_rel).cuda()

    best_mrr = 0.0
    best_result = None
    checkpoint_path = ''

    for epoch in range(last_epoch + 1, args.num_epoch + 1):
        total_loss = 0.0
        for batch, label, filter_mask in kg_loader:
            ent_embs, rel_embs, emb_list = model()
            scores = model.score(ent_embs, rel_embs, batch.cuda())
            loss = cross_entropy_loss_fn(scores, label.cuda())
            # TODO
            align_loss = model.align(emb_list)
            loss += align_loss * 0.01
            # TODO
            pos_logit, neg_logit = model.pos_neg_logits_vectorized_topk(scores, label.cuda(), filter_mask.cuda(),
                                                                        neg_num=3)
            pos_expand = pos_logit.unsqueeze(1).expand_as(neg_logit)  # [B, n]
            target = torch.ones_like(neg_logit)
            res = margin_ranking_loss_fn(pos_expand.reshape(-1), neg_logit.reshape(-1), target.reshape(-1))
            loss += 15 * res
            # TODO
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        scheduler.step()

        elapsed_seconds = time.time() - start
        elapsed_hours = int(elapsed_seconds // 3600)
        elapsed_minutes = int((elapsed_seconds % 3600) // 60)
        elapsed_seconds = int(elapsed_seconds % 60)

        logger.info(f"{epoch} \t {total_loss:.6f} \t {elapsed_hours}h-{elapsed_minutes}m-{elapsed_seconds}s")
        if (epoch) % args.valid_epoch == 0:
            model.eval()
            with torch.no_grad():
                ent_embs, rel_embs, emb_list = model()
                save_dir = 'embeddings/'
                os.makedirs(save_dir, exist_ok=True)
                # np.save(os.path.join(save_dir, f'ent_embeddings_epoch_{epoch}.npy'), ent_embs.cpu().detach().numpy())
                lp_list_rank = []
                for triplet in tqdm(kg.valid):
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
                logger.info("Link Prediction on Validation Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

                lp_list_rank = []
                for triplet in tqdm(kg.test):
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
                logger.info("Link Prediction on Test Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

            if best_mrr < mrr:
                best_mrr = mrr
                best_result = (mr, mrr, hit10, hit3, hit1)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    logger.info(f"Deleted previous checkpoint: {checkpoint_path}")
                checkpoint_path = f"./ckpt/{args.exp}/{args.data}/{file_format}_{epoch}.ckpt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)
            model.train()

    logger.info("Done! {}. The best results are shown below:".format(args.data))
    logger.info(best_result)
