import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from datasets import tqdm
from SPARK import Siamese
from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens
from utils import calculate_rank, metrics


class KG(Dataset):
    def __init__(self, data, logger, max_vis_len=-1):
        self.data = data
        self.logger = logger
        self.dir = f"data/{data}/"
        self.ent2id = {}
        self.id2ent = []
        self.rel2id = {}
        self.id2rel = []
        with open(self.dir + "entities.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.ent2id[line.strip()] = idx
                self.id2ent.append(line.strip())
        self.num_ent = len(self.ent2id)

        with open(self.dir + "relations.txt") as f:
            for idx, line in enumerate(f.readlines()):
                self.rel2id[line.strip()] = idx
                self.id2rel.append(line.strip())
        self.num_rel = len(self.rel2id)

        self.train = []
        with open(self.dir + "train.txt") as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.train.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.valid = []
        with open(self.dir + "valid.txt") as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.valid.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.test = []
        with open(self.dir + "test.txt") as f:
            for line in f.readlines():
                h, r, t = line.strip().split("\t")
                self.test.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))

        self.filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h, r, t = triplet
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        self.max_vis_len_ent = max_vis_len
        self.max_vis_len_rel = max_vis_len
        # self.gather_vis_feature()
        # self.gather_txt_feature()

        # 长尾实体频率统计
        self.entity_frequencies = self.calculate_entity_frequencies()
        # 长尾实体测试集
        if self.data == "DB15K":
            frequency_threshold = 5
        elif self.data == "MKG-W":
            frequency_threshold = 2
        self.low_freq_test_triples = self.split_dataset_by_frequency(frequency_threshold)

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        h, r, t = self.train[idx]
        if random.random() < 0.5:
            masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
            label = h
            all_possible_labels = self.filter_dict.get((-1, r, t), [])
        else:
            masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
            label = t
            all_possible_labels = self.filter_dict.get((h, r, -1), [])
        # 构造负采样filter
        filter_mask = torch.zeros((self.num_ent,), dtype=torch.bool)
        if len(all_possible_labels) > 0:
            filter_mask[all_possible_labels] = True
        return torch.tensor(masked_triplet), torch.tensor(label), torch.tensor(filter_mask)

    def calculate_entity_frequencies(self):
        """
        计算每个实体在训练数据中的出现频率，并返回按频率分组的字典。
        """
        entity_count = {i: 0 for i in range(self.num_ent)}  # 初始化计数器

        # 统计每个实体在三元组中的出现次数
        for h, r, t in self.train:
            entity_count[h] += 1  # 头实体
            entity_count[t] += 1  # 尾实体

        # 创建频率分组的字典
        frequency_groups = {}
        for ent, count in entity_count.items():
            if count not in frequency_groups:
                frequency_groups[count] = []
            frequency_groups[count].append(ent)

        # 记录并返回频率分组字典
        print(f"Total entities: {self.num_ent}")
        print(f"Entity frequency distribution:")
        for freq, ents in frequency_groups.items():
            print(f"Frequency {freq}: {len(ents)} entities")

        max_freq = max(frequency_groups.keys())
        min_freq = min(frequency_groups.keys())
        print("max frequency:", max_freq, "entity:", frequency_groups[max_freq])
        print("min frequency:", min_freq, "entity:", frequency_groups[min_freq])

        return frequency_groups

    def split_dataset_by_frequency(self, frequency_threshold):
        """
        根据实体出现频率划分数据集（训练集/验证集/测试集均可）。

        低频实体定义：出现次数 > 1 且 <= frequency_threshold

        :param frequency_threshold: 低频阈值
        :return: 包含低频实体的三元组列表
        """
        low_freq_entities = set()

        # 筛选低频实体，出现次数 > 1 且 <= threshold
        for freq, ents in self.entity_frequencies.items():
            if 1 < freq <= frequency_threshold:
                low_freq_entities.update(ents)

        # 遍历测试集，提取包含低频实体的三元组
        low_freq_test_triples = []
        for h, r, t in self.test:
            if h in low_freq_entities or t in low_freq_entities:
                low_freq_test_triples.append((h, r, t))

        print(f"Low-frequency triples (>1 and <= {frequency_threshold}): {len(low_freq_test_triples)}")
        return low_freq_test_triples


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DB15K
    parser.add_argument('--data', default="MKG-W", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=256, type=int)
    parser.add_argument('--num_epoch', default=3000, type=int)
    parser.add_argument('--valid_epoch', default=10, type=int)
    parser.add_argument('--log_epoch', default=100, type=int)
    parser.add_argument('--exp', default='SPARK_DB15K_LONG_TAIL')
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
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--max_vis_token', default=16, type=int)
    parser.add_argument('--max_txt_token', default=32, type=int)
    parser.add_argument('--neg_num', default=3, type=int)
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--fusion_function', default="ssm", type=str)
    parser.add_argument('--score_function', default="tucker", type=str)
    parser.add_argument('--loss_modality', default="0.5", type=float)
    parser.add_argument('--loss_entity', default="15", type=float)
    args = parser.parse_args()

    kg = KG('MKG-W', None)
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
        fusion_function=args.fusion_function,
        score_function=args.score_function
    ).cuda()
    model.load_state_dict(torch.load(f'/mnt/data1/zhz/SPARK/ckpt/SPARK_MKGW_16_32_k3/MKG-W/lr_0.0005num_epoch_3000num_head_4hidden_dim_10240.90.40.1batch_size_2048max_vis_token_16max_txt_token_32tucker_2960.ckpt')['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer.load_state_dict(torch.load(f'/mnt/data1/zhz/SPARK/ckpt/SPARK_MKGW_16_32_k3/MKG-W/lr_0.0005num_epoch_3000num_head_4hidden_dim_10240.90.40.1batch_size_2048max_vis_token_16max_txt_token_32tucker_2960.ckpt')['optimizer_state_dict'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)
    scheduler.load_state_dict(torch.load(f'/mnt/data1/zhz/SPARK/ckpt/SPARK_MKGW_16_32_k3/MKG-W/lr_0.0005num_epoch_3000num_head_4hidden_dim_10240.90.40.1batch_size_2048max_vis_token_16max_txt_token_32tucker_2960.ckpt')['scheduler_state_dict'])

    valid_eval_metric(val=kg.low_freq_test_triples)
    valid_eval_metric(val=kg.test)
