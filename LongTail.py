from collections import defaultdict

import torch
from torch.utils.data import Dataset
import random

import numpy as np
import matplotlib.pyplot as plt


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

        # Long tail entity frequency statistics
        self.entity_frequencies = self.calculate_entity_frequencies()

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

    def get_entities_by_frequency(self, frequency):
        """
        获取具有特定频率的所有实体。
        """
        return self.entity_frequencies.get(frequency, [])


def plot_long_tail_distribution(entity_frequencies, data="DB15K", head_ratio=0.7, max_freq=50):
    """
    横轴：实体出现频率 f（高频在左，低频在右）
    纵轴：出现频率为 f 的实体数量
    """
    # 合并频率 > max_freq
    freq_filter = defaultdict(int)
    for f, ents in entity_frequencies.items():
        freq_to_use = f if f <= max_freq else max_freq
        freq_filter[freq_to_use] += len(ents)

    # 排序，频率从高到低
    freqs = sorted(freq_filter.keys(), reverse=True)
    counts = [freq_filter[f] for f in freqs]

    # 使用索引绘图
    barWidth = 2
    barGap = 2

    x = np.arange(len(freqs)) * (barWidth + barGap)
    plt.figure(figsize=(10, 5))
    plt.bar(x, counts, width=barWidth, color='skyblue')

    # Head / Long-tail 分割
    cumulative = np.cumsum(counts)
    total = cumulative[-1]
    split_idx = np.searchsorted(cumulative, total * head_ratio)
    plt.axvline(split_idx * (barWidth + barGap), linestyle='--', color='red')

    # 在红线两侧标注
    split_x = split_idx * (barWidth + barGap)
    head_x = split_x / 2  # 红线左侧中点
    tail_x = split_x + (x[-1] - split_x) / 2  # 红线右侧中点
    plt.text(head_x, max(counts) * 0.8, "Head", color="red", ha='center', fontsize=12)
    plt.text(tail_x, max(counts) * 0.8, "Long Tail", color="red", ha='center', fontsize=12)

    # 横轴显示频率，间隔控制
    max_labels = 15  # 最多显示15个刻度
    step = max(1, len(freqs) // max_labels)  # 自动计算步长

    xtick_labels = [str(f) if f < max_freq else f">={max_freq}" for f in freqs]
    plt.xticks(x[::step], xtick_labels[::step], rotation=45)

    plt.xlabel("Entity Frequency")
    plt.ylabel("Number of Entities")
    plt.title(f"Long-tail distribution of entity frequencies in {data}")
    plt.tight_layout()
    plt.show()

    return freqs[split_idx]


if __name__ == '__main__':
    kg = KG('DB15K', None, max_vis_len=3)
    print(kg)
    frequency_groups = kg.entity_frequencies
    print(f"频率分组的实体数量: {len(frequency_groups)}")
    split_idx = plot_long_tail_distribution(kg.entity_frequencies)
    print("split_idx===>", split_idx)  # LongTailTrail的划分依据

    kg = KG('MKG-W', None, max_vis_len=3)
    print(kg)
    frequency_groups = kg.entity_frequencies
    print(f"频率分组的实体数量: {len(frequency_groups)}")
    split_idx = plot_long_tail_distribution(kg.entity_frequencies)
    print("split_idx===>", split_idx) # LongTailTrail的划分依据
