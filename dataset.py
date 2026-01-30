import torch
from torch.utils.data import Dataset
import random
import os
from tqdm import tqdm


class KG(Dataset):
    def __init__(self, data, logger):
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
        self.train_filter_dict = {}

        for data_split in [self.train, self.valid, self.test]:
            for triplet in data_split:
                h, r, t = triplet
                if (-1, r, t) not in self.filter_dict:
                    self.filter_dict[(-1, r, t)] = []
                self.filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.filter_dict:
                    self.filter_dict[(h, r, -1)] = []
                self.filter_dict[(h, r, -1)].append(t)

        for data_split in [self.train]:
            for triplet in data_split:
                h, r, t = triplet
                if (-1, r, t) not in self.train_filter_dict:
                    self.train_filter_dict[(-1, r, t)] = []
                self.train_filter_dict[(-1, r, t)].append(h)
                if (h, r, -1) not in self.train_filter_dict:
                    self.train_filter_dict[(h, r, -1)] = []
                self.train_filter_dict[(h, r, -1)].append(t)

        if data == "DB15K":
            self.train_filter_dict = self.filter_dict

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        h, r, t = self.train[idx]
        if random.random() < 0.5:
            masked_triplet = [self.num_ent + self.num_rel, r + self.num_ent, t + self.num_rel]
            label = h
            all_possible_labels = self.train_filter_dict.get((-1, r, t), [])
        else:
            masked_triplet = [h + self.num_rel, r + self.num_ent, self.num_ent + self.num_rel]
            label = t
            all_possible_labels = self.train_filter_dict.get((h, r, -1), [])
        # 构造负采样filter
        filter_mask = torch.zeros((self.num_ent,), dtype=torch.bool)
        if len(all_possible_labels) > 0:
            filter_mask[all_possible_labels] = True
        return torch.tensor(masked_triplet), torch.tensor(label), torch.tensor(filter_mask)


if __name__ == '__main__':
    kg = KG('MKG-W', None)
    print(kg)
