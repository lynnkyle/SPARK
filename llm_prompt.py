import argparse
import os
import random
import json
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import networkx as nx
import numpy as np
from transformers import AutoTokenizer
from SPARQLWrapper import SPARQLWrapper, JSON
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_entity(data_dir):
    path = os.path.join(data_dir, 'entity2id.txt')
    ent2id = {}
    id2ent = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        ent_num = lines[0]
        for line in lines[1:]:
            if not line:
                continue
            url, _ = line.strip().split(' ')
            ent2id[url] = int(_)
            id2ent.append(url)
    return ent_num, ent2id, id2ent


def load_relation(data_dir):
    path = os.path.join(data_dir, 'relation2id.txt')
    rel2id = {}
    id2rel = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        rel_num = lines[0]
        for line in lines[1:]:
            if not line:
                continue
            if line is None or line == '\n':
                continue
            url, _ = line.strip().split(' ')
            rel2id[url] = int(_)
            id2rel.append(url)
    return rel_num, rel2id, id2rel


def load_triples(file):
    triples = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            triples.append((int(h), int(r), int(t)))
    return triples


def load_ent_or_rel(file):
    id2x = {}
    x2id = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        num = int(lines[0].strip())
        for line in lines[1:]:
            ent, idx = line.strip().split(' ')
            id2x[int(idx)] = ent
            x2id[ent] = int(idx)
        assert num == len(id2x)
    return id2x, x2id


def load_text(data_dir, tokenizer, max_seq_len):
    def truncate_text(ent2text, tokenizer, max_len=50):
        ents, texts = [], []
        for k, v in ent2text.items():
            ents.append(k)
            texts.append(v)

        encoded = tokenizer(
            texts, add_special_tokens=False, padding=True, truncation=True, max_length=max_len, return_tensors='pt',
            return_token_type_ids=False, return_attention_mask=False
        )

        input_ids = encoded['input_ids']
        truncated_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        assert len(ents) == len(truncated_texts)
        return {ent: truncated_texts[idx] for idx, ent in enumerate(ents)}

    ent_num, ent2id, id2ent = load_entity(data_dir)
    rel_num, rel2id, id2rel = load_relation(data_dir)
    ent2text = json.load(open(os.path.join(data_dir, 'entity.json'), 'r', encoding='utf-8'))
    ent2name = {uri: ent2text[uri]['name'] for uri in id2ent}
    ent2desc = {uri: ent2text[uri]['desc'] for uri in id2ent}
    ent2desc = truncate_text(ent2desc, tokenizer, max_seq_len)
    rel2text = json.load(open(os.path.join(data_dir, 'relation.json'), 'r', encoding='utf-8'))
    rel2name = {uri: rel2text[uri]['name'] for uri in id2rel}

    return ent2name, ent2desc, rel2name


def save_triple2text(data_dir, in_file, out_file):
    ent_num, ent2id, id2ent = load_entity(data_dir)
    rel_num, rel2id, id2rel = load_relation(data_dir)

    def save_triple(read_path, write_path):
        with open(read_path, 'r', encoding='utf-8') as f1, open(write_path, 'w', encoding='utf-8') as f2:
            for line in f1.readlines():
                h, r, t = line.strip().split('\t')
                f2.write(f'{id2ent[int(h)]}\t{id2rel[int(r)]}\t{id2ent[int(t)]}\n')

    print("num_ent:", ent_num)
    save_triple(in_file, out_file)


def get_dbpedia_ent_abstract_sparql(ent, lang='en'):
    endpoint_url = 'http://dbpedia.org/sparql'
    sparql = SPARQLWrapper(endpoint_url)

    full_uri = f"http://dbpedia.org/resource/{ent}"

    # 使用完整 URI，避免语法错误
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?abstract WHERE {{
        <{full_uri}> dbo:abstract ?abstract .
        FILTER (lang(?abstract) = '{lang}')
    }}
    """

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results.get('results', {}).get('bindings', [])
        if bindings:
            return bindings[0]['abstract']['value']
        else:
            return ""
    except Exception as e:
        print(f"SPARQL query error: {e}")
        return ""


def save_ent2desc(data_dir):
    ent_num, ent2id, id2ent = load_entity(data_dir)
    res_dict = {}

    def process(ent):
        ent_name = ent.strip('<>').split('/')[-1]
        desc = get_dbpedia_ent_abstract_sparql(ent_name, lang='en')
        return ent, {'name': ent_name, 'desc': desc}

    # 开启线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process, ent): ent for ent in id2ent}
        for future in as_completed(futures):
            ent, ent_dict = future.result()
            res_dict[ent] = ent_dict

    path = os.path.join(data_dir, 'entity.json')
    # 写入 JSON 文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)


def save_rel2desc(data_dir):
    path = os.path.join(data_dir, 'relation.json')
    rel_num, rel2id, id2rel = load_relation(data_dir)
    res_dict = {}
    for rel in id2rel:
        rel_name = rel.strip('<>').split('/')[-1]
        rel_dict = {'name': rel_name, 'desc': ''}
        res_dict[rel] = rel_dict
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)


def loadEntRelJson(data_dir):
    in_files = ['train.txt', 'valid.txt', 'test.txt']
    out_files = ['train2text.txt', 'valid2text.txt', 'test2text.txt']
    for file_in, file_out in zip(in_files, out_files):
        in_file = os.path.join(data_dir, file_in)
        out_file = os.path.join(data_dir, file_out)
        save_triple2text(data_dir, in_file, out_file)
        save_ent2desc(data_dir)
        save_rel2desc(data_dir)
    print("Done1!!!")


# TODO 采样方式
class RelationOccurrence(object):
    def __init__(self, data_dir):
        self.triples = load_triples(os.path.join(data_dir, 'train.txt'))
        self.rel = self.get_relations()
        self.one_hop_triples, self.one_hop_relations = self.get_one_hop_triples()
        self.rel_occurrences = self.count_rel_occurrences()

    def get_relations(self):
        rel = set()
        for h, r, t in self.triples:
            rel.add(r)
        return rel

    def get_one_hop_triples(self):
        """
            获取one_hop三元组的信息
            one_hop_triples: {1: {(1, 'likes', 2)}, 2: {(1, 'likes', 2), (2, 'knows', 3)}, 3: {(2, 'knows', 3)}}
            one_hop_relations: {1: {('likes', 0)}, 2: {('likes', 1), ('knows', 0)}, 3: {('knows', 1)}}
            :return:
        """
        one_hop_triples = defaultdict(set)
        one_hop_relations = defaultdict(set)
        for h, r, t in self.triples:
            one_hop_triples[h].add((h, r, t))
            one_hop_triples[t].add((h, r, t))
            one_hop_relations[h].add((r, 0))
            one_hop_relations[t].add((r, 1))
        return one_hop_triples, one_hop_relations

    def count_rel_occurrences(self):
        """
            计算成对关系的共现次数
            :return:
        """
        rel_occurrences = defaultdict(int)
        for entity, one_hop_triple in self.one_hop_triples.items():
            for h, r, t in one_hop_triple:
                for r_sample, direction in self.one_hop_relations[entity]:
                    if r_sample == r:
                        continue
                    elif entity == h:
                        rel_occurrences[(r, 0), (r_sample, direction)] += 1
                    else:
                        rel_occurrences[(r, 1), (r_sample, direction)] += 1
        return rel_occurrences


class KnowledgeGraph(object):
    def __init__(self, args, tokenizer):
        self.args = args

        # Ent、Rel Information
        data_dir = os.path.join(args.data_dir, args.dataset)
        self.ent2name, self.ent2desc, self.rel2name = load_text(data_dir, tokenizer, args.max_seq_len)
        self.id2ent, self.ent2id = load_ent_or_rel(os.path.join(data_dir, 'entity2id.txt'))
        self.id2rel, self.rel2id = load_ent_or_rel(os.path.join(data_dir, 'relation2id.txt'))

        # Triples Information
        self.train_triples = load_triples(os.path.join(data_dir, 'train.txt'))
        self.valid_triples = load_triples(os.path.join(data_dir, 'valid.txt'))
        self.test_triples = load_triples(os.path.join(data_dir, 'test.txt'))

        # All Entity AND All Relation
        triples = self.train_triples
        self.ent_list = sorted(
            list(set([h for h, _, _ in triples] + [t for _, _, t in triples])))  # 实体映射id集合
        self.rel_list = sorted(list(set([r for _, r, _ in
                                         triples])))  # 关系映射id集合
        print(f'train entity num: {len(self.ent_list)}; train relation num: {len(self.rel_list)}')

        # Graph Base On Train_Triples
        self.graph = nx.MultiDiGraph()
        for h, r, t in self.train_triples:
            self.graph.add_edge(h, t, relation=r)
        print(self.graph)

        # Sample Method
        # TODO
        self.sample_method = RelationOccurrence(data_dir=data_dir)
        # TODO

    def neighbors_condition(self, ent, rel, direct):
        out_edges = []
        score_out = []
        for h, t, attr_dict in self.graph.out_edges(ent, data=True):
            assert ent == h
            out_edges.append((h, attr_dict['relation'], t))
            score_out.append(self.sample_method.rel_occurrences[((rel, direct), (attr_dict['relation'], 0))])
        out_sorted_indices_desc = np.argsort(score_out)[::-1]

        in_edges = []
        score_in = []
        for h, t, attr_dict in self.graph.in_edges(ent, data=True):
            assert ent == t
            in_edges.append((h, attr_dict['relation'], t))
            score_in.append(self.sample_method.rel_occurrences[((rel, direct), (attr_dict['relation'], 1))])
        in_sorted_indices_desc = np.argsort(score_in)[::-1]

        if self.args.neighbor_num <= len(out_edges):
            return [out_edges[out_sorted_indices_desc[i]] for i in range(self.args.neighbor_num)]
        elif self.args.neighbor_num <= len(out_edges + in_edges):
            return out_edges + [in_edges[in_sorted_indices_desc[i]] for i in
                                range(self.args.neighbor_num - len(out_edges))]
        else:
            edges = out_edges + in_edges
            random.shuffle(edges)
            return edges

    def neighbors(self, ent):
        out_edges = []
        for h, t, attr_dict in self.graph.out_edges(ent, data=True):
            assert ent == h
            out_edges.append((h, attr_dict['relation'], t))

        in_edges = []
        for h, t, attr_dict in self.graph.in_edges(ent, data=True):
            assert ent == t
            in_edges.append((h, attr_dict['relation'], t))

        edges = out_edges + in_edges
        random.shuffle(edges)
        return edges


def Siamese_Preprocess(args, graph):
    data_dir = os.path.join(args.data_dir, args.dataset)

    valid_path, test_path = os.path.join(data_dir, 'valid.txt'), os.path.join(data_dir, 'test.txt')
    valid_triples, test_triples = load_triples(valid_path), load_triples(test_path)

    assert valid_triples == graph.valid_triples
    assert test_triples == graph.test_triples

    ent_path, rel_path = os.path.join(data_dir, 'entity2id.txt'), os.path.join(data_dir, 'relation2id.txt')
    id2ent, ent2id = load_ent_or_rel(ent_path)
    id2rel, rel2id = load_ent_or_rel(rel_path)

    assert ent2id == graph.ent2id
    assert rel2id == graph.rel2id

    file_data_dict = [('train', valid_triples), ('test', test_triples)]
    train_output = []
    test_output = []
    for _, data in file_data_dict:
        file_dir = os.path.join(data_dir, 'llm', _)
        with open(os.path.join(file_dir, 'query.json'), encoding='utf-8') as f:
            query = json.load(f)
        ranks = np.load(os.path.join(file_dir, "ranks.npy"))
        topks = np.load(os.path.join(file_dir, "topks.npy"))
        topks_scores = np.load(os.path.join(file_dir, 'topk_scores.npy'))

        data_list = []
        for idx, (h_idx, r_idx, t_idx) in enumerate(data):
            head_query = query[2 * idx]
            head_rank = int(ranks[2 * idx])
            head_topk = [id2ent[e_idx] for e_idx in topks[2 * idx].tolist()][:args.topk]
            head_topk_scores = [score for score in topks_scores[2 * idx].tolist()[:args.topk]]
            head_topk_names = [graph.ent2name[ent] for ent in head_topk]
            head_entity_ids = [graph.ent2id[ent] for ent in head_topk]

            tail_query = query[2 * idx + 1]
            tail_rank = int(ranks[2 * idx + 1])
            tail_topk = [id2ent[e_idx] for e_idx in topks[2 * idx + 1].tolist()[:args.topk]]
            tail_topk_scores = [score for score in topks_scores[2 * idx + 1].tolist()[:args.topk]]
            tail_topk_names = [graph.ent2name[ent] for ent in tail_topk]
            tail_entity_ids = [graph.ent2id[ent] for ent in tail_topk]
            head_prediction = {
                'query_id': 2 * idx,
                'query': head_query,
                'triple': (id2ent[h_idx], id2rel[r_idx], id2ent[t_idx]),
                'triple2id': (h_idx, r_idx, t_idx),
                'rank': head_rank,
                'topk_ents': head_topk,
                'topk_names': head_topk_names,
                'topk_scores': head_topk_scores,
                'entity_ids': head_entity_ids
            }
            data_list.append(head_prediction)

            tail_prediction = {
                'query_id': 2 * idx + 1,
                'query': tail_query,
                'triple': (id2ent[h_idx], id2rel[r_idx], id2ent[t_idx]),
                'triple2id': (h_idx, r_idx, t_idx),
                'rank': tail_rank,
                'topk_ents': tail_topk,
                'topk_names': tail_topk_names,
                'topk_scores': tail_topk_scores,
                'entity_ids': tail_entity_ids
            }
            data_list.append(tail_prediction)
        if _ == 'train':
            train_output = list(data_list)
        elif _ == 'test':
            test_output = list(data_list)
        else:
            raise NotImplementedError
    return train_output, test_output


def make_prompt(input_dict, graph):
    """
    :param input_dict:  是一个字典, {triplet, inverse, topk_ents, topk_names, topk_scores, rank, query_id, entity_ids}
    :param graph:
    :return:
    """
    ent2name, ent2desc, rel2name = graph.ent2name, graph.ent2desc, graph.rel2name

    idx = input_dict['query_id']
    h, r, t = input_dict['triple']
    h_name, h_desc = ent2name[h], ent2desc[h]
    r_name = rel2name[r]
    t_name, t_desc = ent2name[t], ent2desc[t]

    choices = input_dict['topk_ents']
    input_dict['choices'] = choices
    if args.add_special_tokens:
        try:
            choices = [ent_name + ' [ENTITY]' for ent_name in choices]
        except:
            print(input_dict)
            print(choices)
            exit(0)
    choices = '[' + '; '.join(choices) + ']'

    if idx % 2 == 1:
        if args.add_special_tokens:
            prompt = f'Here is a triplet with tail entity t unknown: ({h_name}, {r_name}, t [QUERY]).\n\n'
        else:
            prompt = f'Here is a triplet with tail entity t unknown: ({h_name}, {r_name}, t).\n\n'
        if args.add_entity_desc:
            prompt += f'Following are some details about {h_name}:\n{h_desc}\n\n'
        if args.add_neighbors:
            if args.condition_neighbors:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in
                             graph.neighbors_condition(h, r, 0)]
            else:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in
                             graph.neighbors(h)]
            neighbors = '[' + '; '.join([f'({e1}, {r1}, {e2})' for e1, r1, e2 in neighbors]) + ']'
            prompt += f'Following are some triples about {h_name}:\n{neighbors}\n\n'
        prompt += f'What is the entity name of t? Select one from the list: {choices}\n\n[Answer]: '

        input_dict['input'] = prompt
        input_dict['output'] = t_name
    else:
        if args.add_special_tokens:
            prompt = f'Here is a triplet with head entity h unknown: (h [QUERY], {r_name}, {t_name}).\n\n'
        else:
            prompt = f'Here is a triplet with head entity h unknown: (h, {r_name}, {t_name}).\n\n'
        if args.add_entity_desc:
            prompt += f'Following are some details about {t_name}:\n{t_desc}\n\n'
        if args.add_neighbors:
            if args.condition_neighbors:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in
                             graph.neighbors_condition(t, r, 1)]
            else:
                neighbors = [(ent2name[e1], rel2name[r1], ent2name[e2]) for e1, r1, e2 in
                             graph.neighbors(t)]
            neighbors = '[' + '; '.join([f'({e1}, {r1}, {e2})' for e1, r1, e2 in neighbors]) + ']'
            prompt += f'Following are some triples about {t_name}:\n{neighbors}\n\n'
        prompt += f'What is the entity name of t? Select one from the list: {choices}\n\n[Answer]: '

        input_dict['input'] = prompt
        input_dict['output'] = h_name

    return input_dict


def make_dataset_mp(data, graph, output_file):
    with Pool(20) as p:
        data = p.map(partial(make_prompt, graph=graph), data)
    json.dump(data, open(output_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--llm_dir', type=str, default='../Flare/models--TheBloke--Llama-2-7B-fp16',
                        help='choose your llm model')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='DB15K', help='FB15K237 | WN18RR')
    parser.add_argument('--output_dir', type=str, default='data_KGELlama', help='output folder for dataset')
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--topk', type=int, default=10, help='number of candidates')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for truncated sampling')
    parser.add_argument('--kge_model', type=str, default='Siamese', help='TransE | SimKGC | CoLE')
    parser.add_argument('--add_special_tokens', type=bool, default=True, help='add special tokens')
    parser.add_argument('--add_entity_desc', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default=50, help='the max length of FB15K237')
    parser.add_argument('--add_neighbors', type=bool, default=True)
    parser.add_argument('--neighbor_num', type=int, default=10)
    parser.add_argument('--condition_neighbors', type=bool, default=True, help='add condition or not')
    parser.add_argument('--shuffle_candidates', type=bool, default=False,
                        help='shuffle candidates for analyses or not')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = f'{args.data_dir}/{args.dataset}'
    # 1. entity.json/relation.json制作
    # loadEntRelJson(data_dir)
    # 2. prompt构建
    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    graph = KnowledgeGraph(args, tokenizer)
    if args.kge_model == 'Siamese':
        train_data, test_data = Siamese_Preprocess(args, graph)
    else:
        raise NotImplementedError()

    output_dir = os.path.join(data_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_examples = make_dataset_mp(train_data, graph, os.path.join(output_dir, 'train.json'))
    test_examples = make_dataset_mp(test_data, graph, os.path.join(output_dir, 'test.json'))
    print("Done2!!!")
