import argparse
import os
import random
import json
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
    path = os.path.join(data_dir, 'entity.json')
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


def Siamese_Preprocess(args):
    data_dir = os.path.join(args.data_dir, args.dataset)

    valid_path, test_path = os.path.join(data_dir, 'valid.txt'), os.path.join(data_dir, 'test.txt')
    valid_triples, test_triples = load_triples(valid_path), load_triples(test_path)

    ent_path, rel_path = os.path.join(data_dir, 'entity2id.txt'), os.path.join(data_dir, 'relation2id.txt')
    id2ent, ent2id = load_ent_or_rel(ent_path)
    id2rel, rel2id = load_ent_or_rel(rel_path)

    with open(os.path.join(data_dir, 'query.json')) as f:
        pass


def make_prompt(input_dict):
    pass


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
    parser.add_argument('--shuffle_candidates', type=bool, default=False, help='shuffle candidates for analyses or not')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = f'{args.data_dir}/{args.dataset}'
    # 1. entity.json/relation.json制作
    # loadEntRelJson(data_dir)
    # 2. prompt构建
    # tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    # tokenizer.pad_token = tokenizer.eos_token
    Siamese_Preprocess(args)
