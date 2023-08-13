import torch
import os
import numpy as np
import json

def read_entity_from_id(filename='./data/WN18RR/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data/WN18RR/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    relation2id_num = {}
    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data, rel_num = [], [], [], []
    unique_entities = set()
    for line in lines:
        e1, e2, relation = parse_line(line)
        e1, e2, relation = int(e1),int(e2),int(relation)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (e1, relation, e2))

        if relation not in relation2id_num:
            relation2id_num[relation] = 0
        else:
            relation2id_num[relation] += 1

        if not directed:
                # Connecting source and tail entity
            rows.append(e1)
            cols.append(e2)
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation)
                rel_num.append(relation2id_num[relation])

        # Connecting tail and source entity
        rows.append(e2)
        cols.append(e1)
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation)
            rel_num.append(relation2id_num[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data, rel_num), list(unique_entities), relation2id_num



def build_data(path='', is_unweigted=False, directed=True):
    entity2id = {}
    with open(path+'entity2id.txt', 'r') as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
        for line in lines:
            entity2id[line[0]] = line[1]
    with open(path+'relation2id.json', 'rb') as f:
        relation2id = json.load(f)

    # entity2id = read_entity_from_id(path + 'entity2id.txt')
    # relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    total_triples, total_adjacency_mat, unique_entities_total, relation2id_num = load_data(os.path.join(
        path, 'total.txt'), entity2id, relation2id, is_unweigted, directed)


    return (total_triples, total_adjacency_mat)
