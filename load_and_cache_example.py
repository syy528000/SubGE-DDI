import os
import torch
from processor_ddie import DDIProcessor,DescProcessor
import logging
import csv
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from torch.utils.data import TensorDataset
from transformers import glue_output_modes as output_modes
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Union, Optional, List, Dict 
from tqdm import tqdm
from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets, MyTensorDataset

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    dep_mask0: Optional[List[int]] = None
    dep_mask1: Optional[List[int]] = None
    dep_mask2: Optional[List[int]] = None
    dep_mask3: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

def convert_examples_to_features(examples,tokenizer,max_length,label_list=None):
    if max_length is None:
        max_length = tokenizer.model_max_length

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (i, example) in tqdm(enumerate(examples), desc="Writing Examples"):
        tokens_a = tokenizer.tokenize(example.text_a)
        entity_one_start = tokens_a.index("<e1>")
        entity_one_end = tokens_a.index("</e1>")
        entity_two_start = tokens_a.index("<e2>")
        entity_two_end = tokens_a.index("</e2>")
        """token->$#"""
        tokens_a[entity_one_start] = "$"
        tokens_a[entity_one_end] = "$"
        tokens_a[entity_two_start] = "#"
        tokens_a[entity_two_end] = "#"
        special_tokens_count = 2
        if len(tokens_a) > max_length - special_tokens_count:
            tokens_a = tokens_a[:(max_length - special_tokens_count)]
        tokens = tokens_a
        tokens += ['[SEP]']
        tokens = ['[CLS]'] + tokens
        token_type_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        """mask"""
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_length)
        
        label_id = int(label_map[example.label])

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      label=label_id, ))

    return features

def load_and_cache_examples(args, task, tokenizer, evaluate = False, pos = "", neg = ""):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier() 
    processor = DDIProcessor()
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(
                            "total", 
                            list(filter(None, args.model_name_or_path.split("/"))).pop(),
                            str(args.max_seq_length),
                            str(task))
                            )
    if args.do_test:
        test_cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(
                                    "test", 
                                    list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                    str(args.max_seq_length),
                                    str(task))
                                    )


    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = torch.load(cached_features_file)
        if args.do_test:
            test_features = torch.load(test_cached_features_file)
    else:
        labelList = processor.get_labels() # mode = "ddie"：['negative', 'mechanism', 'effect', 'advise', 'int']
                                            # mode = "pretraining"：['negative', 'positive']
        if task in ["mnli","mnli-mm"] and args.model_type in ["roberta"]:
            labelList[1],labelList[2] = labelList[2],labelList[1]
        examples = processor.get_train_examples(args.data_dir)

        if args.do_test:
            """test"""
            test_examples = processor.get_test_examples(args.data_dir)
            # [InputExample1,InputExample2,InputExample3,...]
            # InputExample.guid = "train-0"
            # InputExample.label = "negative"
            # InputExample.text_a = "Based on adult data,..."
            # InputExample.text_b = None
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=labelList,
                                                max_length=args.max_seq_length,
                                                )
        if args.do_test:
            """test"""
            test_labelList = processor.get_labels() # mode = "ddie"：['negative', 'mechanism', 'effect', 'advise', 'int']
                                                    # mode = "pretraining"：['negative', 'positive']

        if task in ["mnli","mnli-mm"] and args.model_type in ["roberta"]:
            test_labelList[1],test_labelList[2] = test_labelList[2],test_labelList[1]

        if args.do_test:
            test_features = convert_examples_to_features(test_examples,
                                                    tokenizer,
                                                    label_list = test_labelList,
                                                    max_length=args.max_seq_length,
                                                    )
        if args.local_rank in [-1,0]:
            torch.save(features, cached_features_file)
            if args.do_test:
                torch.save(test_features, test_cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # position index
    drugID = tokenizer.vocab["drug"]
    oneID = tokenizer.vocab["##1"]
    twoID = tokenizer.vocab["##2"]

    all_input_ids = [f.input_ids for f in features]
    all_entity1_pos = []
    all_entity2_pos = []
    for input_ids in all_input_ids:
        entity1_pos = args.max_seq_length - 1
        entity2_pos = args.max_seq_length - 1
        for i in range(args.max_seq_length):
            if input_ids[i] == drugID and input_ids[i+1] == oneID:
                entity1_pos = i
            if input_ids[i] == drugID and input_ids[i+1] == twoID:
                entity2_pos = i
        all_entity1_pos.append(entity1_pos) 
        all_entity2_pos.append(entity2_pos)
    assert len(all_input_ids) == len(all_entity1_pos) == len(all_entity2_pos)

    if args.do_test:
        """test"""
        test_all_input_ids = [f.input_ids for f in test_features]
        test_all_entity1_pos = []
        test_all_entity2_pos = []
        for input_ids in test_all_input_ids:
            entity1_pos = args.max_seq_length - 1
            entity2_pos = args.max_seq_length - 1
            for i in range(args.max_seq_length):
                if input_ids[i] == drugID and input_ids[i+1] == oneID:
                    entity1_pos = i
                if input_ids[i] == drugID and input_ids[i+1] == twoID:
                    entity2_pos = i
            test_all_entity1_pos.append(entity1_pos) 
            test_all_entity2_pos.append(entity2_pos)
        assert len(test_all_input_ids) == len(test_all_entity1_pos) == len(test_all_entity2_pos)

    range_list = list(range(args.max_seq_length, 2*args.max_seq_length))
    all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
    all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

    if args.do_test:
        """test"""
        test_all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in test_all_entity1_pos], dtype=torch.long)
        test_all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in test_all_entity2_pos], dtype=torch.long)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    if args.do_test:
        """test"""
        test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_all_attention_mask = torch.tensor([f.attention_mask for f in test_features], dtype=torch.long)
        test_all_token_type_ids = torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long)

        test_all_labels = torch.tensor([f.label for f in test_features], dtype=torch.long)

    g_dataset = None
    if args.use_sub:
        g_dataset = SubgraphDataset(args.db_path, pos, neg, args.file_paths,
                                    add_traspose_rels=args.add_traspose_rels,
                                    num_neg_samples_per_link=args.num_neg_samples_per_link,
                                    use_kge_embeddings=args.use_kge_embeddings, dataset=args.dataset,
                                    kge_model=args.kge_model,
                                    args=args,)

        args.num_rels = g_dataset.num_rels
        args.aug_num_rels = g_dataset.aug_num_rels
        args.inp_dim = g_dataset.n_feat_dim
        args.train_rels = 200 if args.dataset == 'BioSNAP' else args.num_rels
        args.num_nodes = len(g_dataset.id2entity)

        args.max_label_value = g_dataset.max_n_label
        logging.info(f"Device: {args.device}")
        logging.info(f"Input dim : {args.inp_dim}, # Relations : {args.num_rels}, # Augmented relations : {args.aug_num_rels}")

        if args.do_test:
            subgraphs_indices = torch.tensor(list(range(g_dataset.num_graphs_pos))[:len(features)], dtype=torch.long)
            test_subgraphs_indices = torch.tensor(list(range(g_dataset.num_graphs_pos))[len(features):], dtype=torch.long)
        else:
            subgraphs_indices = torch.tensor(list(range(g_dataset.num_graphs_pos)), dtype=torch.long)    

    else:
        # fake indices
        if args.do_test:
            subgraphs_indices = torch.tensor(list(range(len(features))), dtype=torch.long)

            test_subgraphs_indices = torch.tensor(list(range(len(test_features))), dtype=torch.long)  
        else:
            subgraphs_indices = torch.tensor(list(range(len(features))), dtype=torch.long)

    dataset = MyTensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_relative_dist1, all_relative_dist2,
                            all_labels,
                            subgraphs_indices,)
    if args.do_test:
        test_dataset = MyTensorDataset(test_all_input_ids, test_all_attention_mask, test_all_token_type_ids,
                                test_all_relative_dist1, test_all_relative_dist2,
                                test_all_labels,
                                test_subgraphs_indices,)
    else:
        test_dataset = None
    return dataset, g_dataset, test_dataset