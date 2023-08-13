import argparse
from asyncio.log import logger
import os
# import ptvsd
import torch
import logging
import numpy as np
import random
import copy
# import lmdb
# import pickle
from load_and_cache_example import *

from typing import Union, Optional, List, Dict 
from warnings import simplefilter
from torch.utils.data import(DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from processor_ddie import ddie_processors as processors
from transformers import glue_output_modes as output_modes
from torch.utils.data.distributed import DistributedSampler
from transformers import ( BertConfig,
                                #BertForSequenceClassification,
                                BertTokenizer,
                                RobertaConfig,
                                RobertaForSequenceClassification,
                                RobertaTokenizer,
                                XLMConfig, XLMForSequenceClassification,
                                XLMTokenizer, XLNetConfig,
                                XLNetForSequenceClassification,
                                XLNetTokenizer,
                                DistilBertConfig,
                                DistilBertForSequenceClassification,
                                DistilBertTokenizer)
from transformers.tokenization_utils_base import PaddingStrategy
from transformers import AdamW, get_linear_schedule_with_warmup

from metricsDDIE import ddie_compute_metrics as compute_metrics
# from radam import AdamW
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from modelingDDIE import BertForSequenceClassification
import json
from tqdm import tqdm,trange

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets, MyTensorDataset
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_batch_to_device_dgl_ddi2
from sklearn.model_selection import StratifiedKFold,KFold

simplefilter(action='ignore', category=FutureWarning)

"""tensorboard's save"""
path = os.path.dirname(__file__)
path = path[:path.rfind("\\")]
path = os.path.join(path,"tensorboardSave")
if not os.path.exists(path):
    os.mkdir(path)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

def set_seed(args):
    random.seed(args.seed) 
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  
    # torch.set_grad_enabled(False)

def train(args, train_dataloader, dev_dataloader, graph_dataset, model, tokenizer, kfold):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(path) 

    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        # len(dataloader) = batch
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #warmup gradient
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.parameter_averaging:
        storage_model = copy.deepcopy(model)
        storage_model.zero_init_params()
    else:
        storage_model = None
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad() 


    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    set_seed(args) 
    epoch_losses = []
    for epoch, _ in enumerate(train_iterator, start=1):
        epoch_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])#如果args.local_rank not in [-1,0]，即是分布式计算时，返回False，停用tqdm
        for step, batch in enumerate(epoch_iterator):
            # batch
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'relative_dist1': batch[3],
                    'relative_dist2': batch[4],
                    'labels':         batch[5],
                    'subgraph_index': batch[6],
                    'subgraph':       graph_dataset,}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()

            epoch_loss += loss.detach().item()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 and not args.tpu:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step() 
                if not args.parameter_averaging:
                    scheduler.step()  
                model.zero_grad()
                global_step += 1 

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        #results = evaluate(args, model, tokenizer)
                        results,microF_ = evaluate(args, model, tokenizer, prefix=str(global_step), dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss 

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
        
        epoch_loss /= (step+1)
        print("##########Epoch{}, loss:{:.4f}##########".format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

        if epoch % 10 ==0:
             results,microF_ = evaluate(args, model, tokenizer, prefix="Epoch:{}".format(epoch), dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_during_training:
            prefix = '{}'.format(kfold) + 'epoch' + str(epoch)
            output_dir = os.path.join(args.output_dir, prefix)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if args.parameter_averaging:
                storage_model.average_params()
                result,microF_ = evaluate(args, storage_model, tokenizer, prefix=prefix, dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)
                storage_model.restore_params()
            else:
                results,microF_ = evaluate(args, model, tokenizer, prefix=prefix, dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    #return global_step, tr_loss / global_step
    return global_step, tr_loss / global_step, storage_model, epoch_losses

def evaluate(args, model, tokenizer, prefix="", dev_dataloader=None, graph_dataset=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    set_seed(args)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)
        
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(dev_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'relative_dist1': batch[3],
                        'relative_dist2': batch[4],
                        'labels':         batch[5],
                        'subgraph_index': batch[6],
                        'subgraph':       graph_dataset,}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy() 
                out_label_ids = inputs['labels'].detach().cpu().numpy() 
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        if not os.path.exists(os.path.join(args.output_dir, str(prefix))):
            os.makedirs(os.path.join(args.output_dir, str(prefix)))
        np.save(os.path.join(args.output_dir, str(prefix), 'preds'), preds)
        np.save(os.path.join(args.output_dir, str(prefix), 'labels'), out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1) # [[2,1,0,4,1,0,0,0,0,....],[0,0,1,0,0,0,2,...],[0,3,1,1,0,0,0,0,...],...]
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        output_eval_dir = os.path.join(eval_output_dir, str(prefix))
        if not os.path.exists(output_eval_dir):
            os.mkdir(output_eval_dir)
        result = compute_metrics(eval_task, preds, out_label_ids, every_type=False, output_dir=output_eval_dir)
        results.update(result)
        output_eval_file = os.path.join(output_eval_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** 评估结果 {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, result["microF"]

def main():
    parser = argparse.ArgumentParser()

    ## essential
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## optional
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")
    parser.add_argument("--use_cross_fold", action='store_true',
                        help="Whether to use cross fold")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--parameter_averaging', action='store_true', help="Whether to use parameter averaging")
    parser.add_argument("--dropout_prob", default=.0, type=float, help="Dropout probability")
    parser.add_argument('--middle_layer_size', type=int, default=0, help="Dimention of middle layer")

    ## CNN
    parser.add_argument('--use_cnn', action='store_true', help="Whether to use CNN")
    parser.add_argument('--conv_window_size', type=int, nargs='+', default=[3], help="List of convolution window size")
    parser.add_argument('--pos_emb_dim', type=int, default=10, help="Dimention of position embeddings.")
    parser.add_argument('--activation', type=str, default='relu', help="Activation function")

    ## SubAGCN
    parser.add_argument('--use_sub', action='store_true', help='Whether to use sub graph features')
    parser.add_argument('--work_dir', default=None, type=str, help="The address of the project's location")
    parser.add_argument('--graph_dim', '-g_dim', type=int, default=50, help='The dim of the subgraph features')
    parser.add_argument('--experiment_name', '-e', type=str, default='default1', help='A folder with this name would be created to dump saved models and log files')
    parser.add_argument('--load_model', action='store_true',help='Load existing model?')
    parser.add_argument('--total_file', type=str, default='total', help='Name of file containing total triplets')
    parser.add_argument('--train_file', '-tf', type=str, default='train', help='Name of file containing training triplets')
    parser.add_argument('--valid_file', '-vf', type=str, default='dev', help='Name of file containing validation triplets')
    parser.add_argument('--test_file', '-ttf', type=str, default='test', help='Name of file containing validation triplets')
    parser.add_argument('--dataset', '-d', type=str, help="Dataset String")
    parser.add_argument('--max_links', type=int, default=250000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument('--feat', '-f', type=str, default='morgan', help='the type of the feature we use in molecule modeling')
    parser.add_argument('--feat_dim', type=int, default=2048, help='the dimension of the feature')
    parser.add_argument('--emb_dim', "-dim", type=int, default=50, help="Entity embedding size")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='Whether to append adj matrix list with symmetric relations')
    parser.add_argument('--num_neg_samples_per_link', '-neg', type=int, default=0,help="Number of negative examples to sample per positive link")
    parser.add_argument('--use_kge_embeddings', "-kge", type=bool, default=False, help='Whether to use pretrained KGE embeddings')
    parser.add_argument('--kge_model', type=str, default='TransE', help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument('--hop', type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0, help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument('--max_nodes_per_hop', '-max_h', type=list, default=[20,20,20], help='if > 0, upper bound the # nodes per hop by subsampling')
    parser.add_argument('--num_bases', '-b', type=int, default=4, help='Number of basis functions to use for GCN weights')
    parser.add_argument('--attn_rel_emb_dim', '-ar_dim', type=int, default=32, help='Relation embedding size for attention')
    parser.add_argument('--num_gcn_layers', '-l', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in GNN layers')
    parser.add_argument('--edge_dropout', type=float, default=0.4, help='Dropout rate in edges of the subgraphs')
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum', help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True, help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--add_sb_emb', '-sb', type=bool, default=True, help='whether to concatenate subgraph embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True, help='whether to have attn in model or not')
    parser.add_argument('--has_kg', '-kg', type=bool, default=True, help='whether to have kg in model or not')
    parser.add_argument('--add_transe_emb', type=bool, default=True, help='whether to have knowledge graph embedding in model or not')
    parser.add_argument('--gamma', type=float, default=0.2, help='The threshold for attention')

    parser.add_argument('--pretrained_dir', type=str, help="The path to pre-trained model dir")
    parser.add_argument('--freeze_pretrained_parameters', action='store_true', help="Whether to freeze parameters pretrained on database")

    ## Focal Loss
    parser.add_argument('--num_labels',type=int,help="The number of labels")
    parser.add_argument('--gamma_loss', type=int, default=2, help='The gamma of Focal Loss')

    ## kfold
    parser.add_argument('--kfold', '-k', type=int, default=1, help='The fold of the cross validation')

    args = parser.parse_args()
    initialize_experiment(args, __file__) 

    # subgraph's file_paths
    args.file_paths = {
    'total': os.path.join(args.work_dir, 'data/{}/{}.txt'.format(args.dataset, args.total_file)), 
    }

    args.move_batch_to_device = move_batch_to_device_dgl if args.dataset == 'drugbank' else move_batch_to_device_dgl_ddi2

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("The folder {} already has an output file. Please use overwrite_output_dir if you want to overwrite the output".format(args.output_dir))
    
    args.db_path = os.path.join(args.work_dir, f'data/{args.dataset}/subgraphs_en_{args.enclosing_sub_graph}_neg_{args.num_neg_samples_per_link}_hop_{args.hop}')
    if not os.path.isdir(args.db_path):
        generate_subgraph_datasets(args,splits=['total']) #

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    set_seed(args)

    args.task_name = args.task_name.lower() 
    if args.task_name not in processors:
        raise ValueError("task not found:%s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name] 
    labelList = processor.get_labels()  # ddie:['negative', 'mechanism', 'effect', 'advise', 'int']  pretraining:["'negative', 'positive'"]
    args.num_labels = len(labelList)


    if args.local_rank not in [-1,0]:
        torch.distributed.barried() 

    args.model_type = args.model_type.lower()
    configClass, modelClass, tokenizerClass = MODEL_CLASSES[args.model_type]
    config = configClass.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels = args.num_labels, finetuning_task = args.task_name)
    tokenizer = tokenizerClass.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case = args.do_lower_case)
    tokenizer._bos_token = "[BOS]"
    tokenizer._eos_token = "[EOS]"
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>","</e1>","<e2>","</e2>"]
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    
    dataset,graph_dataset,test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate = False, pos="total_pos", neg="total_neg")
    model = modelClass(args, config)

    if not args.use_cross_fold:
        """train&test"""

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(dataset)
        test_sampler = SequentialSampler(test_dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

        if not args.do_train:
            global_step = 0
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'state_dict')))
        if args.local_rank == 0:
            torch.distributed.barrier()

        model.to(args.device)

        if args.do_train:
            global_step, tr_loss, storage_model, epoch_losses = train(args, train_dataloader, test_dataloader, graph_dataset, model, tokenizer, "train")

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            import matplotlib.pyplot as plt
            plt.switch_backend("Agg")

            plt.figure()
            plt.plot(epoch_losses,"b",label="loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir,"loss_curve.jpg"))
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'state_dict'))

        results = {}
        if args.do_test and args.local_rank in [-1, 0]:
            if args.parameter_averaging:
                storage_model.average_params()
                result, _ = evaluate(args, storage_model, tokenizer, prefix = "test", dev_dataloader=test_dataloader, graph_dataset=graph_dataset)
            else:
                result, _ = evaluate(args, model, tokenizer, prefix = "test", dev_dataloader=test_dataloader, graph_dataset=graph_dataset)

            precision = float(result['Precision'])
            recall = float(result['Recall'])
            microF = float(result['microF'])
            microAUC = float(result['micro_auc'])
            macroAUC = float(result['macro_auc'])
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

            output_test_file = os.path.join(args.output_dir,"train&test_test_results.txt")
            with open(output_test_file, "w") as writer:
                writer.write("precision = %s\n" % (str(precision)))
                writer.write("recall = %s\n" % (str(recall)))
                writer.write("microF = %s\n" % (str(microF)))
                writer.write("microAUC = %s\n" % (str(microAUC)))
                writer.write("microAUC = %s\n" % (str(macroAUC)))

    else:
        """k-fold"""
        from utils.KFold_utils import KFold_group
        from torch.utils.data.dataset import Subset

        output_dir = os.path.join(args.output_dir, 'best_model')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        kf = KFold_group(n_splits=args.kfold, shuffle=False) 

        Precisions = []
        Recalls = []
        microFs = []
        microAUCs = []
        macroAUCs = []
        i = 0
        best_model_score = 0
        for train_index, dev_index in kf.split(X=dataset,att_mask=dataset.get(1)):
            i += 1
            train_fold = Subset(dataset, train_index)
            dev_fold = Subset(dataset, dev_index)

            train_sampler = RandomSampler(train_fold)
            dev_sampler = SequentialSampler(dev_fold)

            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            train_dataloader = DataLoader(train_fold, sampler=train_sampler, batch_size=args.train_batch_size)
            dev_dataloader = DataLoader(dev_fold, sampler=dev_sampler, batch_size=args.eval_batch_size)

            model.to(args.device)
            # Training
            if args.do_train:
                logger.info("train/eval params: %s", args)

                global_step, tr_loss, storage_model, epoch_losses = train(args, train_dataloader, dev_dataloader, graph_dataset, model, tokenizer, i)
            
            if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and not args.tpu:
                if not os.path.exists(args.output_dir) and args.local_rank in [-1,0]:
                    os.makedirs(args.output_dir)
                
                logger.info("save model to%s", args.output_dir)
                torch.save(model.state_dict(), os.path.join(args.output_dir, "state_dict"))

            # Evaluation
            results={}
            if args.do_eval and args.local_rank in [-1,0]:
                if args.parameter_averaging:
                    storage_model.average_params()
                    result, microF_ = evaluate(args, storage_model, tokenizer, prefix = i, dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)
                else:
                    result, microF_ = evaluate(args, model, tokenizer, prefix = i, dev_dataloader=dev_dataloader, graph_dataset=graph_dataset)
                Precisions.append(float(result['Precision']))
                Recalls.append(float(result['Recall']))
                microFs.append(float(result['microF']))
                microAUCs.append(float(result['micro_auc']))
                macroAUCs.append(float(result['macro_auc']))
                result = dict((k + "_{}".format(global_step),v) for k,v in result.items())
                results.update(result)

                if microF_ > best_model_score:
                    best_model_score = microF_
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(model.state_dict(), os.path.join(output_dir, "state_dict"))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        
        """results"""
        precision = np.mean(Precisions)
        precision_std = np.std(Precisions)
        recall = np.mean(Recalls)
        recall_std = np.std(Recalls)
        microF = np.mean(microFs)
        microF_std = np.std(microFs)
        microAUC = np.mean(microAUCs)
        microAUC_std = np.std(microAUCs)
        macroAUC = np.mean(macroAUCs)
        macroAUC_std = np.std(macroAUCs)
        output_total_file = os.path.join(args.output_dir,"fivefold_results.txt")
        with open(output_total_file, "w") as writer:
            writer.write("precision = %s\tstd = %s\n" % (str(precision), str(precision_std)))
            writer.write("recall = %s\tstd = %s\n" % (str(recall), str(recall_std)))
            writer.write("microF = %s\tstd = %s\n" % (str(microF), str(microF_std)))
            writer.write("microAUC = %s\tstd = %s\n" % (str(microAUC), str(microAUC_std)))
            writer.write("microAUC = %s\tstd = %s\n" % (str(macroAUC), str(macroAUC_std)))

        if args.do_test:
            test_sampler = SequentialSampler(test_dataset)
            args.test_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.test_batch_size)

            global_step = 0
            model.load_state_dict(torch.load(os.path.join(output_dir,"state_dict")))
            model.to(args.device)
            test_result, _ = evaluate(args, model, tokenizer, prefix = "test", dev_dataloader=test_dataloader, graph_dataset=graph_dataset)

            precision = float(test_result['Precision'])
            recall = float(test_result['Recall'])
            microF = float(test_result['microF'])
            microAUC = float(test_result['micro_auc'])
            macroAUC = float(test_result['macro_auc'])

            output_test_file = os.path.join(args.output_dir,"test_results.txt")
            with open(output_test_file, "w") as writer:
                writer.write("precision = %s\n" % (str(precision)))
                writer.write("recall = %s\n" % (str(recall)))
                writer.write("microF = %s\n" % (str(microF)))
                writer.write("microAUC = %s\n" % (str(microAUC)))
                writer.write("microAUC = %s\n" % (str(macroAUC)))

    return results

if __name__ == "__main__":
    main()