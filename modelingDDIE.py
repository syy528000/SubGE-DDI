import os
from sklearn.compose import TransformedTargetRegressor
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torch import nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss
from dgl import mean_nodes
from rgcn_model import RGCN
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl, move_batch_to_device_dgl_ddi2
import dgl
from torch.autograd import Variable
from MultiFocalLoss import MultiFocalLoss

class GELU(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh( math.sqrt( 2 / math.pi ) * (x + 0.044715 * torch.pow(x,3))))

class GraphClassifier(nn.Module):
    def __init__(self, args, relation2id): 
        super().__init__()

        self.params = args
        self.relation2id = relation2id
        self.dropout = nn.Dropout(p = 0.3)
        self.relu = nn.ReLU()
        self.train_rels = args.train_rels 
        self.relations = args.num_rels
        self.gnn = RGCN(args)  

        self.mp_layer1 = nn.Linear(args.feat_dim, args.emb_dim) 
        self.mp_layer2 = nn.Linear(args.emb_dim, args.emb_dim)

        if args.add_ht_emb and args.add_sb_emb:
            self.fc_layer = nn.Linear(3 * (1+args.num_gcn_layers) * args.emb_dim, args.graph_dim)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1+args.num_gcn_layers) * args.emb_dim, args.graph_dim)
        else:
            self.fc_layer = nn.Linear(args.num_gcn_layers * args.emb_dim, args.graph_dim)

    def drug_feat(self, emb):
        self.drugfeat = emb

    def forward(self, data):
        g = data
        g.ndata['h'] = self.gnn(g)
        g_out = mean_nodes(g, 'repr')
        
        # head's embedding
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        # tail's embedding
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb and self.params.add_sb_emb:
             g_rep = torch.cat([g_out.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                ], dim=1)
            
        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                                head_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim),
                                tail_embs.view(-1, (1+self.params.num_gcn_layers) * self.params.emb_dim)
                                ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)

        g_rep = self.fc_layer(self.dropout(g_rep))

        return g_rep

def use_transform(vecs, kernel, bias):
    return torch.mm((vecs + bias),kernel)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,args,config):
        super(BertForSequenceClassification,self).__init__(config)
        self.numLabels = config.num_labels
        self.args = args

        self.dropout = nn.Dropout(args.dropout_prob)  

        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(), 
            'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        
        self.activation = activations[args.activation]

        if args.use_cnn:
            self.convList = nn.ModuleList([nn.Conv1d(config.hidden_size + 2 * args.pos_emb_dim, config.hidden_size, w, padding = (w-1)//2) for w in args.conv_window_size])
            self.pos_emb = nn.Embedding(2 * args.max_seq_length, args.pos_emb_dim, padding_idx=0)

        if args.use_sub:
            self.mlp = nn.Linear(config.hidden_size + args.graph_dim, config.hidden_size)
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(config.hidden_size + args.graph_dim, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(config.hidden_size, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)
        else:
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(len(args.conv_window_size)*config.hidden_size, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)

        self.init_weights()

        if args.use_cnn:
            self.pos_emb.weight.data.uniform_(-1e-3, 1e-3)
        
        self.config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=self.numLabels)
        self.config.output_hidden_states = True
        self.Bert = BertModel.from_pretrained(args.model_name_or_path, config=self.config)
        for param in self.Bert.parameters():
            param.requires_grad = True
        self.bert_layer_weights = nn.Parameter(torch.rand(13, 1))

        if args.use_sub:
            self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model) 

        self.use_cnn = args.use_cnn
        self.use_sub = args.use_sub
        self.middle_layer_size = args.middle_layer_size

    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,position_ids=None, head_mask=None,
                relative_dist1=None, relative_dist2=None,
                labels=None,
                subgraph_index=None,
                subgraph=None,):

        outputs = self.Bert(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask)

        pooled_output = outputs[1].to(self.args.device)

        if self.use_cnn:
            relative_dist1 *= attention_mask
            relative_dist2 *= attention_mask
            pos_emb1 = self.pos_emb(relative_dist1)
            pos_emb2 = self.pos_emb(relative_dist2)
            conv_input = torch.cat((outputs[0], pos_emb1, pos_emb2),2)

            conv_outputs = []
            for c in self.convList:
                conv_output = self.activation(c(conv_input.transpose(1,2))) 
                conv_output,_ = torch.max(conv_output,-1) 
                conv_outputs.append(conv_output)

            pooled_output = torch.cat(conv_outputs,1) 
        
        if self.use_sub:
            self.graph_classifier.train()

            g_dgl_pos,r_labels_pos,targets_pos = [],[],[]
            for idx in subgraph_index:
                g_dgl_pos_, r_labels_pos_, targets_pos_ = subgraph[idx]
                g_dgl_pos.append(g_dgl_pos_);r_labels_pos.append(r_labels_pos_);targets_pos.append(targets_pos_)

            g_dgl_pos = dgl.batch(g_dgl_pos).to(self.args.device)
            subgraph_batch= ((g_dgl_pos,r_labels_pos),targets_pos)

            data_pos, r_labels_pos, targets_pos = self.args.move_batch_to_device(subgraph_batch, self.args.device)
            sub_output = self.graph_classifier(data_pos) 

            pooled_output = torch.cat((pooled_output,sub_output), 1) 

        pooled_output = self.dropout(pooled_output)

        if self.middle_layer_size == 0:
            logits = self.classifier(pooled_output)
        else:
            middle_output = self.activation(self.middle_classifier(pooled_output))
            logits = self.classifier(middle_output)

        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            if self.numLabels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1),labels.view(-1)) 
            else:
                loss_fct = MultiFocalLoss(self.numLabels, [0.8, 0.07, 0.08, 0.04, 0.01])
                loss = loss_fct(logits.view(-1,self.numLabels), labels.view(-1)) 

        outputs = (loss,) + outputs

        return outputs # (loss), logits, (hidden_states), (attentions)

    
    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def average_params(self):
        for x in self.parameters():
            x.data /= self.update_cnt

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt
