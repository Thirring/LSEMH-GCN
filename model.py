from typing import Any
import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

import config




class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(994, config.class_num)    
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
    def forward(self, gcn_outputs,  label_sm, weight_prob, tensor_masks):
                
        node = gcn_outputs

        batch, seq, dim = node.shape

        node1 = node.unsqueeze(1).expand(batch, seq, seq, dim)
        node2 = node1.permute(0, 2, 1, 3).contiguous()
        double_node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(weight_prob, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, -1)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        
        
        s_triu = torch.zeros_like(label_sm)
        s_triu[:, :, :, 0] = label_sm[:, :, :, 0]
        s_triu[:, :, :, 1:4] = torch.triu(label_sm[:, :, :, 1:4].permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
        s_triu[:, :, :, 4:] = torch.tril(label_sm[:, :, :, 4:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()

        pre = self.W(torch.cat([double_node, edge_i, edge_j, weight_prob, s_triu], dim=-1))

        return pre


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = nn.LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)

        self.dropout_output = torch.nn.Dropout(config.emb_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, weight_prob_softmax, gcn_inputs, self_loop):


        batch, seq, dim = gcn_inputs.shape

        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)
        weight_prob_softmax += self_loop
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)

        
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        gcn_outputs = self.dropout_output(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        
        return node_outputs


class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(config.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(config.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine

class LSEMHGCN(torch.nn.Module):
    def __init__(self):
        super(LSEMHGCN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_path)
        self.dropout_output = torch.nn.Dropout(config.emb_dropout)

        self.post_emb = torch.nn.Embedding(config.post_size, config.embed_class_num, padding_idx=0)
        self.deprel_emb = torch.nn.Embedding(config.deprel_size, config.embed_class_num, padding_idx=0)
        self.postag_emb  = torch.nn.Embedding(config.pos_pair_size, config.embed_class_num, padding_idx=0)
        self.synpost_emb = torch.nn.Embedding(config.syn_post_size, config.embed_class_num, padding_idx=0)

        
        self.quad_biaffine = Biaffine(config.bert_feature_dim, config.bert_feature_dim, config.class_num, bias=(True, True))
        
        self.label_biaffine = Biaffine(config.bert_feature_dim, config.bert_feature_dim*2, 1, bias=(True, True))
        self.ap_fc = nn.Linear(config.bert_feature_dim, config.bert_feature_dim)
        self.op_fc = nn.Linear(config.bert_feature_dim, config.bert_feature_dim)

        self.aspect_imply = nn.Linear(config.bert_feature_dim, config.bert_feature_dim)
        self.opinion_imply = nn.Linear(config.bert_feature_dim, config.bert_feature_dim)

        self.dense = nn.Linear(config.bert_feature_dim*2, 600)

        self.dense_gcn = nn.Linear(config.bert_feature_dim, config.gcn_dim)


        self.gcn_layers = GraphConvLayer(config.device, config.gcn_dim, config.class_num+4*config.embed_class_num+17, config.class_num, config.pooling, )


        self.highway = RefiningStrategy(config.gcn_dim, 5*config.embed_class_num, config.embed_class_num, dropout_ratio=0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.ap_fc.weight)
        nn.init.xavier_uniform_(self.op_fc.weight)
        nn.init.xavier_uniform_(self.aspect_imply.weight)
        nn.init.xavier_uniform_(self.opinion_imply.weight)
        nn.init.xavier_uniform_(self.dense.weight)
    def forward(self, tokens_set, sen_lens,\
                    input_ids, token_type_ids, attention_mask, sm_input,\
                    bert_id2tokens, spans_set,\
                    word_pair_position_set, word_pair_deprel_set, word_pair_pos_set, word_pair_synpost_set):
        

        outputs = self.bert(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask = attention_mask).last_hidden_state
        model_feature = torch.zeros(config.batch_size, config.max_len, 768).to(config.device)
        model_feature_woimply = torch.zeros(config.batch_size, config.max_len-2, 768).to(config.device)
        bert_id2tokens = F.one_hot(bert_id2tokens).type(torch.float32)
        bert_id_mask = attention_mask[:, 2:].unsqueeze(-1).expand(-1, -1, bert_id2tokens.size(-1))
        bert_id2tokens *= bert_id_mask
        bert_id2tokens = bert_id2tokens.permute(0, 2, 1).contiguous()
        num = bert_id2tokens.sum(dim = -1)
        num = num.masked_fill_(num == 0, 1).unsqueeze(-1)
        
        model_feature_woimply[:, :bert_id2tokens.size(1), :] = torch.bmm(bert_id2tokens, outputs[:, 1:-1, :])/num
        model_feature[:, 1:bert_id2tokens.size(1)+1, :] = torch.bmm(bert_id2tokens, outputs[:, 1:-1, :])/num
        
        CLS = outputs[:, 0, :]
        A_imply = self.aspect_imply(CLS)
        O_imply = self.opinion_imply(CLS)
        model_feature[:, 0, :] = A_imply
        sen_last_idex = sen_lens + 1
        model_feature[range(config.batch_size), sen_last_idex, :] = O_imply
        idx = torch.arange(config.max_len).unsqueeze(0).to(config.device)  
        sen_mask = torch.lt(idx, (sen_lens+2).unsqueeze(1))

        _ = sen_mask.unsqueeze(1).expand(-1, config.max_len, -1)
        mask = _ * _.permute(0, 2, 1).contiguous()
        tensor_masks = mask.unsqueeze(-1)   
        word_pair_post_emb = self.post_emb(word_pair_position_set)
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel_set)
        word_pair_postag_emb = self.postag_emb(word_pair_pos_set)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost_set)

        ap_node = F.relu(self.ap_fc(model_feature))
        op_node = F.relu(self.op_fc(model_feature))
        biaffine_edge = self.quad_biaffine(ap_node, op_node)
        weight_prob_list = [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb, word_pair_synpost_emb]
        weight_prob = torch.cat(weight_prob_list, dim=-1)


        word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1)*tensor_masks
        word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1)*tensor_masks
        word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1)*tensor_masks 
        word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1)*tensor_masks
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1)*tensor_masks

        weight_softmax_list = [biaffine_edge_softmax, word_pair_post_emb_softmax, word_pair_deprel_emb_softmax, word_pair_postag_emb_softmax, word_pair_synpost_emb_softmax]
        weight_prob_softmax = torch.cat(weight_softmax_list, dim=-1)

        batch, seq, dim = model_feature.size()

        node1 = model_feature.unsqueeze(1).expand(batch, seq, seq, dim)
        node2 = node1.permute(0, 2, 1, 3).contiguous()
        double_node = torch.cat([node1, node2], dim=-1)
        double_node.view(batch, seq*seq, dim*2)
        double_node = double_node.view(batch, seq*seq, dim*2)

        label_feature = self.bert(**sm_input).last_hidden_state[:, 0, :]
        
        label_sm = self.label_biaffine(label_feature.unsqueeze(0).expand(batch, -1, -1), double_node).squeeze(-1)
        label_sm = label_sm.view(batch, seq, seq, 17)

        gcn_input = F.relu(self.dense_gcn(model_feature))

        label_sm_softmax = F.softmax(label_sm, dim=-1)
        
        weight_prob_softmax = torch.cat([weight_prob_softmax, label_sm_softmax], dim=-1)

        self_loop = []
        for _ in range(config.batch_size):
            self_loop.append(torch.eye(config.max_len))
        self_loop = torch.stack(self_loop).to(config.device).unsqueeze(1).expand(config.batch_size, config.class_num+4*config.embed_class_num+17, config.max_len, config.max_len) * tensor_masks.permute(0, 3, 1, 2).contiguous()
        
        gcn_outputs = self.gcn_layers(weight_prob_softmax, gcn_input, self_loop)  
       


        weight_prob_list.append(label_sm)

        pre_prob = self.highway(gcn_outputs, label_sm, weight_prob, tensor_masks)
        weight_prob_list.append(pre_prob)
        return weight_prob_list

    



