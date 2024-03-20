#coding utf-8

import json, os
import random
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from torch.optim import AdamW
from data_process import DataLoder, label

import config
from model import LSEMHGCN
import utils



def get_bert_optimizer(model):

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "bert" in n],
            "weight_decay": config.weight_decay,
            "lr": config.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if 'mhatt' in n],
            "weight_decay": 0.0,
            "lr": config.mhatt_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if all([name not in n for name in ['bert', 'mhatt']]) ],
            "weight_decay": 0.0,
            "lr": config.learning_rate
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=config.adam_epsilon)

    return optimizer


def train():
    train_set = DataLoder(config.dataset, 'train')
    dev_set = DataLoder(config.dataset, 'dev')

    model = LSEMHGCN()

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    elif os.path.exists(f"{config.model_dir}bert.pt"):
        model_state_path = f"{config.model_dir}bert.pt"
        model.load_state_dict(torch.load(model_state_path))
        # model.load_state_dict(model_state)

    model.to(config.device)

    optimizer = get_bert_optimizer(model)

    weight = torch.ones(len(label)).to(config.device)
    weight[1:7] = 2
    weight[1] = 3
    weight[4] = 3
    weight[7:10] = 4
    category_count = json.load(open(f"./dataset/{config.dataset}/category_count.json"))
    max_num = None
    for _ in category_count.values():
        max_num = _
        break
    for idx, num in enumerate(category_count.values()):
        weight[10+idx] = 5*(max_num//num)
    
    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(config.epoch):
        train_set.shuffle_data()
        print('Epoch:{}'.format(i))
        for j in trange(train_set.batch_num):

            tokens_set, sen_lens,\
            input_ids, token_type_ids, attention_mask, sm_input,\
            bert_id2tokens, spans_set,\
            tags_set, sc_tags_set, lsm_tags_set,\
            word_pair_position_set, word_pair_deprel_set, word_pair_pos_set, word_pair_synpost_set = train_set[j]
            tags_flatten = tags_set.reshape([-1])
            sc_tags_set = sc_tags_set.reshape([-1])
            lsm_tags_set = lsm_tags_set.reshape([-1])
            predictions = model(tokens_set, sen_lens,\
                                input_ids, token_type_ids, attention_mask, sm_input,\
                                bert_id2tokens, spans_set,\
                                word_pair_position_set, word_pair_deprel_set, word_pair_pos_set, word_pair_synpost_set)
            [biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, word_pair_postag_emb, word_pair_synpost_emb, label_sm,
             pred]= predictions
            l_ba = 0.10 * F.cross_entropy(biaffine_edge.reshape([-1, biaffine_edge.shape[3]]), tags_flatten, ignore_index=-1)
            l_rpd = 0.01 * F.cross_entropy(word_pair_post_emb.reshape([-1, word_pair_post_emb.shape[3]]), sc_tags_set, ignore_index=-1)
            l_dep = 0.01 * F.cross_entropy(word_pair_deprel_emb.reshape([-1, word_pair_deprel_emb.shape[3]]), sc_tags_set, ignore_index=-1)
            l_psc = 0.01 * F.cross_entropy(word_pair_postag_emb.reshape([-1, word_pair_postag_emb.shape[3]]), sc_tags_set, ignore_index=-1)
            l_tbd = 0.05 * F.cross_entropy(word_pair_synpost_emb.reshape([-1, word_pair_synpost_emb.shape[3]]), sc_tags_set, ignore_index=-1)
            l_lsm = 0.05 * F.cross_entropy(label_sm.reshape([-1, label_sm.shape[3]]), lsm_tags_set, ignore_index=-1)
            

            l_p = F.cross_entropy(pred.reshape([-1, pred.shape[3]]), tags_flatten, weight= weight, ignore_index=-1)
            loss = l_ba + l_rpd + l_dep + l_psc + l_tbd + l_p + l_lsm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_f1 = eval(model, dev_set)

        if joint_f1 > best_joint_f1:
            model_path = f"{config.model_dir}bert.pt"
            torch.save(model.state_dict(), model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, 'Quads', best_joint_f1))
    


def eval(model, dataset, FLAG=False):
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_pred_set = []
        all_labels = []
        all_sens_lengths = []

        all_sentences = []


        #dataset.shuffle_data()
        for i in range(dataset.batch_num):
            tokens_set, sen_lens,\
            input_ids, token_type_ids, attention_mask, sm_input,\
            bert_id2tokens, spans_set,\
            tags_set, sc_tags_set, lsm_tags_set,\
            word_pair_position_set, word_pair_deprel_set, word_pair_pos_set, word_pair_synpost_set = dataset[i]
             
            pred = model(tokens_set, sen_lens,\
                        input_ids, token_type_ids, attention_mask, sm_input,\
                        bert_id2tokens, spans_set,\
                        word_pair_position_set, word_pair_deprel_set, word_pair_pos_set, word_pair_synpost_set)[-1]
            
            preds = F.softmax(pred, dim=-1)
            pred = torch.argmax(preds, dim=3)

            all_preds.append(pred)
            all_pred_set.append(preds)
            all_labels.append(tags_set)
            all_sens_lengths.extend(sen_lens)

            all_sentences.extend([" ".join(tokens) for tokens in tokens_set])

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_pred_set = torch.cat(all_pred_set, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

        metric = utils.Metric(all_preds, all_labels, all_sens_lengths, all_sentences, all_pred_set)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))

        print('Quad\t\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if FLAG:
            metric.tagReport()

    model.train()
    return precision, recall, f1


def test():
    print("Evaluation on testset:")
    testset = DataLoder(config.dataset, 'test') 
    model_state_path = f"{config.model_dir}bert.pt"
    model = LSEMHGCN()
    model.load_state_dict(torch.load(model_state_path))
    # model.load_state_dict(model_state)

    model.to(config.device)
    model.eval()

    eval(model, testset, False)


if __name__ == '__main__':
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    if config.mode == 'train_':
        train()
        test()
    else:
        test()
    


