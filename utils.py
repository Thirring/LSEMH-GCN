import numpy as np
import torch

from data_process import label2id, id2label






def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length + 2):
        if tags[i][i] == ignore_index:
            continue
        label = id2label[tags[i][i]]
        if label == 'B-A':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-A':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, end])
    return spans


def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start, end = -1, -1
    for i in range(length + 2):
        if tags[i][i] == ignore_index:
            continue
        label = id2label[tags[i][i]]
        if label == 'B-O':
            if start != -1:
                spans.append([start, end])
            start, end = i, i
        elif label == 'I-O':
            end = i
        else:
            if start != -1:
                spans.append([start, end])
                start, end = -1, -1
    if start != -1:
        spans.append([start, end])

    return spans


class Metric():
    def __init__(self, predictions, goldens, sen_lengths, all_sentences, all_pred_set):
        self.predictions = predictions
        self.goldens = goldens
        self.sen_lengths = sen_lengths
        self.ignore_index = -1
        self.data_num = len(self.predictions)

        self.sentences = all_sentences

        self.all_pred_set = all_pred_set


    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_quad(self, tags, aspect_spans, opinion_spans):
        quad_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = np.zeros(len(label2id))
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                        tag_num[int(tags[j][i])] += 1

                if sum(tag_num[7:10]) == 0: continue
                if sum(tag_num[10:]) == 0:continue

                sentiment = -1
                category = -1

                sentiment = np.argmax(tag_num[7:10]) + 7
                category = np.argmax(tag_num[10:]) + 10

                if sentiment == -1 or category == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    exit()
                quad_utm.append([category, al, ar, pl, pr, sentiment])
        return quad_utm

    def find_triplet(self, tags, aspect_spans, opinion_spans):
        quad_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = np.zeros(len(label2id))
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        tag_num[int(tags[i][j])] += 1
                        tag_num[int(tags[j][i])] += 1
                if sum(tag_num[7:10]) == 0: continue

                sentiment = -1

                sentiment = np.argmax(tag_num[7:10]) + 7

                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    exit()
                quad_utm.append([al, ar, pl, pr, sentiment])
        return quad_utm

    def find_quad_(self, pred_set, aspect_spans, opinion_spans, k):
        tags_k, idx = torch.topk(pred_set, k)
        quad_utm = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                pair = np.zeros(len(label2id))
                tag_num = np.zeros(len(label2id))
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        pair[int(idx[i][j][0])] += 1
                        pair[int(idx[j][i][0])] += 1
                if np.sum(pair[7:]) > 0:
                    if np.sum(pair[7:10])==0 or np.sum(pair[10:])==0:
                        for i in range(al, ar + 1):
                            for j in range(pl, pr + 1):
                                for n in range(k):
                                    if tags_k[i][j][n]/tags_k[i][j][0]==1:
                                        tag_num[int(idx[i][j][n])] += tags_k[i][j][n]
                                    if tags_k[j][i][n]/tags_k[j][i][0]==1:
                                        tag_num[int(idx[j][i][n])] += tags_k[j][i][n]
                        if np.sum(pair[7:10])==0:
                            if np.sum(tag_num[7:10])==0:
                                continue
                            sentiment = np.argmax(tag_num[7:10]) + 7
                            category = np.argmax(pair[10:]) + 10
                            quad_utm.append([category, al, ar, pl, pr, sentiment])
                            continue
                        if np.sum(pair[10:])==0:
                            if np.sum(tag_num[10:])==0:
                                continue
                            sentiment = np.argmax(pair[7:10]) + 7
                            category = np.argmax(tag_num[10:]) + 10
                            quad_utm.append([category, al, ar, pl, pr, sentiment])
                            continue
                else: continue  

                sentiment = -1
                category = -1

                sentiment = np.argmax(pair[7:10]) + 7
                category = np.argmax(pair[10:]) + 10

                if sentiment == -1 or category == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    exit()
                quad_utm.append([category, al, ar, pl, pr, sentiment])
        return quad_utm



    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i])
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i])
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i])
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i])
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1



    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = get_aspects(self.goldens[i], self.sen_lengths[i])
            golden_opinion_spans = get_opinions(self.goldens[i], self.sen_lengths[i])

            golden_quads = self.find_quad(self.goldens[i], golden_aspect_spans, golden_opinion_spans)

            for pair in golden_quads:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))
            predicted_aspect_spans = get_aspects(self.predictions[i], self.sen_lengths[i])
            predicted_opinion_spans = get_opinions(self.predictions[i], self.sen_lengths[i])

            predicted_quads = self.find_quad_(self.all_pred_set[i], predicted_aspect_spans, predicted_opinion_spans, 2)
            for pair in predicted_quads:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))
        correct_num = len(golden_set & predicted_set)

        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1


