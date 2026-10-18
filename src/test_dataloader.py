import torch
import torch.utils.data as data
import os
import numpy as np
import json
import copy
import random
from torch.nn.utils.rnn import pad_sequence


class FewShotDataset(data.Dataset):
    def __init__(self, args, sentence, span, label, total_type, neg_sample):
        self.args = args
        self.class2sampleid = {}
        self.tokenizer = args.tokenizer
        self.sentences = sentence
        self.span = span
        self.label = label
        self.total_type = total_type
        self.label2tag = {i: idx for idx, i in enumerate(total_type)}
        self.neg_sample = neg_sample

        self.max_negspan_num = args.max_negspan_num
        self.ignore_label_id = args.ignore_label_id
        self.max_span_size = args.max_span_size
        self.char_vocab = self._build_char_vocab()
        self.max_char_len = 15

        self._word_token_cache = {}
        self._word_char_cache = {}
        self._precompute_all()

    def _build_char_vocab(self):
        chars = set()
        chars.add('<pad>')
        chars.add('<unk>')
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            chars.add(c)
        for c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ':
            chars.add(c)
        return {char: idx for idx, char in enumerate(sorted(chars))}

    def _precompute_all(self):
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self._CLS_token = CLS
        self._SEP_token = SEP
        self._CLS_char = self._word_to_char_ids('[CLS]')
        self._SEP_char = self._word_to_char_ids('[SEP]')
        for sent in self.sentences:
            for w in sent:
                if w not in self._word_token_cache:
                    self._word_token_cache[w] = self.tokenizer.encode(w, add_special_tokens=False)
                    self._word_char_cache[w] = self._word_to_char_ids(w)

    def _word_to_char_ids(self, word):
        char_ids = []
        for c in word[:self.max_char_len]:
            char_ids.append(self.char_vocab.get(c, self.char_vocab['<unk>']))
        while len(char_ids) < self.max_char_len:
            char_ids.append(self.char_vocab['<pad>'])
        return char_ids

    def __additem__(self, d, doc_encoding, entity_types, entity_masks, context_masks, spans, token_idx, char_ids):
        d['word'].append(doc_encoding)
        d['entity_types'].append(entity_types)
        d['entity_masks'].append(entity_masks)
        d['text_mask'].append(context_masks)
        d['pos_span'].append(spans)
        d['char_ids'].append(char_ids)
        span_num = [len(entity_types)]
        d['sentence_num'].append(span_num)
        d['token_idx'].append(token_idx)

    def __getraw__(self, encoding, spans, token_idx, labels, neg_sample=False):
        max_span_size = self.max_span_size
        token_count = len(token_idx)
        context_size = len(encoding)
        pos_entity_spans = [spans[i] for i in range(len(spans))]
        pos_entity_types = [labels[i] for i in range(len(spans))]

        neg_entity_spans = []
        pos_set = set(tuple(s) for s in pos_entity_spans)
        token_idx_list = token_idx
        for size in range(1, max_span_size + 1):
            for i in range(0, (token_count - size)):
                span = (token_idx_list[i], token_idx_list[i + size])
                if span not in pos_set:
                    neg_entity_spans.append(span)

        if self.neg_sample and len(neg_entity_spans) > 0:
            sample_num = min(len(neg_entity_spans), self.max_negspan_num)
            if len(neg_entity_spans) > sample_num:
                idx = random.sample(range(len(neg_entity_spans)), sample_num)
                idx.sort()
                neg_entity_spans = [neg_entity_spans[i] for i in idx]
        neg_entity_types = [0] * len(neg_entity_spans)

        pos_entity_types = [self.label2tag[i] for i in pos_entity_types]
        entity_types = pos_entity_types + neg_entity_types
        entity_masks = [list(s) for s in pos_entity_spans] + [list(s) for s in neg_entity_spans]
        context_masks = [1] * context_size

        return entity_types, entity_masks, context_masks

    def __get_token_label_list__(self, words, tags):
        spans, token_idx = [], []
        char_ids_list = []

        doc_encoding = [self._CLS_token]
        char_ids_list.append(self._CLS_char)
        idx = 1

        for j, i in enumerate(words):
            token_encoding = self._word_token_cache.get(i)
            if token_encoding is None:
                token_encoding = self.tokenizer.encode(i, add_special_tokens=False)
                self._word_token_cache[i] = token_encoding
                self._word_char_cache[i] = self._word_to_char_ids(i)
            if token_encoding:
                token_idx.append(idx)
                idx += len(token_encoding)
                doc_encoding += token_encoding
                char_ids_list.extend([self._word_char_cache[i]] * len(token_encoding))

        doc_encoding += [self._SEP_token]
        char_ids_list.append(self._SEP_char)
        token_idx.append(idx)

        temp = []
        for j, i in enumerate(words):
            token_encoding = self._word_token_cache.get(i)
            if token_encoding and not token_idx:
                temp.append(j)

        for j in range(len(tags)):
            start_minus_count = 0
            end_minus_count = 0
            for i in temp:
                if tags[j][0] > i:
                    start_minus_count += 1
                if tags[j][1] > i:
                    end_minus_count += 1
            tags[j][0] -= start_minus_count
            tags[j][1] -= end_minus_count

        for idx, tag in enumerate(tags):
            if tag[0] >= len(token_idx) or tag[1] >= len(token_idx):
                continue
            if tag[0] < 0 or tag[1] < 0:
                continue
            start = token_idx[tag[0]]
            end = token_idx[tag[1]]
            spans.append([start, end])

        return doc_encoding, spans, token_idx, char_ids_list

    def __populate__(self, sentence, span, label, savelabeldic=False, mask_dataset=False):
        dataset = {'word': [], 'entity_types': [], 'entity_masks': [], 'text_mask': [],
                   'pos_span': [], 'sentence_num': [], 'label2tag': [], 'token_idx': [], 'char_ids': []}

        current_word = copy.deepcopy(sentence)
        current_span = copy.deepcopy(span)
        current_label = copy.deepcopy(label)
        idx = []
        for i in range(len(current_label)):
            if current_label[i] not in self.label2tag:
                idx.append(i)
        current_span = [current_span[i] for i in range(len(current_label)) if i not in idx]
        current_label = [current_label[i] for i in range(len(current_label)) if i not in idx]

        if mask_dataset:
            for idx in range(len(current_span)):
                if current_span[idx] != 'O':
                    current_word[idx] = '[MASK]'

        doc_encoding, spans, token_idx, char_ids = self.__get_token_label_list__(current_word, current_span)
        entity_types, entity_masks, context_masks = self.__getraw__(doc_encoding, spans, token_idx, current_label)
        self.__additem__(dataset, doc_encoding, entity_types, entity_masks, context_masks, spans, token_idx, char_ids)

        if savelabeldic:
            dataset['label2tag'] = self.label2tag
        return dataset

    def __getitem__(self, index):
        sentence = self.sentences[index]
        span = self.span[index]
        label = self.label[index]
        support_set = self.__populate__(sentence, span, label, savelabeldic=True)
        return support_set

    def __len__(self):
        return len(self.sentences)

    def nested_idx(self, spans):
        result = []
        for i in range(len(spans) - 1):
            for j in range(i + 1, len(spans)):
                if self._nest_span(spans[i], spans[j]):
                    result.append(i)
                    result.append(j)
        result = list(set(result))
        return result

    def _nest_span(self, span1, span2):
        if span1[0] <= span2[0] and span1[1] >= span2[1]:
            return True
        elif span2[0] <= span1[0] and span2[1] >= span1[1]:
            return True
        return False


def collate_fn(data):
    support_sets = data
    batch_support = {'word': [], 'entity_types': [], 'entity_masks': [], 'text_mask': [], 'sentence_num': [],
                     'label2tag': [], 'pos_span': [], 'token_idx': [], 'non_nested_idx': [], 'char_ids': []}

    for i in support_sets:
        for key in i:
            if key != 'label2tag':
                batch_support[key].extend(i[key])
    batch_support['label2tag'] = support_sets[0]['label2tag']

    for i in range(len(batch_support['word'])):
        for key in batch_support:
            if key != 'label2tag' and key != 'pos_span' and key != 'token_idx' and key != 'non_nested_idx':
                batch_support[key][i] = torch.tensor(batch_support[key][i])
    batch_support['word'] = pad_sequence(batch_support['word'], batch_first=True, padding_value=0)
    batch_support['text_mask'] = pad_sequence(batch_support['text_mask'], batch_first=True, padding_value=0)
    batch_support['char_ids'] = pad_sequence(batch_support['char_ids'], batch_first=True, padding_value=0)

    return batch_support


def get_test_loader(args, sentence, span, label, total_type, batch_size, neg_sample,
                    num_workers=2, collate_fn=collate_fn):
    dataset = FewShotDataset(args, sentence, span, label, total_type, neg_sample)
    kwargs = dict(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=False,
                  pin_memory=True,
                  num_workers=num_workers,
                  collate_fn=collate_fn)
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    data_loader = data.DataLoader(**kwargs)
    return data_loader
