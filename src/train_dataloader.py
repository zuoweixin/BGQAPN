import torch
import torch.utils.data as data
import os
import numpy as np
import json
import copy
import random
from torch.nn.utils.rnn import pad_sequence


class FewNerdDataset(data.Dataset):
    def __init__(self, filepath, args, neg_sample):
        if not os.path.exists(filepath):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.class2sampleid = {}
        self.tokenizer = args.tokenizer
        self.samples, self.types = self.__load_data_from_file__(filepath)
        self.max_negspan_num = args.max_negspan_num
        self.max_span_size = args.max_span_size
        self.neg_sample = neg_sample
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

    def __load_data_from_file__(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            lines = f.readlines()

        parsed_lines = []
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            try:
                parsed_lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        types = []
        for i in parsed_lines:
            for j in i["types"]:
                if j not in types:
                    types.append(j)
        return parsed_lines, types

    def _precompute_all(self):
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')
        sep_char = self._word_to_char_ids('[SEP]')
        cls_char = self._word_to_char_ids('[CLS]')

        for s in self.samples:
            for split_name in ('support', 'query'):
                split = s[split_name]
                for sent in split['word']:
                    for w in sent:
                        if w not in self._word_token_cache:
                            self._word_token_cache[w] = self.tokenizer.encode(w, add_special_tokens=False)
                            self._word_char_cache[w] = self._word_to_char_ids(w)

        self._CLS_token = CLS
        self._SEP_token = SEP
        self._CLS_char = cls_char
        self._SEP_char = sep_char

    def __additem__(self, d, doc_encoding, entity_types, entity_masks, context_masks, char_ids):
        d['word'].append(doc_encoding)
        d['entity_types'].append(entity_types)
        d['entity_masks'].append(entity_masks)
        d['text_mask'].append(context_masks)
        d['char_ids'].append(char_ids)
        span_num = [len(entity_types)]
        d['sentence_num'].append(span_num)

    def _word_to_char_ids(self, word):
        char_ids = []
        for c in word[:self.max_char_len]:
            char_ids.append(self.char_vocab.get(c, self.char_vocab['<unk>']))
        while len(char_ids) < self.max_char_len:
            char_ids.append(self.char_vocab['<pad>'])
        return char_ids

    def _fast_tokenize(self, words, tags):
        spans, labels, token_idx = [], [], []
        char_ids_list = []

        doc_encoding = [self._CLS_token]
        char_ids_list.append(self._CLS_char)
        idx = 1
        max_length = 510

        for i in words:
            token_encoding = self._word_token_cache.get(i)
            if token_encoding is None:
                token_encoding = self.tokenizer.encode(i, add_special_tokens=False)
                self._word_token_cache[i] = token_encoding
                self._word_char_cache[i] = self._word_to_char_ids(i)
            if token_encoding:
                if idx + len(token_encoding) > max_length:
                    break
                token_idx.append(idx)
                idx += len(token_encoding)
                doc_encoding += token_encoding
                char_ids_list.extend([self._word_char_cache[i]] * len(token_encoding))

        doc_encoding += [self._SEP_token]
        char_ids_list.append(self._SEP_char)
        token_idx.append(idx)

        actual_word_count = len(token_idx) - 1
        tags = tags[:actual_word_count]

        current_label, current_span = [], []
        tags_ = tags + ['O']
        for idx, tag in enumerate(tags_):
            if tag == 'O' or (current_label != [] and tag != current_label[0]):
                if current_label:
                    labels.append(current_label[0])
                    start = current_span[0]
                    end = current_span[-1] + 1
                    if start < len(token_idx) and end <= len(token_idx):
                        start = token_idx[start]
                        end = token_idx[end]
                        spans.append([start, end])
                    current_label, current_span = [], []
            else:
                current_label.append(tag)
                current_span.append(idx)

        return doc_encoding, spans, labels, token_idx, char_ids_list

    def _fast_getraw(self, encoding, spans, labels, token_idx, max_span_size, neg_sample=True):
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

        if neg_sample and len(neg_entity_spans) > 0 and len(pos_entity_spans) > 0:
            pos_t = torch.tensor(pos_entity_spans, dtype=torch.int64)
            neg_t = torch.tensor(neg_entity_spans, dtype=torch.int64)
            pos_broadcast = pos_t.unsqueeze(1)
            neg_broadcast = neg_t.unsqueeze(0)
            dist = (pos_broadcast - neg_broadcast).abs().sum(-1)
            min_dist = dist.min(0)[0]
            order = torch.argsort(min_dist)
            limit = min(len(neg_entity_spans), self.max_negspan_num)
            picked = order[:limit].tolist()
            neg_entity_spans = [neg_entity_spans[i] for i in picked]
        elif len(neg_entity_spans) > 0:
            limit = min(len(neg_entity_spans), self.max_negspan_num)
            neg_entity_spans = neg_entity_spans[:limit]

        neg_entity_types = [0] * len(neg_entity_spans)

        pos_entity_types = [self.tag2label[i] for i in pos_entity_types]
        entity_types = pos_entity_types + neg_entity_types
        entity_masks = [list(s) for s in pos_entity_spans] + [list(s) for s in neg_entity_spans]
        context_masks = [1] * context_size

        return entity_types, entity_masks, context_masks

    def __populate__(self, data, savelabeldic=False, mask_dataset=False, neg_sample=True):
        dataset = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [], 'text_mask': [], 'char_ids': []}
        for i in range(len(data['word'])):
            current_word = copy.deepcopy(data['word'][i])
            current_label = copy.deepcopy(data['label'][i])
            if mask_dataset:
                for idx in range(len(current_label)):
                    if current_label[idx] != 'O':
                        current_word[idx] = '[MASK]'

            doc_encoding, spans, labels, token_idx, char_ids = self._fast_tokenize(current_word, current_label)
            entity_types, entity_masks, context_masks = self._fast_getraw(doc_encoding, spans, labels, token_idx, self.max_span_size, neg_sample)
            self.__additem__(dataset, doc_encoding, entity_types, entity_masks, context_masks, char_ids)

        if savelabeldic:
            dataset['label2tag'] = [self.tag2label]
        return dataset

    def __getitem__(self, index):
        sample = self.samples[index]
        support = sample['support']
        query = sample['query']
        target_classes = sample['types']
        distinct_tags = ['O'] + target_classes
        self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
        self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}

        support_set = self.__populate__(support, neg_sample=True)
        query_set = self.__populate__(query, savelabeldic=True, neg_sample=self.neg_sample)

        return support_set, query_set

    def __len__(self):
        return len(self.samples)


def resort_neg(positive, negtive):
    neg = torch.tensor(negtive)
    result = []
    for pos in positive:
        pos = torch.tensor(pos).unsqueeze(0)
        temp = torch.abs(neg - pos).sum(-1)
        result.append(temp)
    result = torch.stack(result, 0)
    result, _ = result.min(0)
    result = result.tolist()
    temp = list(zip(result, negtive))
    temp.sort()
    _, result = zip(*temp)
    return list(result)


def collate_fn(data):
    support_sets, query_sets = zip(*data)
    batch_support = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [], 'text_mask': [], 'char_ids': []}
    batch_query = {'word': [], 'entity_types': [], 'entity_masks': [], 'sentence_num': [], 'text_mask': [], 'label2tag': [], 'char_ids': []}
    for i in support_sets:
        for key in i:
            batch_support[key].extend(i[key])
    for i in query_sets:
        for key in i:
            if key != 'label2tag':
                batch_query[key].extend(i[key])
    batch_query['label2tag'] = query_sets[0]['label2tag'][0]

    for i in range(len(batch_support['word'])):
        for key in batch_support:
            batch_support[key][i] = torch.tensor(batch_support[key][i])

    for i in range(len(batch_query['word'])):
        for key in batch_query:
            if key != 'label2tag':
                batch_query[key][i] = torch.tensor(batch_query[key][i])

    batch_support['word'] = pad_sequence(batch_support['word'], batch_first=True, padding_value=0)
    batch_support['text_mask'] = pad_sequence(batch_support['text_mask'], batch_first=True, padding_value=0)
    batch_support['char_ids'] = pad_sequence(batch_support['char_ids'], batch_first=True, padding_value=0)
    batch_query['word'] = pad_sequence(batch_query['word'], batch_first=True, padding_value=0)
    batch_query['text_mask'] = pad_sequence(batch_query['text_mask'], batch_first=True, padding_value=0)
    batch_query['char_ids'] = pad_sequence(batch_query['char_ids'], batch_first=True, padding_value=0)

    return batch_support, batch_query


def get_train_loader(filepath, args, neg_sample=True, num_workers=2, collate_fn=collate_fn):
    dataset = FewNerdDataset(filepath, args, neg_sample)
    kwargs = dict(dataset=dataset,
                  batch_size=1,
                  shuffle=True,
                  pin_memory=True,
                  num_workers=num_workers,
                  collate_fn=collate_fn)
    if num_workers > 0:
        kwargs['persistent_workers'] = True
    data_loader = data.DataLoader(**kwargs)
    return data_loader
