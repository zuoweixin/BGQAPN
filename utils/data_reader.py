import xml.dom.minidom
import random
import os
import jsonlines
import json

def genia(xml_path):
    def get_sentence(node, n):
        sent_str = []
        sent_type = []
        len = 0
        if node == None:
            return '', [], 0

        if node.nodeName == '#text':
            assert node.firstChild == None
            data = node.data.strip()
            if data != '':
                sent_str = data.split(' ')
                len = sent_str.__len__()
        else:
            for child in node.childNodes:
                str_, type_, len_ = get_sentence(child, n + len)
                sent_str += str_
                sent_type += type_
                len += len_

        if node.nodeName == 'cons':
            type = node.getAttribute('sem')
            if type != '':
                sent_type.append(dict(type=type, start=n, end=n + len))

        return sent_str, sent_type, len

    dom = xml.dom.minidom.parse(xml_path)

    root = dom.documentElement

    articles = []
    sentences = []
    import_ = []
    other = []
    empty = []
    for nodes in root.childNodes:
        if nodes.nodeName == '#text':
            empty.append(nodes)
        if nodes.nodeName == 'import':
            import_.append(nodes)
        if nodes.nodeName == 'article':
            articles.append(nodes)
            sentences += nodes.getElementsByTagName('sentence')

    res = []
    total_type = []
    type_num = 0
    for sentence in sentences:
        s, t, l = get_sentence(sentence, 0)
        for i in t:
            current_type = i['type'].split(' ')
            if len(current_type) == 1:
                current_type = current_type[0][2:]
                i['type'] = current_type
                continue
            else:
                for j in current_type:
                    if j[:2] == 'G#':
                        current_type = j[2:]
                        i['type'] = current_type
                        break

        for i in t:
            if i['type'] not in total_type:
                total_type.append(i['type'])
        res.append(dict(sentence=s, type=t))
        type_num += t.__len__()

    sentence, span, label = [], [], []
    for i in res:
        sentence.append(i['sentence'])
        current_span, current_label = [], []
        for j in i['type']:
            current_span.append([j['start'], j['end']])
            current_label.append(j['type'])
        span.append(current_span)
        label.append(current_label)

    label_num = {i: 0 for i in total_type}
    for i in label:
        for j in i:
            label_num[j] += 1
    
    ignore_label = [key for key, value in label_num.items() if value < 50]
    total_type = [i for i in total_type if i not in ignore_label]
    return sentence, span, label, total_type


def remove_space(sentence, span):
    result_sentence = []
    temp = []
    for idx, i in enumerate(sentence):
        if i != '':
            result_sentence.append(i)
        else:
            temp.append(idx)

    for i in temp:
        for j in range(len(span)):
            if span[j][0] >= i:
                span[j][0] -= 1
            if span[j][1] > i:
                span[j][1] -= 1

    return result_sentence, span


def nerel(path):
    for root, dirs, files in os.walk(path):
        pass
    filenum = int(len(files) / 2)
    sentences, spans, labels = [], [], []

    ann_file_all, txt_file_all = [], []
    for i in files:
        if i[-4:] == '.ann':
            ann_file_all.append(i)
            text_file_name = i[:-4]+'.txt'
            assert text_file_name in files
            txt_file_all.append(text_file_name)


    for ann_file,  txt_file in zip(ann_file_all, txt_file_all):
        # ann_file, txt_file = files[2 * file], files[2 * file + 1]
        assert ann_file[:-4] == txt_file[:-4]
        ann_file = path + f'/{ann_file}'
        txt_file = path + f'/{txt_file}'
        current_sentences, current_sentences_len = [], []

        start = 0
        with open(txt_file, 'r', encoding='UTF-8') as f:
            orign_sentences = f.readlines()
            for i in range(len(orign_sentences)):
                temp_sentence = orign_sentences[i].replace('\n', '')
                if temp_sentence:
                    temp = temp_sentence.split(' ')
                    temp_len = []
                    current_sentences.append(temp)
                    for j in temp:
                        end = start + len(j)
                        temp_len.append([start, end])
                        start = end + 1
                    current_sentences_len.append(temp_len)
                else:
                    start += 1

        senten_len = [i[-1][-1] for i in current_sentences_len]
        dic = [{} for i in range(len(senten_len))]
        dic_label = [{} for i in range(len(senten_len))]

        with open(ann_file, 'r', encoding='UTF-8') as f:
            orign_sentences = f.readlines()
        for i in range(len(orign_sentences)):
            current = orign_sentences[i].replace('\n', '')
            if current[0] == 'T':
                current = current.replace('\t', ' ')
                current = current.split(' ')[1:]
                if ';' in current[2]:
                    continue
                start, end = int(current[1]), int(current[2])
                entity = current[3:]
                for idx, j in enumerate(senten_len):
                    if end <= j:
                        key = str(start) + ' ' + str(end)
                        dic[idx][key] = entity
                        dic_label[idx][key] = current[0]
                        break

        for i in range(len(current_sentences)):

            current_spans, current_entity = list(dic[i].keys()), list(dic[i].values())
            current_labels = list(dic_label[i].values())
            for j in range(len(current_spans)):
                current_spans[j] = current_spans[j].split(' ')
            start_list = [int(j[0]) for j in current_spans]
            end_list = [int(j[1]) for j in current_spans]
            temp = current_sentences[i]
            temp_len = current_sentences_len[i]
            orign_start = [k[0] for k in temp_len]

            for j in start_list:
                if j not in orign_start:
                    most = []
                    for k in orign_start:
                        if j - k > 0:
                            most.append(j - k)
                        else:
                            most.append(5e10)
                    min_num = min(most)
                    idx = most.index(min_num)
                    temp_words, temp_len_ = temp[idx], temp_len[idx][0]
                    after_words = [temp_words[:min_num], temp_words[min_num:]]
                    after_len = [temp_len_, temp_len_ + min_num]

                    current_sentences[i][idx] = after_words[1]
                    current_sentences[i].insert(idx, after_words[0])
                    current_sentences_len[i][idx][0] = after_len[1]
                    current_sentences_len[i].insert(idx, after_len)
                    temp = current_sentences[i]
                    temp_len = current_sentences_len[i]
                    orign_start = [k[0] for k in temp_len]

            temp = current_sentences[i]
            temp_len = current_sentences_len[i]
            orign_end = [k[1] for k in temp_len]
            for j in end_list:
                if j not in orign_end:
                    most = []
                    for k in orign_end:
                        if k - j > 0:
                            most.append(k - j)
                        else:
                            most.append(5e10)
                    min_num = min(most)
                    idx = most.index(min_num)
                    temp_words, temp_len_ = temp[idx], temp_len[idx][1]
                    after_words = [temp_words[:-min_num], temp_words[-min_num:]]
                    after_len = [temp_len_ - min_num, temp_len_]

                    current_sentences[i][idx] = after_words[0]
                    current_sentences[i].insert(idx, after_words[1])
                    current_sentences[i][idx], current_sentences[i][idx + 1] = current_sentences[i][idx + 1], \
                                                                               current_sentences[i][idx]
                    current_sentences_len[i][idx][1] = after_len[0]
                    current_sentences_len[i].insert(idx, after_len)
                    current_sentences_len[i][idx], current_sentences_len[i][idx + 1] = current_sentences_len[i][
                                                                                           idx + 1], \
                                                                                       current_sentences_len[i][idx]
                    temp = current_sentences[i]
                    temp_len = current_sentences_len[i]
                    orign_end = [k[1] for k in temp_len]
            temp_len = current_sentences_len[i]
            orign_start = [k[0] for k in temp_len]
            orign_end = [k[1] for k in temp_len]
            for j in start_list:
                assert j in orign_start
            for j in end_list:
                assert j in orign_end

            for j in range(len(start_list)):
                start = orign_start.index(start_list[j])
                end = orign_end.index(end_list[j]) + 1
                current_spans[j] = [start, end]

            if '' in current_sentences[i]:
                temp, current_spans = remove_space(current_sentences[i], current_spans)
            sentences.append(temp)
            spans.append(current_spans)
            labels.append(current_labels)

    total_type = []
    for i in labels:
        for j in i:
            if j not in total_type:
                total_type.append(j)

    return sentences, spans, labels, total_type


def samp_deepth(a):
    deepth = len(a[0])
    temp = True
    for i in a:
        if len(i) != deepth:
            temp = False
    return temp


def germ(path):
    test_path = path + '/NER-de-test.tsv'
    # test_path = path
    sentences, spans, labels, total_type = [], [], [], []
    with open(test_path, encoding='UTF-8') as f:
        filelines = f.readlines()
    current_sentence = []
    for idx, line in enumerate(filelines):
        current_sentence.append(line)
        if line == '\n':
            if current_sentence[0][0] == '#':
                current_sentence = current_sentence[1:]
            sentence, label, span, temp_label = [], [], [], []
            for i in current_sentence:
                temp = i.replace('\n', '')
                if temp:
                    temp = temp.replace('\t', ' ')
                    temp = temp.split(' ')
                    sentence.append(temp[1])
                    label.append(temp[2:])
            assert samp_deepth(label)
            deepth = len(label[0])
            for i in range(deepth):
                start = []
                current_label = []
                j = 0
                while j < len(label):
                    if label[j][i] == 'O':
                        if start == []:
                            j += 1
                            current_label = []
                        else:
                            span.append([start[0], start[-1] + 1])
                            temp_label.append(current_label)
                            j = start[-1] + 1
                            start, current_label = [], []

                    elif label[j][i][0] == 'I':
                        start.append(j)
                        j += 1
                    else:
                        if current_label:
                            temp_label.append(current_label)
                            current_label = label[j][i][2:]
                            span.append([start[0], start[-1] + 1])

                            start = []
                            start.append(j)
                            j = start[-1] + 1
                        else:
                            current_label = label[j][i][2:]
                            start.append(j)
                            j += 1
                if start:
                    span.append([start[0], start[-1] + 1])
                    temp_label.append(current_label)
                    start = []
            if temp_label:
                sentences.append(sentence)
                spans.append(span)
                labels.append(temp_label)
            current_sentence = []

    for i in labels:
        for j in i:
            if j not in total_type:
                total_type.append(j)

    return sentences, spans, labels, total_type

def conll2003(path):
    file_names = ['train', 'dev', 'test']
    sentence, span, label = [], [], []
    total_type = []

    for fn in file_names:
        file_path = path + f'/{fn}.txt'
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            current_tokens, current_tags = [], []
            for line in f:
                line = line.strip()
                if line == '':
                    if current_tokens:
                        spans, labels = _bio_to_spans(current_tags)
                        if spans:
                            sentence.append(current_tokens)
                            span.append(spans)
                            label.append(labels)
                            for l in labels:
                                if l not in total_type:
                                    total_type.append(l)
                    current_tokens, current_tags = [], []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_tokens.append(parts[0])
                        current_tags.append(parts[1])
            if current_tokens:
                spans, labels = _bio_to_spans(current_tags)
                if spans:
                    sentence.append(current_tokens)
                    span.append(spans)
                    label.append(labels)
                    for l in labels:
                        if l not in total_type:
                            total_type.append(l)

    return sentence, span, label, total_type


def _bio_to_spans(tags):
    spans, labels = [], []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == 'O':
            i += 1
            continue
        if tag.startswith('B-'):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == 'I-' + etype:
                i += 1
            spans.append([start, i])
            labels.append(etype)
        elif tag.startswith('I-'):
            etype = tag[2:]
            start = i
            i += 1
            while i < len(tags) and tags[i] == 'I-' + etype:
                i += 1
            spans.append([start, i])
            labels.append(etype)
        else:
            i += 1
    return spans, labels



def ace04_reader(path, mode):
    train_path = path + r'/'+ mode + r'/'
    for root, dirs, train_files in os.walk(train_path):
        pass
    
    # total_sentence, total_span = {}, {}
    # total_label, total_type = {}, {}
    total_sentence, total_span = [], []
    total_label, total_type = [], []
    for train_json in train_files:
        with open(train_path + train_json, 'r', encoding="utf-8") as f:
            temp_sentence, temp_span = [], []
            temp_label, temp_type = [], []
            
            for items in jsonlines.Reader(f):
                total_len = 0
                for idx, spans in enumerate(items['ner']):
                    
                    if spans:
                        temp_sentence.append(items['sentences'][idx])
                        span_ = [[i[0]-total_len, i[1]+1-total_len] for i in spans]
                        label_ = [i[2] for i in spans]
                        temp_span.append(span_)
                        temp_label.append(label_)
                        for i in label_:
                            if i not in temp_type:
                                temp_type.append(i)

                    total_len += len(items['sentences'][idx])

            # total_sentence[int(train_json[0])] = temp_sentence
            # total_span[int(train_json[0])] = temp_span
            # total_label[int(train_json[0])] = temp_label
            # total_type[int(train_json[0])] = temp_type

            total_sentence += temp_sentence
            total_span  += temp_span
            total_label += temp_label
            for i in temp_type:
                if i not in total_type:
                    total_type.append(i)
    
    return total_sentence, total_span, total_label, total_type


def ace05_reader(path, grained):
    """
    读取ACE2005数据集（原始方式：混合train+dev+test）
    """
    files = ['train', 'test', 'dev']
    total_sentence, total_span = [], []
    total_label, total_type = [], []
    
    for file in files:
        # 先尝试.jsonlines格式（与ACE2004一致）
        jsonlines_path = path + r'/'+ file + '.jsonlines'
        json_path = path + r'/'+ file + '.json'
        
        file_path = None
        use_jsonlines = False
        
        if os.path.exists(jsonlines_path):
            file_path = jsonlines_path
            use_jsonlines = True
        elif os.path.exists(json_path):
            file_path = json_path
            use_jsonlines = False
        else:
            print(f"[WARNING] Neither {jsonlines_path} nor {json_path} exists, skipping...")
            continue
        
        with open(file_path, 'r', encoding="utf-8") as f:
            if use_jsonlines:
                # 使用jsonlines格式（与ACE2004相同的格式）
                for items in jsonlines.Reader(f):
                    if items.get('entity_mentions'):
                        temp_span, temp_label = [], []
                        sentence = items['tokens']
                        
                        for entity in items['entity_mentions']:
                            temp_span.append([entity['start'], entity['end']])
                            if grained == 'coarse':
                                temp_label.append(entity['entity_type'])
                            else:
                                temp_label.append(entity['entity_type'] + entity.get('entity_subtype', ''))
                        
                        if temp_span:
                            total_sentence.append(sentence)
                            total_span.append(temp_span)
                            total_label.append(temp_label)
                            
                            for label in temp_label:
                                if label not in total_type:
                                    total_type.append(label)
            else:
                # 使用json格式（原始格式）
                load_data = json.load(f)
                total_type_start = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
                
                for items in load_data:
                    if items.get('golden-entity-mentions'):
                        sentence = items['words']
                        spans = items['golden-entity-mentions']
                        temp_span, temp_label = [], []
                        
                        for i in spans:
                            current_span = [i['start'], i['end']]
                            current_label = i['entity-type']
                            current_label_start = current_label[:3]
                            if current_span in temp_span or current_label_start not in total_type_start:
                                continue
                            
                            if grained == 'coarse':
                                temp_label.append(current_label_start)
                            elif grained == 'fine':
                                temp_label.append(current_label)
                            else:
                                print("[ERROR] grained should be 'coarse' or 'fine'!")
                                assert(0)
                            
                            temp_span.append(current_span)
                        
                        if temp_span:
                            total_sentence.append(sentence)
                            total_span.append(temp_span)
                            total_label.append(temp_label)
                            
                            for i in temp_label:
                                if i not in total_type:
                                    total_type.append(i)

    return total_sentence, total_span, total_label, total_type


def ace04(path, grained='coarse'):
    """
    读取ACE2004数据集（原始方式：混合train+dev+test）
    """
    file_name = ['train', 'dev', 'test']

    sentence, span, label = [], [], []
    total_type = []

    for i in file_name:
        train_path = path + f'/{i}.jsonlines' 
        with open(train_path, 'r', encoding="utf-8") as f:
            for items in jsonlines.Reader(f):
                if items['entity_mentions']:
                    temp_span, temp_label = [], []
                    sentence.append(items['tokens'])
                    for entity in items['entity_mentions']:
                        temp_span.append([entity['start'], entity['end']])
                        if grained=='coarse':
                            temp_label.append(entity['entity_type'])
                        else:
                            if entity['entity_subtype'] != '':
                                print()
                            temp_label.append(entity['entity_type'] + entity['entity_subtype'])
                    
                    span.append(temp_span)
                    label.append(temp_label)

                    for entity_type in temp_label:
                        if entity_type not in total_type:
                            total_type.append(entity_type)

    return sentence, span, label, total_type


def ace05(path, grained='coarse'):
    train_sentence, train_span, train_label, train_type = ace05_reader(path, grained)
    return train_sentence, train_span, train_label, train_type



def ace05_chinese_reader(path, grained):
    # files = ['dev']
    files = ['train', 'test', 'dev']
    total_type_start = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
    total_sentence, total_span = [], []
    total_label, total_type = [], []
    for file in files:
        with open(path + r'/'+ file + '.json', 'r', encoding="utf-8") as f:
            load_data = json.load(f)
            for items in load_data:
                if items['golden-entity-mentions']:
                    sentence = items['sentence']
                    spans = items['golden-entity-mentions']
                    temp_span, temp_label = [], []
                    for i in spans:
                        current_span = [i['start'], i['end']]
                        current_label = i['entity-type']
                        current_label_start = current_label[:3]
                        if current_span in temp_span or current_label_start not in total_type_start:
                            continue
                        
                        if grained=='coarse':
                            temp_label.append(current_label_start)
                        elif grained=='fine':
                            temp_label.append(current_label)
                        else:
                            print("[ERROR] grained should be \'coarse\' or \'fine\' !")
                            assert(0)

                        temp_span.append(current_span)

                    if temp_span:
                        total_sentence.append(sentence)
                        total_span.append(temp_span)
                        total_label.append(temp_label)
            
                    for i in temp_label:
                        if i not in total_type:
                            total_type.append(i)

    return total_sentence, total_span, total_label, total_type

def ace05_chinese(path, grained='coarse'):
    train_sentence, train_span, train_label, train_type = ace05_chinese_reader(path, grained)
    train_sentence_ = [list(i) for i in train_sentence]

    #####################
    long_sentence_idx = []
    for idx, i in enumerate(train_sentence_):
        if len(i)>200:
            long_sentence_idx.append(idx)
    
    train_sentence_ = [i for idx, i in enumerate(train_sentence_) if idx not in long_sentence_idx]
    train_span = [i for idx, i in enumerate(train_span) if idx not in long_sentence_idx]
    train_label = [i for idx, i in enumerate(train_label) if idx not in long_sentence_idx]

    return train_sentence_, train_span, train_label, train_type




def vlsp_reader(path):
    total_sentences, total_span,  = [], []
    total_label = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for sentence in  lines:
            sentence = sentence.replace('\n', '')
            if '<ENAMEX' in sentence:
                sentence = sentence.replace('<ENAMEX', '')
                sentence = sentence.replace('">', '"> ')
                sentence = sentence.replace('</ENAMEX>', ' </ENAMEX> ')
                span_stack, label_stack = [], []
                word_num = 0
                temp_sentence, temp_span, temp_label = [], [], []
                sentence = sentence.split()
                for idx, words in enumerate(sentence):
                    if words[:6] == 'TYPE="':
                        span_stack.append(word_num)
                        label_stack.append(words[6:-2])
                    elif words == '</ENAMEX>':
                        temp_span.append([span_stack.pop(), word_num])
                        temp_label.append(label_stack.pop())
                    else:
                        temp_sentence.append(words)
                        word_num += 1
                
                assert span_stack == []
                total_sentences.append(temp_sentence)
                total_span.append(temp_span)
                total_label.append(temp_label)

    return total_sentences, total_span, total_label

def vlsp18(path):

    train_path = path + r'/VLSP2018-NER-train-Jan14/'
    dev_path = path + r'/VLSP2018-NER-dev/'
    test_path = path + r'/VLSP2018-NER-Test-Domains/'

    # total_path = [train_path, dev_path, test_path]
    total_path = [train_path]

    total_sentence, total_span = [], []
    total_label, total_type = [], []


    for current_path in total_path:
        listdir = os.listdir(current_path)
        for currentdir in listdir:
            sub_path = current_path + currentdir
            for _, _, files in os.walk(sub_path):
                pass
            for file in files:
                file_path = sub_path + '/'+ file
                sentences, span, label = vlsp_reader(file_path)
                total_sentence += sentences
                total_span += span
                total_label += label

                for i in label:
                    for j in i:
                        if j not in total_type:
                            total_type.append(j)

    remove_idx = []
    for idx, i in enumerate(total_sentence):
        if len(i)>150:
            remove_idx.append(idx)
    
    total_sentence = [total_sentence[i] for i in range(len(total_sentence)) if i not in remove_idx]
    total_span = [total_span[i] for i in range(len(total_span)) if i not in remove_idx]
    total_label = [total_label[i] for i in range(len(total_label)) if i not in remove_idx]


    return total_sentence, total_span, total_label, total_type


def vlsp16(path):
    for root, dirs, files in os.walk(path):
        pass
    sentences, spans, labels, total_type = [], [], [], []
    for file in files:
        test_path = path + f'/{file}'
        with open(test_path, encoding='UTF-8') as f:
            filelines = f.readlines()
        current_sentence = []
        for idx, line in enumerate(filelines):
            current_sentence.append(line)
            if line == '</s>\t\t\t\t\n':
                if current_sentence[0][0] == '\ufeff':
                    current_sentence = current_sentence[4:-1]
                else:
                    current_sentence = current_sentence[1:-1]
                sentence, label, span, temp_label = [], [], [], []
                for i in current_sentence:
                    temp = i.replace('\n', '')
                    if temp:
                        temp = temp.replace('\t', ' ')
                        temp = temp.split(' ')
                        s = temp[:-4]
                        sen = ''
                        if len(s) > 1:
                            for _,j in enumerate(s):
                                sen = sen + j +' '
                                sen.strip()
                        else:
                            sen = temp[0]   
                        sentence.append(sen)
                        label.append(temp[-2:])
                assert samp_deepth(label)
                deepth = len(label[0])
                for i in range(deepth):
                    start = []
                    current_label = []
                    j = 0
                    while j < len(label):
                        if label[j][i] == 'O':
                            if start == []:
                                j += 1
                                current_label = []
                            else:
                                span.append([start[0], start[-1] + 1])
                                temp_label.append(current_label)
                                j = start[-1] + 1
                                start, current_label = [], []

                        elif label[j][i][0] == 'I':
                            start.append(j)
                            j += 1
                        else:
                            if current_label:
                                temp_label.append(current_label)
                                current_label = label[j][i][2:]
                                span.append([start[0], start[-1] + 1])

                                start = []
                                start.append(j)
                                j = start[-1] + 1
                            else:
                                current_label = label[j][i][2:]
                                start.append(j)
                                j += 1
                    if start:
                        span.append([start[0], start[-1]])
                        temp_label.append(current_label)
                        start = []
                if temp_label:
                    sentences.append(sentence)
                    spans.append(span)
                    labels.append(temp_label)
                current_sentence = []
                
                if len(sentences) == 315:
                    print("hello")
                
    for i in labels:
        for j in i:
            if j not in total_type:
                total_type.append(j)

    return sentences, spans, labels, total_type




def _nest_span(span1, span2): 
    if span1[0] <= span2[0] and span1[1] >= span2[1]:
        result=True
    elif span2[0] <= span1[0] and span2[1] >= span1[1]:
        result=True
    else:
        result =False
    return result

def nested_idx(spans, label):
    result = []
    for i in range(len(spans)-1):
        for j in range(i+1, len(spans)):
            if _nest_span(spans[i], spans[j]):
                result.append(label[i])
                result.append(label[j])
    result = list(set(result))
    return result


def nested_type(spans, labels, total_labels):
    result = []
    for idx, i in enumerate(spans):
        label_nested = nested_idx(i, labels[idx])
        for j in label_nested:
            if j not in result:
                result.append(j)
        
    final = []
    for i in total_labels:
        if i not in result:
            final.append(i)
    
    return


def need_sample(dic: dict, shots, require_label=False):
    result = []
    for key, value in dic.items():
        if value < shots:
            result.append(key)

    if require_label:
        for i in require_label:
            if i in result:
                break
        return i
    else:
        if result:
            return True
        else:
            return False
        

def remove_ignore_span(span, label, total_type):
    nonignore_idx = []
    for idx, i in enumerate(label):
        if i in total_type:
            nonignore_idx.append(idx)
    result_span = [span[i] for i in nonignore_idx]
    result_label = [label[i] for i in nonignore_idx]
    return result_span, result_label

def split_data(sentence, span, label, total_type, shots):
    total_type_num = {}
    #先计算所有的total_type有多少对应的实体数目, 并按照从小到大顺序排序字典
    for i in label:
        for j in i:
            if j in total_type:
                if j not in total_type_num:
                    total_type_num[j] = 1
                else:
                    total_type_num[j] += 1
    temp = sorted(total_type_num.items(), key=lambda x:x[1])
    temp = [i[0] for i in temp]
    #开始采样
    sample_dit = {key: 0 for key in total_type}
    train_sen, train_span, train_label = [], [], []
    test_sen, test_span, test_label = [], [], []
    sampled_id = []
    total_idx = list(range(len(label)))
    while need_sample(sample_dit, shots):
        random.shuffle(total_idx)
        need_sample_label = need_sample(sample_dit, shots, temp)
        could_sample_index = []
        for i in total_idx:
            if need_sample_label in label[i] and i not in sampled_id:
                could_sample_index.append(i)
            if len(could_sample_index) > shots:
                break
        assert could_sample_index!= []
        idx = random.choice(could_sample_index)
        sampled_id.append(idx)
        train_sen.append(sentence[idx])
        result_span, result_label = remove_ignore_span(span[idx], label[idx], total_type)
        train_span.append(result_span)
        train_label.append(result_label)
        for i in label[idx]:
            if i in sample_dit:
                sample_dit[i] += 1
    
    for i in range(len(label)):
        if i not in sampled_id:
            test_sen.append(sentence[i])
            result_span, result_label = remove_ignore_span(span[i], label[i], total_type)
            test_span.append(result_span)
            test_label.append(result_label)
            
    delete_id = []   
    for i in range(len(train_sen)):
        temp_label = train_label[i]
        temp_dict = {}
        for j in temp_label:
            if j not in temp_dict:
                temp_dict[j] = 1
            else:
                temp_dict[j] += 1
        delete = True
        for j in temp_dict:
            if sample_dit[j] - temp_dict[j] <shots:
               delete = False
        if delete:
            delete_id.append(i)
            for j in temp_dict:
                sample_dit[j] -= temp_dict[j]
    
    train_sen_, train_span_, train_label_ = [],[],[]
    for i in range(len(train_sen)):
        if i not in delete_id:
            train_sen_.append(train_sen[i])
            train_span_.append(train_span[i])
            train_label_.append(train_label[i])
        else:
            test_sen.append(train_sen[i])
            test_span.append(train_span[i])
            test_label.append(train_label[i])
    
                
    return train_sen_, train_span_, train_label_, test_sen, test_span, test_label



def label_num(label, total_type):
    result = {}
    for i in label: 
        for j in i:
            if j in total_type:
                if j not in result:
                    result[j]=1
                else:
                    result[j]+=1
    return result



def weather_nested(current_span, temp_span):
        result= False
        if current_span[1]>temp_span[0] and current_span[1]<=temp_span[1]:
            if current_span[0]<=temp_span[0]:
                result = True
        
        if current_span[0]>=temp_span[0] and current_span[0]<temp_span[1]:
            if current_span[1]>=temp_span[1]:
                result = True
        
        if current_span[0]>=temp_span[0] and current_span[1]<=temp_span[1]:
            result = True   
        
        if current_span[0]<=temp_span[0] and current_span[1]>=temp_span[1]:
            result = True    
                
        return result


def nested_span_idx(span):
    flat_idx, nested_idx = [], []
    for i in range(len(span)):
        current_span = span[i]
        nested = False
        for j in range(len(span)):
            temp_span = span[j]
            if current_span == temp_span:
                continue
            if weather_nested(current_span, temp_span):
                nested = True
        
        if nested:
            nested_idx.append(i)
        else:
            flat_idx.append(i)

    return nested_idx, flat_idx

def count_num(sentence, span):
    
    total_span, nested_span, flat_span = 0, 0, 0
    total_sentence = 0
    for i in span:
        nested_idx, flat_idx = nested_span_idx(i)
        total_span+=len(i)
        nested_span+=len(nested_idx)
        flat_span+=len(flat_idx)
    total_sentence += len(sentence)

    return total_span, nested_span, flat_span

    

if __name__ == '__main__':
    # path = r'../../data/genia/GENIAcorpus3.02.xml'
    # sentence, span, label, total_type = genia(path)
    # train_sen, train_span, train_label, test_sen, \
    # test_span, test_label = split_data(sentence, span, label, total_type, 5)

    # path = r"../../data/GermEval"
    # sentence, span, label, total_type = germ(path)
    # ace_path = '../../data/ACE2005'
    # train_sentence, train_span, train_label, train_type = ace05(ace_path)

    # path = '../../data/VLSP2018'
    # train_sentence, train_span, train_label, train_type = vlsp18(path)

    # path = '../../data/ACE2005_Chinese'
    # #grained='coarse' or 'fine'
    # train_sentence, train_span, train_label, train_type = ace05_chinese(path, grained='coarse')

    # path = '../../data/ACE2005'
    # train_sentence, train_span, train_label, train_type = ace05(path, grained='coarse')

    path = '../../data/ACE2004'
    train_sentence, train_span, train_label, train_type = ace04(path)
    # total_span, nested_span, flat_span = count_num(train_sentence, train_span)

    print()



def split_train_test(train_sentence, train_span, train_label,
                     test_sentence, test_span, test_label,
                     total_type, shots):
    """
    从train集采样support，test集作为query
    标准的few-shot逻辑：每个类型采样shots个句子
    
    Args:
        train_sentence, train_span, train_label: 训练集数据
        test_sentence, test_span, test_label: 测试集数据
        total_type: 实体类型列表
        shots: 每类采样的句子数量（5-shot = 每类5个句子）
    
    Returns:
        support_sen, support_span, support_label: support集
        query_sen, query_span, query_label: query集（整个test集）
    """
    # 调试信息
    print(f"[DEBUG split_train_test] 收到参数:")
    print(f"  train: sentence={len(train_sentence)}, span={len(train_span)}, label={len(train_label)}")
    print(f"  test: sentence={len(test_sentence)}, span={len(test_span)}, label={len(test_label)}")
    print(f"  total_type={total_type}, shots={shots}")
    # 为每个类型找到包含该类型的所有句子
    type_to_sentences = {t: [] for t in total_type}
    
    for idx, labels in enumerate(train_label):
        for label in labels:
            if label in total_type:
                # 避免重复添加同一个句子
                if idx not in type_to_sentences[label]:
                    type_to_sentences[label].append(idx)
    
    # 检查每个类型是否有足够的句子
    for entity_type, sentence_ids in type_to_sentences.items():
        if len(sentence_ids) < shots:
            raise ValueError(
                f"类型 {entity_type} 在train集中只出现在 {len(sentence_ids)} 个句子中，"
                f"少于 {shots} 个句子。请减少shot数或过滤该类型。"
            )
    
    # 为每个类型独立采样shots个句子
    sampled_indices = set()
    
    for entity_type in total_type:
        available_sentences = type_to_sentences[entity_type]
        # 随机选择shots个句子
        selected = random.sample(available_sentences, shots)
        sampled_indices.update(selected)
    
    # 构建support集
    support_sen, support_span, support_label = [], [], []
    
    # 确保train数据长度一致
    min_train_len = min(len(train_sentence), len(train_span), len(train_label))
    
    for idx in sorted(sampled_indices):
        # 检查索引是否在有效范围内
        if idx >= min_train_len:
            print(f"[WARNING] Support索引 {idx} 超出范围（最大{min_train_len}），跳过")
            continue
        
        sentence = train_sentence[idx]
        
        # 确保span和label长度一致
        if len(train_span[idx]) != len(train_label[idx]):
            print(f"[WARNING] Support句子 {idx} 的span和label长度不匹配，跳过")
            continue
        
        # 只保留目标类型的实体，并验证span是否有效
        filtered_span, filtered_label = [], []
        for s, l in zip(train_span[idx], train_label[idx]):
            if l in total_type:
                # 验证span索引是否在有效范围内
                if isinstance(s, list) and len(s) == 2:
                    if s[0] >= 0 and s[1] <= len(sentence) and s[0] < s[1]:
                        filtered_span.append(s)
                        filtered_label.append(l)
        
        # 只添加有有效实体的样本
        if filtered_span:
            support_sen.append(sentence)
            support_span.append(filtered_span)
            support_label.append(filtered_label)
    
    # query集就是整个test集（只保留目标类型的实体）
    query_sen, query_span, query_label = [], [], []
    
    # 确保三个列表长度一致
    min_len = min(len(test_sentence), len(test_span), len(test_label))
    if len(test_sentence) != len(test_span) or len(test_sentence) != len(test_label):
        print(f"[WARNING] Test数据长度不一致: sentence={len(test_sentence)}, span={len(test_span)}, label={len(test_label)}")
        print(f"[INFO] 使用最小长度: {min_len}")
    
    for idx in range(min_len):
        sentence = test_sentence[idx]
        
        # 只保留目标类型的实体，并验证span是否有效
        filtered_span, filtered_label = [], []
        
        # 确保span和label长度一致
        if len(test_span[idx]) != len(test_label[idx]):
            print(f"[WARNING] 句子 {idx} 的span和label长度不匹配，跳过")
            continue
        
        for s, l in zip(test_span[idx], test_label[idx]):
            if l in total_type:
                # 验证span索引是否在有效范围内
                if isinstance(s, list) and len(s) == 2:
                    if s[0] >= 0 and s[1] <= len(sentence) and s[0] < s[1]:
                        filtered_span.append(s)
                        filtered_label.append(l)
        
        # 添加所有样本（即使没有实体）
        query_sen.append(sentence)
        query_span.append(filtered_span)
        query_label.append(filtered_label)
    
    return support_sen, support_span, support_label, query_sen, query_span, query_label
