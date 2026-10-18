"""
FewNERD数据预处理脚本
将BIO格式的txt文件转换为episode格式的jsonl文件
"""
import json
import random
from collections import defaultdict

def read_bio_file(file_path):
    """读取BIO格式的文件"""
    sentences = []
    current_sentence = {'word': [], 'label': []}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence['word']:
                    sentences.append(current_sentence)
                    current_sentence = {'word': [], 'label': []}
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    word, label = parts
                    current_sentence['word'].append(word)
                    current_sentence['label'].append(label)
    
    if current_sentence['word']:
        sentences.append(current_sentence)
    
    return sentences

def get_entity_types(sentences):
    """获取所有实体类型"""
    types = set()
    for sent in sentences:
        for label in sent['label']:
            if label != 'O':
                # 保持完整的实体类型，不要只取最后一部分
                entity_type = label
                types.add(entity_type)
    return list(types)

def filter_labels(words, labels, selected_types):
    """只保留选定类型的标签，其他标签改为O"""
    filtered_labels = []
    for label in labels:
        if label in selected_types:
            filtered_labels.append(label)
        else:
            filtered_labels.append('O')
    return words, filtered_labels

def create_episodes(sentences, N=5, K=5, Q=5, num_episodes=1000):
    """创建N-way K-shot Q-query的episode"""
    # 按实体类型组织句子
    type_to_sentences = defaultdict(list)
    all_types = get_entity_types(sentences)
    
    for sent in sentences:
        sent_types = set()
        for label in sent['label']:
            if label != 'O':
                # 保持完整的实体类型
                entity_type = label
                sent_types.add(entity_type)
        for t in sent_types:
            type_to_sentences[t].append(sent)
    
    episodes = []
    for _ in range(num_episodes):
        # 随机选择N个类型
        if len(all_types) < N:
            selected_types = all_types
        else:
            selected_types = random.sample(all_types, N)
        
        # 为每个类型选择K+Q个样本
        support_data = {'word': [], 'label': []}
        query_data = {'word': [], 'label': []}
        
        valid_episode = True
        for entity_type in selected_types:
            available_sents = [s for s in type_to_sentences[entity_type]]
            if len(available_sents) < K + Q:
                valid_episode = False
                break
            
            selected_sents = random.sample(available_sents, K + Q)
            
            # 前K个作为support
            for sent in selected_sents[:K]:
                # 过滤标签，只保留选定的类型
                filtered_words, filtered_labels = filter_labels(
                    sent['word'], sent['label'], selected_types
                )
                support_data['word'].append(filtered_words)
                support_data['label'].append(filtered_labels)
            
            # 后Q个作为query
            for sent in selected_sents[K:K+Q]:
                # 过滤标签，只保留选定的类型
                filtered_words, filtered_labels = filter_labels(
                    sent['word'], sent['label'], selected_types
                )
                query_data['word'].append(filtered_words)
                query_data['label'].append(filtered_labels)
        
        if valid_episode:
            episode = {
                'support': support_data,
                'query': query_data,
                'types': selected_types
            }
            episodes.append(episode)
    
    return episodes

def save_episodes(episodes, output_path):
    """保存episodes到jsonl文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    print(f"Saved {len(episodes)} episodes to {output_path}")

if __name__ == '__main__':
    # 设置参数
    N, K, Q = 5, 5, 5
    
    # 处理训练集
    print("Processing train.txt...")
    train_sentences = read_bio_file('data/FewNERD/train.txt')
    train_episodes = create_episodes(train_sentences, N, K, Q, num_episodes=10000)
    save_episodes(train_episodes, f'data/FewNERD/train_{N}_{K}.jsonl')
    
    # 处理验证集
    print("Processing dev.txt...")
    dev_sentences = read_bio_file('data/FewNERD/dev.txt')
    dev_episodes = create_episodes(dev_sentences, N, K, Q, num_episodes=500)
    save_episodes(dev_episodes, f'data/FewNERD/dev_{N}_{K}.jsonl')
    
    # 处理测试集
    print("Processing test.txt...")
    test_sentences = read_bio_file('data/FewNERD/test.txt')
    test_episodes = create_episodes(test_sentences, N, K, Q, num_episodes=1000)
    save_episodes(test_episodes, f'data/FewNERD/test_{N}_{K}.jsonl')
    
    print("Done!")
