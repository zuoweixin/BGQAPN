import os
# 设置 CUDA 内存分配策略，减少碎片化导致的 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import random
import numpy as np
import torch
from utils.data_reader import germ, split_data, nerel, genia, ace04, conll2003
from transformers import BertTokenizer
from src.trainer import Trainer
from src.train_dataloader import get_train_loader
from src.test_dataloader import  get_test_loader
from utils.data_reader import germ, split_data, nerel, genia, ace04
from utils.data_reader import ace05, ace05_chinese, vlsp18, vlsp16,label_num
from args import argparser
from transformers import logging
from datetime import datetime
logging.set_verbosity_warning()
logging.set_verbosity_error()

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparser()
    args = parser.parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda:" + str(args.select_gpu))
    args.device = device
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    args.tokenizer = tokenizer
    
    # 生成带时间戳的模型路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'checkpoint/FewNERD_{timestamp}.pkl'
    
    N, K = args.train_ways, args.train_shots
    trainer = Trainer(args)
    if args.do_train:
        print('training......')
        train_path = f'{args.train_path}/train_{N}_{K}.jsonl' 
        dev_path = f'{args.train_path}/dev_{N}_{K}.jsonl'
        train_data_loader = get_train_loader(train_path, args)
        val_data_loader = get_train_loader(dev_path, args, neg_sample=False)
        trainer.train(train_data_loader, val_data_loader, model_path=model_path)
    
    if args.do_predict:
        print('test......')
        
        # 如果没有训练，使用最新的checkpoint
        if not args.do_train:
            if args.checkpoint_path:
                model_path = args.checkpoint_path
                print(f'Using specified checkpoint: {model_path}')
            else:
                checkpoint_files = [f for f in os.listdir('checkpoint') if f.startswith('FewNERD_') and f.endswith('.pkl')]
                if checkpoint_files:
                    # 按时间排序，使用最新的
                    checkpoint_files.sort(reverse=True)
                    model_path = f'checkpoint/{checkpoint_files[0]}'
                    print(f'Using checkpoint: {model_path}')
                else:
                    print('[ERROR] No checkpoint found! Please train first.')
                    return
        
        result = []
        micro, macro, nested_micro,  = [], [], []
        nested_macro, flat_micro, flat_macro = [], [], []
        if args.test_data == 'fewnerd':
            print('test on fewnerd')
            test_path = f'{args.train_path}/test_{N}_{K}.jsonl' 
            test_data_loader = get_train_loader(test_path, args, neg_sample=False)
            for seed in range(10):
                set_seeds(seed)
                f1 = trainer.test_fewnerd(test_data_loader,model_path= model_path)
                result.append(f1)
            
        else:
            test_path = args.test_path
            if args.test_data == 'genia':
                if test_path == '':
                    test_path = r"data/genia/GENIAcorpus3.02.xml"
                sentence, span, label, total_type = genia(test_path)

            elif args.test_data == 'nerel':
                if test_path == '':
                    test_path = r"data/NEREL/NEREL-v1.1/test"
                sentence, span, label, total_type = nerel(test_path)

            elif args.test_data == 'germ':
                if test_path == '':
                    test_path = r"data/GermEval"
                sentence, span, label, total_type = germ(test_path)

            elif args.test_data == 'ace04':
                if test_path == '':
                    test_path = r"data/ACE2004_EN"
                sentence, span, label, total_type = ace04(test_path)

            elif args.test_data == 'ace05':
                if test_path == '':
                    test_path = r"data/ACE2005_EN"
                sentence, span, label, total_type = ace05(test_path, args.ace05_type)
            
            elif args.test_data == 'ace05_chinese':
                if test_path == '':
                    test_path = r"data/ACE2005_Chinese"
                sentence, span, label, total_type = ace05_chinese(test_path, args.ace05_type)

            elif args.test_data == 'vlsp18':
                if test_path == '':
                    test_path = r"../../data/VLSP2018"
                sentence, span, label, total_type = vlsp18(test_path)

            elif args.test_data == 'vlsp16':
                if test_path == '':
                    test_path = r"../../data/VLSP2016"
                sentence, span, label, total_type = vlsp16(test_path)

            elif args.test_data == 'conll2003':
                if test_path == '':
                    test_path = r"data/conll2003"
                sentence, span, label, total_type = conll2003(test_path)
            else:
                print("[ERROR] test data must be genia, nerel, germ, ace04, ace05, vlsp16 or vlsp18")
                assert (0)

            
            total_label_num = label_num(label, total_type)
            ignore_label = [key for key, value in total_label_num.items() if value < args.test_shots]
            total_type = [i for i in total_type if i not in ignore_label]
            
            for seed in range(0,10):
                set_seeds(seed)
                train_sen, train_span, train_label, test_sen, \
                test_span, test_label  = split_data(sentence, span, label, total_type, args.test_shots)
                total_type_witho = ['O'] + total_type
                
         
                finetune_dataloader = get_test_loader(args, train_sen, train_span, train_label, 
                                                    total_type_witho,args.finetune_batchsize, True)
                test_dataloader = get_test_loader(args, test_sen, test_span, test_label, 
                                                    total_type_witho,1, False)
                f1 = trainer.test(finetune_dataloader,test_dataloader, model_path=model_path)

                micro.append(f1[0])
                macro.append(f1[1])
                nested_micro.append(f1[2])
                nested_macro.append(f1[3])
                flat_micro.append(f1[4])
                flat_macro.append(f1[5])
                

        micro = np.array(micro)
        macro = np.array(macro)
        nested_micro = np.array(nested_micro)
        nested_macro = np.array(nested_macro)
        flat_micro = np.array(flat_micro)
        flat_macro = np.array(flat_macro)

        # 创建数据集对应的结果目录
        dataset_name = args.test_data.upper()
        if dataset_name == 'ACE04':
            result_dir = 'result/ACE2004'
        elif dataset_name == 'ACE05':
            result_dir = 'result/ACE2005'
        else:
            result_dir = 'result/OTHER'
        
        os.makedirs(result_dir, exist_ok=True)
        
        # 生成带时间戳的结果文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f'{result_dir}/{dataset_name}_{args.test_shots}shot_{timestamp}.txt'
        
        with open(result_filename, 'a+', encoding='utf-8') as f:
            # 写入实验配置信息
            f.write('='*80 + '\n')
            f.write(f'实验时间: {timestamp}\n')
            f.write(f'模型: {args.model_name}\n')
            f.write(f'数据集: {args.test_data}\n')
            f.write(f'Shot数: {args.test_shots}\n')
            f.write(f'数据划分策略: 混合所有数据(train+dev+test)，随机采样support集，剩余作为query集\n')
            f.write(f'微调迭代数: {args.finetune_iter}\n')
            f.write(f'微调学习率: {args.test_lr}\n')
            f.write(f'使用的checkpoint: {model_path}\n')
            f.write('='*80 + '\n\n')
            
            if args.test_data=='ace05' or args.test_data=='ace05_chinese':
                data = args.test_data + '-'+args.ace05_type
            else:
                data = args.test_data 

            if args.model_name == 'gbpe':
                introduction = f'model: {args.model_name}  shot: {args.test_shots}  data: {data}  margin: {args.margin} \n'
            else:
                introduction = f'model: {args.model_name}  shot: {args.test_shots}  data: {data} \n'
            
            f.write(introduction)
            mean = round(micro.mean()*100, 2)
            std = round(micro.std()*100, 2)
            f.write(f'micro_f1 value:  \n')
            f.write(str(micro.tolist()))
            f.write('\n')
            f.write(f'micro_f1 value    len: {len(micro)}, mean: {mean}, std: {std} \n')
           

            f.write(f'macro_f1 value:  \n')
            f.write(str(macro.tolist()))
            f.write('\n')
            f.write(f'macro value    len: {len(macro)}, mean: {round(macro.mean()*100, 2)}, std: {round(macro.std()*100, 2)} \n')
            # print(nested_micro)

            f.write(f'nested_micro value:  \n')
            f.write(str(nested_micro.tolist()))
            f.write('\n')
            f.write(f'nested_micro  len: {len(nested_micro)}, mean: {round(nested_micro.mean()*100, 2)}, std: {round(nested_micro.std()*100, 2)} \n')
            # print(nested_macro)

            f.write(f'nested_macro value:  \n')
            f.write(str(nested_macro.tolist()))
            f.write('\n')
            f.write(f'nested_macro   len: {len(nested_macro)}, mean: {round(nested_macro.mean()*100, 2)}, std: {round(nested_macro.std()*100, 2)} \n')
            # print(flat_micro)

            f.write(f'flat_micro value:  \n')
            f.write(str(flat_micro.tolist()))
            f.write('\n')
            f.write(f'flat_micro  len: {len(flat_micro)}, mean: {round(flat_micro.mean()*100, 2)}, std: {round(flat_micro.std()*100, 2)} \n')
            # print(flat_macro)

            f.write(f'flat_macro value:  \n')
            f.write(str(flat_macro.tolist()))
            f.write('\n')
            f.write(f'flat_macro   len: {len(flat_macro)}, mean: {round(flat_macro.mean()*100, 2)}, std: {round(flat_macro.std()*100, 2)} \n')
            
   

            f.write(f'======================================================================================== \n')
            f.write(f'======================================================================================== \n')
            f.write(f'\n')




if __name__ == '__main__':
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.exists('result'):
        os.mkdir('result')
    main()