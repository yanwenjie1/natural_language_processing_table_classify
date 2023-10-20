# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : train.py
@Function: 训练主程序
@Other: XX
"""
import datetime
import os
import shutil
import logging
import torch
from utils.functions import set_seed, set_logger, save_json, reset_console
from utils.train_models import TrainModel
import config
import json
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pickle

args = config.Args().get_parser()
logger = logging.getLogger(__name__)


class TableDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)
        self.token_ids = [example.token_ids.long() for example in features]
        self.attention_masks = [example.attention_masks.byte() for example in features]
        self.token_type_ids = [example.token_type_ids.long() for example in features]
        self.masks = [example.masks.byte() for example in features]
        self.location = [example.location.long() for example in features]
        self.labels = [example.labels.long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'masks': self.masks[index],
                'location': self.location[index],
                'labels': self.labels[index]}

        return data


if __name__ == '__main__':
    args.data_name = os.path.basename(os.path.abspath(args.data_dir))
    args.model_name = os.path.basename(os.path.abspath(args.bert_dir))
    args.save_path = os.path.join('./checkpoints',
                                  args.data_name + '-' + args.model_name
                                  + '-' + str(datetime.date.today()))

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    # args.save_path = './checkpoints/ZhaoTouBiao-MultiClassify-chinese-albert-base-cluecorpussmall-2023-08-31'

    # 复制对应的labels文件
    shutil.copy(os.path.join(args.data_dir, 'labels.json'), os.path.join(args.save_path, 'labels.json'))
    set_logger(os.path.join(args.save_path, 'log.txt'))
    torch.set_float32_matmul_precision('high')

    if args.data_name == "ZhaoTouBiao":
        # set_seed(args.seed)
        args.batch_size = 6
        args.train_epochs = 50
        args.Row_Count = 20

    if args.data_name == "ZhaoTouBiao-MultiClassify":
        # set_seed(args.seed)
        args.Row_Count = 20
        args.use_advert_train = True

    # read sample
    with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    args.num_tags = len(label_list)

    # logger.info(args)
    reset_console(args)
    save_json(args.save_path, vars(args), 'args')

    with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
        train_features = pickle.load(f)
    train_dataset = TableDataset(train_features)
    train_sampler = SequentialSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=0)
    with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
        dev_features = pickle.load(f)
    dev_dataset = TableDataset(dev_features)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.batch_size,
                            sampler=dev_sampler,
                            num_workers=0)

    GpForTable = TrainModel(args, train_loader, dev_loader, label_list, logger)
    GpForTable.train()
    # GpForTable.test(os.path.join(args.save_path, 'model_best.pt'))
