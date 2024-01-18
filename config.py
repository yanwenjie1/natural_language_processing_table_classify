# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:32
@File    : config.py
@Function: 通用参数设置
@Other: XX
"""
import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()  # linux命令行格式下 解析配置信息
        return parser

    @staticmethod
    def initialize(parser):  # 初始化配置信息
        # args for path
        # chinese-bert-wwm-ext
        # chinese-albert-base-cluecorpussmall
        # chinese-albert-tiny
        # chinese-roberta-wwm-ext
        # chinese-roberta-base-wwm-cluecorpussmall
        # chinese-nezha-base
        parser.add_argument('--bert_dir', default='../model/chinese-roberta-small-wwm-cluecorpussmall/',
                            help='pre trained model dir for uer')
        parser.add_argument('--data_dir', default='./data/PaiMai/',
                            help='data dir for uer')

        # other args
        parser.add_argument('--model_type', default='cnn', type=str,
                            help='type of model, cnn or gp')
        parser.add_argument('--seed', type=int, default=1024,
                            help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        parser.add_argument('--max_seq_len', default=48, type=int,
                            help='cut the length of each table cell')
        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--batch_size', default=8, type=int)
        parser.add_argument('--train_epochs', default=50, type=int,
                            help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.3, type=float,
                            help='the drop out probability of pre train model ')
        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate of pre trained models')
        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='other learning rate')
        parser.add_argument('--max_grad_norm', default=0.5, type=float,
                            help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--use_advert_train', type=bool, default=True,
                            help='use advert or not --PGD')
        parser.add_argument('--fine_tuning', type=bool, default=True,
                            help='fine tuning or not --PGD')

        # table args
        parser.add_argument('--Row_Count', type=int, default=10)
        parser.add_argument('--Col_Count', type=int, default=10)
        parser.add_argument('--RoPE', type=bool, default=True,
                            help='是否使用旋转位置编码')
        parser.add_argument('--AbsoluteEncoding', type=bool, default=True,
                            help='是否使用绝对位置编码')
        parser.add_argument('--head_size', type=int, default=128,
                            help='GP中注意力头的数量')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
