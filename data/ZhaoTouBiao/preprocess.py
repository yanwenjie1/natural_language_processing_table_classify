# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/16
@Time    : 8:48
@File    : preprocess.py
@Function: XX
@Other: XX
"""
import os
import json
import re
import torch
from tqdm import tqdm
import pickle
from transformers import BertTokenizer
import random
from utils.models import TableFeature
import config
import warnings


def load_label(filename):
    with open(filename, encoding='utf-8') as file:
        contents = file.read()
    contents = json.loads(contents)
    return contents


def load_data_table(filename):
    results = []
    labels_to_ids = {j: i for i, j in enumerate(all_entity_labels)}
    with open(filename, encoding='utf-8') as file:
        contents = file.read()
    contents = json.loads(contents)
    contents = sorted(contents, key=lambda x: x['id'])
    wrong_list = []
    for content in contents:
        try:
            # 从html中加载表格
            tables = re.findall(r'<table[^<>]*>[\s\S]*?</table>', content['data']['html'])
            assert len(tables) == 2
            # 正常来说，这里有俩表格，一个是加载了完整的实体，一个是为了获取位置信息
            entities = re.findall(r'<td>([\s\S]*?)</td>', tables[1])
            entities_location = re.findall(r'<td>([\s\S]*?)</td>', tables[0])
            entities_location = [get_location(i) for i in entities_location]
            assert len(entities) == pargs.Row_Count * pargs.Col_Count

            labels = content['annotations'][0]['result']
            # 这里默认取了第0个label，如果是多标签分类，这里要改
            labels = [
                {'id': i['id'], 'start': i['value']['start'], 'end': i['value']['end'],
                 'label': i['value']['labels'][0]}
                for i in labels]
            for item in labels:
                tr = int(re.findall(r'tr\[(\d+)\]', item['start'])[0]) - 1
                td = int(re.findall(r'td\[(\d+)\]', item['start'])[0]) - 1
                item['entity_ids'] = (tr, td)
            labels_new = [(i['entity_ids'], labels_to_ids[i['label']]) for i in labels]
            results.append((labels_new, entities, entities_location))
        except Exception as e:
            wrong_list.append(content['id'])

    print(wrong_list)

    return results


def get_location(text):
    # 'pad    0,4'
    # '标段(包)编号    0,1'
    results = re.findall(r'\s+(\d+),(\d+)$', text)
    result1 = int(results[0][0])
    result2 = int(results[0][1])
    return (result1, result2)


def convert_examples_to_features_cnn(examples, tokenizer: BertTokenizer):
    features = []
    for (entities, texts, location) in tqdm(examples):  # texts: list of str; entities: list of tuple (from_ids, to_ids, label)
        text_len = len(texts)
        masks = torch.zeros((text_len,), dtype=torch.uint8)
        labels = torch.zeros((len(all_entity_labels), pargs.Row_Count, pargs.Col_Count), dtype=torch.uint8)
        locations = torch.zeros((text_len, 2), dtype=torch.uint8)

        word_ids = tokenizer.batch_encode_plus(texts,
                                               max_length=pargs.max_seq_len,
                                               padding="max_length",
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        token_ids = word_ids['input_ids']
        attention_masks = word_ids['attention_mask'].byte()
        token_type_ids = word_ids['token_type_ids'].byte()

        for index_text, text in enumerate(texts):
            if text != 'PAD' and text != 'pad':
                masks[index_text] = 1

        for index_location, one_location in enumerate(location):
            locations[index_location, 0] = one_location[0]
            locations[index_location, 1] = one_location[1]

        for entity in entities:
            labels[entity[1], entity[0][0], entity[0][1]] = 1  # label to_ids from_ids

        feature = TableFeature(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            masks=masks,
            location=locations,
            labels=labels,
        )
        features.append(feature)

    return features


def deprecated(func):
    def wrapper(*args, **kwargs):
        warnings.warn("此方法已不在维护，逻辑上已弃用，当前仅作参考", category=DeprecationWarning)
        return func(*args, **kwargs)
    return wrapper


@deprecated
def convert_examples_to_features_gp(examples, tokenizer: BertTokenizer):
    features = []
    for (entities, texts, location) in tqdm(examples):  # texts: list of str; entities: list of tuple (from_ids, to_ids, label)
        text_len = len(texts)
        masks = torch.zeros((text_len,), dtype=torch.uint8)
        labels = torch.zeros((len(all_entity_labels), text_len, text_len), dtype=torch.uint8)
        locations = torch.as_tensor(location, dtype=torch.uint8)  # 总不会有 255 * 255 的大小吧

        word_ids = tokenizer.batch_encode_plus(texts,
                                               max_length=pargs.max_seq_len,
                                               padding="max_length",
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        token_ids = word_ids['input_ids']
        attention_masks = word_ids['attention_mask'].byte()
        token_type_ids = word_ids['token_type_ids'].byte()

        for index_text, text in enumerate(texts):
            if text != 'PAD' and text != 'pad':
                masks[index_text] = 1

        for index_location, one_location in enumerate(location):
            locations[index_location, 0] = one_location[0]
            locations[index_location, 1] = one_location[1]

        for entity in entities:
            entity_ids = entity[0][0] * pargs.Col_Count + entity[0][1]
            labels[entity[1], entity_ids, entity_ids] = 1  # label to_ids from_ids

        feature = TableFeature(
            token_ids=token_ids,
            attention_masks=attention_masks,
            token_type_ids=token_type_ids,
            masks=masks,
            location=locations,
            labels=labels,
        )
        features.append(feature)

    return features


if __name__ == '__main__':
    pargs = config.Args().get_parser()
    #  调整配置
    pargs.task_type = 'cnn'
    pargs.data_dir = os.getcwd()
    pargs.max_seq_len = 48
    pargs.Row_Count = 20
    pargs.Col_Count = 10

    my_tokenizer = BertTokenizer(os.path.join('../../' + pargs.bert_dir, 'vocab.txt'))
    all_entity_labels = json.load(open('labels.json', encoding='utf-8'))

    all_data = load_data_table('project-51-at-2023-08-25-11-09-a626349d.json')

    print('总样本量 ', len(all_data))

    random.shuffle(all_data)  # 打乱数据集
    train_data = all_data[int(len(all_data) / 8):]
    dev_data = all_data[:int(len(all_data) / 8) + 1]

    train_data = convert_examples_to_features_cnn(train_data, my_tokenizer)
    dev_data = convert_examples_to_features_cnn(dev_data, my_tokenizer)
    with open(os.path.join(pargs.data_dir, '{}.pkl'.format('train_data')), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(pargs.data_dir, '{}.pkl'.format('dev_data')), 'wb') as f:
        pickle.dump(dev_data, f)
