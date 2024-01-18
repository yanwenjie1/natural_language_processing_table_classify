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
from bs4 import BeautifulSoup
import config


def write_label(filename):
    results_labels = []
    with open(filename, encoding='utf-8') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda x: x['id'])
    for content in tqdm(contents):
        assert len(content['annotations']) == 1, '存在多人标注结果'
        annotations = content['annotations'][0]
        labels = annotations['result']
        labels = [{
            'id': i['id'],
            'start': i['value']['start'],
            'end': i['value']['end'],
            'labels': i['value']['hypertextlabels']
            }for i in labels]
        for item in labels:
            for one_label in item['labels']:
                if one_label not in results_labels:
                    results_labels.append(one_label)
    results_labels.sort()

    with open('labels.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(results_labels, ensure_ascii=False))
    return results_labels


def load_data_table(filename):
    results = []
    labels_to_ids = {j: i for i, j in enumerate(all_entity_labels)}
    with open(filename, encoding='utf-8') as file:
        contents = json.load(file)
    contents = sorted(contents, key=lambda x: x['id'])

    no_ids = [379161, 383224, 387671, 387771, 388220, 388262, 388267, 388399, 388425, 388535, 388348, 387793, 387827,
              387875, 389960, 390042, 390059, 390154, 390258, 388023, 388247, 388340, 388464, 388505, 388519, 388581,
              387739, 387885, 389967, 390213, 390261, 389166, 389181, 389202, 389368, 389369, 389429, 390131, 390184,
              390199, 389264, 388249, 389307, 389379, 390233, 389489, 388400, 389590, 388524, 389735, 387569, 388104,
              387874, 387984]

    wrong_list = []
    for content in contents:
        try:
            # 从html中加载表格
            # 语料ID范围是379160-379309、383020-383462
            # ID范围为：387553-390278
            if content['id'] in no_ids:
                continue
            if not ((content['id'] >= 379160 and content['id'] <= 379309) or
                    (content['id'] >= 383020 and content['id'] <= 383462) or
                    (content['id'] >= 387553 and content['id'] <= 390278)):
                continue

            html = content['data']['html']
            tables = re.findall(r'<table[^<>]*>[\s\S]*?</table>', html)
            assert len(tables) == 2, '表格数量不对'
            # 从标注结果判寻index-label
            labels = content['annotations'][0]['result']
            labels = [{
                'id': i['id'],
                'text': i['value']['text'],
                'start': i['value']['startOffset'],
                'end': i['value']['endOffset'],
                'labels': i['value']['hypertextlabels']
            } for i in labels]
            # 车号@1
            ids_to_labels = {}
            for label in labels:
                # '车号@1'
                index = re.findall(r'@(\d+)$', label['text'])
                assert len(index) == 1, '取@后数字失败' + '    ' + str(content['id']) + "    " + label['id']
                index = int(index[0])
                ids_to_labels[index] = label['labels']



            # 从第二个表格出发，填充模板
            entities = [['PAD' for _ in range(args.Col_Count)] for _ in range(args.Row_Count)]
            this_labels = []
            locations = [[args.Col_Count * args.Row_Count - 1 for _ in range(args.Col_Count)] for _ in range(args.Row_Count)]

            soup = BeautifulSoup(tables[1], 'html.parser')
            table = soup.find('table')

            rows = table.find_all('tr')
            for row_index, row in enumerate(rows):
                if row_index >= args.Row_Count:
                    continue
                cells = row.find_all('td')
                for col_index, cell in enumerate(cells):
                    if col_index >= args.Col_Count:
                        continue
                    content1 = cell.text.strip()
                    # '车号@1'
                    re_match = re.findall(r'(.*?)@(\d+)$', content1)
                    if len(re_match) != 1:
                        pass
                    assert len(re_match) == 1, '匹配单元格完整内容失败' + '    ' + str(content['id']) + content1
                    content1 = re_match[0][0]
                    index_father = int(re_match[0][1])
                    this_label = ids_to_labels.get(index_father, [])
                    for i in this_label:
                        # 固定映射
                        # 除了序号值，含有值的->其它指标值
                        # 序号值 -> 串联标识
                        # 主拍品名称 -> 其它指标值
                        if i == '序号值':
                            this_labels.append((labels_to_ids['串联标识'], row_index, col_index))
                        elif i == '主拍品名称':
                            this_labels.append((labels_to_ids['其它指标值'], row_index, col_index))
                        elif '值' in i and i != '其它指标值':
                            this_labels.append((labels_to_ids['其它指标值'], row_index, col_index))
                        this_labels.append((labels_to_ids[i], row_index, col_index))
                    entities[row_index][col_index] = content1
                    locations[row_index][col_index] = index_father - 1

            this_entities = []
            this_locations = []
            # 定义一维数组
            for row in entities:
                this_entities.extend(row)
            for location in locations:
                this_locations.extend(location)

            results.append((this_labels, this_entities, this_locations))
        except Exception as e:
            print(str(e))
            wrong_list.append(content['id'])

    print(wrong_list)

    return results

def convert_examples_to_features(examples, tokenizer: BertTokenizer):
    features = []
    for (one_labels, one_texts, one_location) in tqdm(examples):  # texts: list of str; entities: list of tuple (from_ids, to_ids, label)
        text_len = len(one_texts)
        assert text_len == args.Row_Count * args.Col_Count, '单元格数量不对'
        masks = torch.zeros((text_len,), dtype=torch.uint8)
        labels = torch.zeros((len(all_entity_labels), args.Row_Count, args.Col_Count), dtype=torch.uint8)
        locations = torch.as_tensor(one_location, dtype=torch.int16) # 8位无符号整数 取值范围为[0, 255]
        # locations = torch.zeros((text_len, 2), dtype=torch.uint8) # 8位无符号整数 取值范围为[0, 255]

        word_ids = tokenizer.batch_encode_plus(one_texts,
                                               max_length=args.max_seq_len,
                                               padding="max_length",
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
        token_ids = word_ids['input_ids']
        attention_masks = word_ids['attention_mask'].byte()
        token_type_ids = word_ids['token_type_ids'].byte()

        for index_text, text in enumerate(one_texts):
            if text != 'PAD':
                masks[index_text] = 1

        # for index_location, one_location in enumerate(location):
        #     locations[index_location, 0] = one_location[0]
        #     locations[index_location, 1] = one_location[1]

        for entity in one_labels:
            labels[entity[0], entity[1], entity[2]] = 1  # label to_ids from_ids

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
    args = config.Args().get_parser()
    #  调整配置
    args.task_type = 'cnn'
    args.data_dir = os.getcwd()
    args.max_seq_len = 48
    args.Row_Count = 20
    args.Col_Count = 10

    my_tokenizer = BertTokenizer(os.path.join('../../' + args.bert_dir, 'vocab.txt'))

    all_entity_labels = write_label('project-145-at-2023-12-05-06-24-750926bd.json')

    all_data = load_data_table('project-145-at-2023-12-05-06-24-750926bd.json')
    # all_data.extend(load_data_table('project-152-at-2023-12-12-08-17-98c484a9.json'))


    print('总样本量 ', len(all_data))

    random.shuffle(all_data)  # 打乱数据集
    train_data = all_data[int(len(all_data) / 8):]
    dev_data = all_data[:int(len(all_data) / 8) + 1]

    train_data = convert_examples_to_features(train_data, my_tokenizer)
    dev_data = convert_examples_to_features(dev_data, my_tokenizer)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('train_data')), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('dev_data')), 'wb') as f:
        pickle.dump(dev_data, f)
