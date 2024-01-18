# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/5/9
@Time    : 13:12
@File    : server.py
@Function: 开服务的脚本
@Other: XX
"""
import json
import os
import torch
import socket
import re
from flask import Flask, request
from gevent import pywsgi
from transformers import BertTokenizer
from bs4 import BeautifulSoup
from utils.functions import load_model_and_parallel, get_result
from utils.models import CnnTableClassify, GPTableClassify


def torch_env():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    """
    ip获取
    :return:
    """
    myIp = [item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None) if ':' not in item[4][0]][0]
    return myIp


def encode(texts):
    """

    :param texts: list of str
    :return:
    """
    # 一个比较简单的实现是 按照list直接喂给 tokenizer
    word_ids = tokenizer.batch_encode_plus(texts,
                                           max_length=args.max_seq_len,
                                           padding="max_length",
                                           truncation='longest_first',
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
    token_ids = word_ids['input_ids'].to(device)
    attention_masks = word_ids['attention_mask'].byte().to(device)
    token_type_ids = word_ids['token_type_ids'].to(device)

    return token_ids, attention_masks, token_type_ids


def decode(token_ids, attention_masks, token_type_ids, masks, location):
    """

    :param location:
    :param masks:
    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :return:
    """

    logits = model(token_ids, attention_masks, token_type_ids, masks, location)

    model_pre = get_result(logits, False, True)

    model_pre = [(label_list[i[1]], i[2], i[3], float(i[4])) for i in model_pre]

    return model_pre


def filling_entities(entities, table_str, locations):
    """
    填充预输入的文本矩阵
    :param entities:
    :param table_str:
    :return:
    """
    soup = BeautifulSoup(table_str, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')
    # entities_all = copy.deepcopy(entities)
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
            # assert len(re_match) == 1, '匹配单元格完整内容失败' + '    ' + str(content['id'])
            if len(re_match) == 1:
                entities[row_index][col_index] = re_match[0][0]
                locations[row_index][col_index] = int(re_match[0][1]) - 1
    return entities


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = './checkpoints/PaiMai-chinese-roberta-small-wwm-cluecorpussmall-2024-01-18'
args_path = os.path.join(model_name, 'args.json')
model_path = os.path.join(model_name, 'model_best.pt')
labels_path = os.path.join(model_name, 'labels.json')

port = 10087
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(labels_path, 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
if args.model_type == 'gp':
    model, device = load_model_and_parallel(GPTableClassify(args), args.gpu_ids, model_path)
elif args.model_type == 'cnn':
    model, device = load_model_and_parallel(CnnTableClassify(args), args.gpu_ids, model_path)
model.eval()
for name, param in model.named_parameters():
    param.requires_grad = False
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        # msgs = request.get_json("content")
        msgs = msgs.decode('utf-8')
        # print(msg)
        assert type(msgs) == str
        # 传入table 传出预测结果
        entities = [['PAD' for _ in range(args.Col_Count)] for _ in range(args.Row_Count)]
        locations = [[args.Col_Count * args.Row_Count - 1 for _ in range(args.Col_Count)] for _ in
                     range(args.Row_Count)]
        # 填充entities
        entities = filling_entities(entities, msgs, locations)

        this_entities = []
        this_locations = []
        # 定义一维数组
        for row in entities:
            this_entities.extend(row)
        for location in locations:
            this_locations.extend(location)

        location =torch.as_tensor(this_locations, dtype=torch.int16)
        location = torch.as_tensor(location, device=device).unsqueeze(0)

        token_ids, attention_masks, token_type_ids = encode(this_entities)
        masks = torch.tensor([i != 'PAD' for i in msgs], dtype=torch.long, device=device).unsqueeze(0)
        results = decode(token_ids.unsqueeze(0), attention_masks.unsqueeze(0), token_type_ids.unsqueeze(0), masks,
                         location)

        results = [(i[0], i[1], i[2], float(i[3]), entities[i[1]][i[2]], locations[i[1]][i[2]]) for i in results]

        res = json.dumps(results, ensure_ascii=False)
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port) + '/prediction')
    server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
