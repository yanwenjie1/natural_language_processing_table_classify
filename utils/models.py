# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:46
@File    : models.py
@Function: 自定义的表格模型
@Other: XX
"""
import torch
import torch.nn as nn
from utils.base_model import BaseModel


class TableFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids, masks, location, labels):
        """
        表格的样本类定义
        :param token_ids:
        :param attention_masks:
        :param token_type_ids:
        :param masks:
        :param location:
        :param labels:
        """
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.masks = masks
        self.location = location
        self.labels = labels


class CnnTableClassify(BaseModel):
    def __init__(self, args):
        super(CnnTableClassify, self).__init__(bert_dir=args.bert_dir,
                                               dropout_prob=args.dropout_prob,
                                               model_name=args.model_name)
        self.args = args
        if self.args.AbsoluteEncoding:
            self.base_config.hidden_size += 384
            self.location_embed = nn.Parameter(torch.randn(self.args.Row_Count * self.args.Col_Count, 384))
        self.conv1 = nn.Conv2d(self.base_config.hidden_size, self.base_config.hidden_size // 2, 5, padding=2)
        self.conv2 = nn.Conv2d(self.base_config.hidden_size, self.base_config.hidden_size // 2, 3, padding=1)
        self.dense = nn.Linear(self.base_config.hidden_size, self.args.num_tags)

    def forward(self, token_ids, attention_masks, token_type_ids, masks, location):
        batch_size = token_ids.size(0)
        max_seq_len = token_ids.size(1)
        assert max_seq_len == self.args.Row_Count * self.args.Col_Count
        output = self.bert_module(input_ids=torch.reshape(token_ids, (batch_size * max_seq_len, token_ids.size(-1))),
                                  attention_mask=torch.reshape(attention_masks, (batch_size * max_seq_len, attention_masks.size(-1))),
                                  token_type_ids=torch.reshape(token_type_ids, (batch_size * max_seq_len, token_type_ids.size(-1))))
        sequence_output = output[1]
        sequence_output = torch.reshape(sequence_output, (batch_size, self.args.Row_Count, self.args.Col_Count, self.base_config.hidden_size - 384))

        location = torch.reshape(location, (batch_size * max_seq_len, 2))
        location = location[:, 0] * 10 + location[:, 1]
        index = location.view(-1).long()
        batch_location_embed = torch.index_select(self.location_embed, 0, index).view(batch_size, self.args.Row_Count, self.args.Col_Count, -1)

        sequence_output = torch.cat([sequence_output, batch_location_embed], dim=-1)

        sequence_output = torch.einsum('abcd->adbc', sequence_output)
        logits_1 = self.conv1(sequence_output)
        logits_2 = self.conv2(sequence_output)
        logits = torch.cat((logits_1, logits_2), dim=1)

        logits = torch.einsum('adbc->abcd', logits)
        logits = self.dense(logits)
        logits = torch.einsum('abcd->adbc', logits)

        return logits.contiguous()


class GlobalPointer(nn.Module):
    def __init__(self, hidden_size, heads, head_size, RoPE, args):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.dense = nn.Linear(hidden_size, self.heads * self.head_size * 2)
        self.RoPE = RoPE
        self.args = args

    def sinusoidal_position_embedding(self, input_tensor, batch_size, seq_len, output_dim, expend_dim):
        """

        :param input_tensor:
        :param batch_size:
        :param seq_len:
        :param output_dim:
        :param expend_dim: 0:row 1:col
        :return:
        """
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=input_tensor.device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=input_tensor.device)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        if expend_dim == 0:
            cos_pos = embeddings[..., None, 1::2].repeat_interleave(2, dim=-1).repeat_interleave(self.args.Col_Count, dim=-3)
            sin_pos = embeddings[..., None, 0::2].repeat_interleave(2, dim=-1).repeat_interleave(self.args.Col_Count, dim=-3)
        else:
            cos_pos = embeddings[..., None, 1::2].repeat_interleave(2, dim=-1).repeat(1, self.args.Row_Count, 1, 1)
            sin_pos = embeddings[..., None, 0::2].repeat_interleave(2, dim=-1).repeat(1, self.args.Row_Count, 1, 1)
        input_tensor_2 = torch.stack([-input_tensor[..., 1::2], input_tensor[..., 0::2]], -1).reshape(input_tensor.shape)
        return input_tensor * cos_pos + input_tensor_2 * sin_pos

    def forward(self, inputs, attention_mask):
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(inputs)
        outputs = torch.split(outputs, self.head_size * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.head_size], outputs[..., self.head_size:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim/2)
            qw_row = self.sinusoidal_position_embedding(qw, batch_size, self.args.Row_Count, self.head_size, 0)
            qw_col = self.sinusoidal_position_embedding(qw, batch_size, self.args.Col_Count, self.head_size, 1)
            qw = torch.cat((qw_row, qw_col), dim=-1)
            kw_row = self.sinusoidal_position_embedding(kw, batch_size, self.args.Row_Count, self.head_size, 0)
            kw_col = self.sinusoidal_position_embedding(kw, batch_size, self.args.Col_Count, self.head_size, 1)
            kw = torch.cat((kw_row, kw_col), dim=-1)

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len, seq_len)

        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        # 排除 上三角
        logits = logits - torch.triu(torch.ones_like(logits), 1) * 1e12

        return logits / self.head_size ** 0.5


class GPTableClassify(BaseModel):
    def __init__(self, args):
        super(GPTableClassify, self).__init__(bert_dir=args.bert_dir,
                                              dropout_prob=args.dropout_prob,
                                              model_name=args.model_name)
        self.args = args

        if self.args.AbsoluteEncoding:
            self.location_embed = nn.Parameter(torch.randn(self.args.Row_Count * self.args.Col_Count, 128))
            self.global_pointer = GlobalPointer(hidden_size=self.base_config.hidden_size + 128,
                                                heads=self.args.num_tags,
                                                head_size=self.args.head_size,
                                                RoPE=self.args.RoPE,
                                                args=self.args)
        else:
            self.global_pointer = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                                heads=self.args.num_tags,
                                                head_size=self.args.head_size,
                                                RoPE=self.args.RoPE,
                                                args=self.args)

    def forward(self, token_ids, attention_masks, token_type_ids, masks, location):
        batch_size = token_ids.size(0)
        max_seq_len = token_ids.size(1)
        output = self.bert_module(input_ids=torch.reshape(token_ids, (batch_size * max_seq_len, token_ids.size(-1))),
                                  attention_mask=torch.reshape(attention_masks, (batch_size * max_seq_len, attention_masks.size(-1))),
                                  token_type_ids=torch.reshape(token_type_ids, (batch_size * max_seq_len, token_type_ids.size(-1))))
        sequence_output = output[1]
        sequence_output = torch.reshape(sequence_output, (batch_size, max_seq_len, sequence_output.size(-1)))
        if self.args.AbsoluteEncoding:
            # 按照location坐标取location_embed
            location = torch.reshape(location, (batch_size * max_seq_len, 2))
            location = location[:, 0] * 10 + location[:, 1]
            index = location.view(-1).long()
            batch_location_embed = torch.index_select(self.location_embed, 0, index)
            batch_location_embed = batch_location_embed.view(batch_size, max_seq_len, -1)
            sequence_output = torch.cat([sequence_output, batch_location_embed], dim=-1)

        logits = self.global_pointer(sequence_output, masks)
        return logits
