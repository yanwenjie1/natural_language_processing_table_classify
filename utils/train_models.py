# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:04
@File    : train_models.py
@Function: 训练模型的类
@Other: XX
"""
import os
import torch
import numpy as np
import pynvml
from tqdm import tqdm
from utils.models import CnnTableClassify, GPTableClassify
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, save_model, get_result, criterion
from utils.adversarial_training import PGD


class TrainModel:
    def __init__(self, args, train_loader, dev_loader, labels, log):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.args = args
        self.labels = labels
        self.id2label = {k: v for k, v in enumerate(labels)}
        self.log = log
        if self.args.model_type == 'gp':
            self.model, self.device = load_model_and_parallel(GPTableClassify(self.args), args.gpu_ids)
        elif self.args.model_type == 'cnn':
            self.model, self.device = load_model_and_parallel(CnnTableClassify(self.args), args.gpu_ids)
        self.t_total = len(self.train_loader) * args.train_epochs  # global_steps
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(args, self.model, self.t_total)

    def train(self):
        best_f1 = 0.0
        self.model.zero_grad()
        if self.args.use_advert_train:
            pgd = PGD(self.model, emb_name='word_embeddings.')
            K = 3
        if not self.args.fine_tuning:
            for name, param in self.model.bert_module.named_parameters():
                param.requires_grad = False
        for epoch in range(1, self.args.train_epochs + 1):
            bar = tqdm(self.train_loader, ncols=80)
            losses = []
            for batch_data in bar:
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                train_outputs = self.model(batch_data['token_ids'],
                                           batch_data['attention_masks'],
                                           batch_data['token_type_ids'],
                                           batch_data['masks'],
                                           batch_data['location'])
                loss = criterion(train_outputs, batch_data['labels'])
                losses.append(loss.detach().item())
                bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                bar.set_description("[epoch] %s" % str(epoch))
                loss.backward()  # 反向传播 计算当前梯度

                if self.args.use_advert_train:
                    pgd.backup_grad()  # 保存之前的梯度
                    # 对抗训练
                    for t in range(K):
                        # 在embedding上添加对抗扰动, first attack时备份最开始的param.processor
                        # 可以理解为单独给embedding层进行了反向传播(共K次)
                        pgd.attack(is_first_attack=(t == 0))
                        if t != K - 1:
                            self.model.zero_grad()  # 如果不是最后一次对抗 就先把现有的梯度清空
                        else:
                            pgd.restore_grad()  # 如果是最后一次对抗 恢复所有的梯度
                        train_outputs_adv = self.model(batch_data['token_ids'],
                                                       batch_data['attention_masks'],
                                                       batch_data['token_type_ids'],
                                                       batch_data['masks'],
                                                       batch_data['location'])
                        loss_adv = criterion(train_outputs_adv, batch_data['labels'])
                        losses.append(loss_adv.detach().item())
                        bar.set_postfix(loss='%.4f' % (sum(losses) / len(losses)))
                        loss_adv.backward()

                    pgd.restore()  # 恢复embedding参数

                # 梯度裁剪 解决梯度爆炸问题 不解决梯度消失问题  对所有的梯度乘以一个小于1的 clip_coef=max_norm/total_grad
                # 和clip_grad_value的区别在于 clip_grad_value暴力指定了区间 而clip_grad_norm做范数上的调整
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()  # 根据梯度更新网络参数
                self.scheduler.step()  # 更新优化器的学习率
                self.model.zero_grad()  # 将所有模型参数的梯度置为0
            if epoch > self.args.train_epochs * 0.1:
                _, f1 = self.dev()
                if f1 > best_f1:
                    best_f1 = f1
                    save_model(self.args, self.model)
                self.log.info('[eval] epoch:{} f1_score={:.6f} best_f1_score={:.6f}'.format(epoch, f1, best_f1))
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里的0是GPU id
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                print("剩余显存：" + str(meminfo.free / 1024 / 1024))  # 显卡剩余显存大小
        self.test(os.path.join(self.args.save_path, 'model_best.pt'))

    def dev(self):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            X, Y, Z = 1e-15, 1e-15, 1e-15  # 相同的实体 预测的实体 真实的实体
            for dev_batch_data in tqdm(self.dev_loader, leave=False, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_outputs = self.model(dev_batch_data['token_ids'],
                                         dev_batch_data['attention_masks'],
                                         dev_batch_data['token_type_ids'],
                                         dev_batch_data['masks'],
                                         dev_batch_data['location']
                                         )

                R = set(get_result(dev_outputs, False))
                T = set(get_result(dev_batch_data['labels'], False))
                X += len(R & T)
                Y += len(R)
                Z += len(T)

            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            return tot_dev_loss, f1

    def test(self, model_path):
        if self.args.model_type == 'gp':
            model, device = load_model_and_parallel(GPTableClassify(self.args), self.args.gpu_ids, model_path)
        elif self.args.model_type == 'cnn':
            model, device = load_model_and_parallel(CnnTableClassify(self.args), self.args.gpu_ids, model_path)
        model.eval()
        # 根据label确定有哪些实体类
        entitys = self.labels
        entitys_to_ids = {v: k for k, v in enumerate(entitys)}
        X, Y, Z = np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15), np.full((len(entitys),), 1e-15)
        X_all, Y_all, Z_all = 1e-15, 1e-15, 1e-15
        with torch.no_grad():
            for dev_batch_data in tqdm(self.dev_loader, ncols=80):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(device)
                dev_outputs = model(dev_batch_data['token_ids'],
                                    dev_batch_data['attention_masks'],
                                    dev_batch_data['token_type_ids'],
                                    dev_batch_data['masks'],
                                    dev_batch_data['location'])

                R = set(get_result(dev_outputs, False))
                T = set(get_result(dev_batch_data['labels'], False))
                X_all += len(R & T)
                Y_all += len(R)
                Z_all += len(T)

                for item in R & T:
                    X[item[1]] += 1
                for item in R:
                    Y[item[1]] += 1
                for item in T:
                    Z[item[1]] += 1
        f1, precision, recall = 2 * X_all / (Y_all + Z_all), X_all / Y_all, X_all / Z_all

        str_log = '\n' + '实体\t' + 'precision\t' + 'pre_count\t' + 'recall\t' + 'true_count\t' + 'f1-score\n'
        str_log += '' \
                   + '全部实体\t' \
                   + '%.4f' % precision + '\t' \
                   + '%.0f' % Y_all + '\t' \
                   + '%.4f' % recall + '\t' \
                   + '%.0f' % Z_all + '\t' \
                   + '%.4f' % f1 + '\n'
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        for entity in entitys:
            str_log += '' \
                       + entity + '\t' \
                       + '%.4f' % precision[entitys_to_ids[entity]] + '\t' \
                       + '%.0f' % Y[entitys_to_ids[entity]] + '\t' \
                       + '%.4f' % recall[entitys_to_ids[entity]] + '\t' \
                       + '%.0f' % Z[entitys_to_ids[entity]] + '\t' \
                       + '%.4f' % f1[entitys_to_ids[entity]] + '\n'
        self.log.info(str_log)
