# 表格单元格分类
本项目用于解决表格单元格的分类问题，使用前请考虑如何把手上的表格项目转换到任意的分类问题上，理论上已经可以满足绝大部分的需求。


## 环境依赖
```
conda create -n yourname python==3.10.0
conda activate yourname / source activate yourname
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install transformers==4.29.2
pip install rich==12.5.1
pip install flask
pip install gevent
```

## 项目结构
```html
|-- config.py
|-- README.md
|-- server.py
|-- train.py
|-- data
|   |-- PaiMai
|       |-- dev_data.pkl
|       |-- labels.json
|       |-- preprocess.py
|       |-- train_data.pkl
|       |-- 样本1.json
|-- utils
    |-- adversarial_training.py
    |-- base_model.py
    |-- functions.py
    |-- models.py
    |-- train_models.py
    |-- table_sample_make_v2.cs
```

## 快速开始

```html
一、训练
1.参考table_sample_make_v2.cs内的GetTabelSample函数 制造样本，并导入Label_Studio平台
2.标注后导出标注结果，模仿拍卖项目【data/PaiMai】，新建项目，拷贝preprocess.py
3.执行preprocess.py
4.修改config内data_dir参数
5.修改train.py脚本，新增一个if分支，然后执行train.py
二、服务
1.修改server.py内的model_name变量
2.运行server.py即可
三、升级
参考docker升级手册和Dockerfile即可，可在107.46上部署升级
```

## 注意事项

```html
1.支持解决多标签分类问题，各函数已做兼容，但无超参控制，如有需求，需联系管理员解决
2.项目开发的时候，请尽量降低标注需求（能我们程序批量处理的，尽量批量处理，体谅一下业务的难处~）
3.默认参数下，需要20G左右的显存，如果在16G显存的机器上训练，建议将batch_size调整为2或4，在8G显存的机器上训练，建议将fine_tuning参数调整为False（训练显存会降到4个G以内），或者改用小模型，而不是bert-base，简单业务场景，建议robert-small
4.如果想获得完美的外推性，需设置AbsoluteEncoding为False，但是需要更大量的样本支撑，一般情况下设置为True，外推性也足以应对绝大部分场景
5.理论上设置表格为10*10，至多20*10即可，考虑到显存问题，若想要获得更好的外推性，请联系管理员解决
```

## label_studio平台标签

```html
标签体系范例
<View>
  <Labels name="ner" toName="text">
    <Label value="key" background="#020f00"/>
    <Label value="value" background="blue"/>
    <Label value="content" background="red"/>
    <Label value="title" background="yellow"/>
  </Labels>

  <View style="border: 1px solid #CCC; border-radius: 10px; padding: 5px">
    <HyperText name="text" value="$html"/>
  </View>
</View>
```

## 更新日志
 - 2024-01-18：使用了新的标注定位方案，增加了示例PaiMai项目
 - 2024-01-18：同时更新了models.py train.py server.py 以此支持示例项目 同时放弃了对旧项目的支持 故table_sample_make.cs文件已过时 下次更新
 - 2024-01-19：增加了table_sample_make_v2.cs,table_sample_make.cs以后仅作参考
