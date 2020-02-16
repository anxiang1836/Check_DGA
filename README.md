# DGA主机域名检测

> 项目概要：通过深度学习的方法，构建DGA域名检测的识别算法，进而加强网络安全防护能力。
>
> 数据来源：
>
> - 正常域名：[Alexa Top 100W](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip)
> - 可疑域名：[360DGA](http://data.netlab.360.com/dga/)
>
> DL算法：TextRNN，等

## 僵尸网络

僵尸网络 Botnet 是指采用一种或多种传播手段，将大量主机感染bot程序（[僵尸程序](https://baike.baidu.com/item/僵尸程序)）病毒，从而在控制者和被感染[主机](https://baike.baidu.com/item/主机/455151)之间所形成的一个可一对多控制的网络  。

攻击者通过各种途径传播僵尸程序感染互联网上的大量主机，而被感染的主机将通过一个[控制信道](https://baike.baidu.com/item/控制信道/5623827)接收攻击者的指令，组成一个僵尸网络。之所以用僵尸网络这个名字，是为了更形象地让人们认识到这类危害的特点：众多的计算机在不知不觉中如同中国古老传说中的僵尸群一样被人驱赶和指挥着，成为被人利用的一种工具。

## 僵尸网络和DGA的关系

僵尸网络中用来控制主机，负责处理信息，下发任务的中心机器，也称C&C服务器（control & command）。

> 一个成熟的僵尸网络往往具有多个C&C服务 器。 对于安全人员来说，查杀了僵尸网络的C&c服务器也就等于毁灭而这个僵尸网络，那么作为安全对抗，僵尸网络的缔造者也会想尽办法对其进行隐藏和保护其主控端。

那么对于固定的C&C服务器域名，安全人员一般来说很容易对其进行查杀，因此，基于DGA方法僵尸网络也就产生了。

DGA全称：**随机域名生成算法**，是指使用主控端和被控端协商好的一种基于随机算法的域名生成协议，简单来说就是生成一个随机字符串来作为域名并进行注册，将其作为C&C服务器的域名并不定时经常性更换。由于具备强随机性，短时效性，通过DGA生成的域名往往在查杀上更具备难度。

## 项目结构

```bash
.
├── data_process    # 数据处理部分（利用torchtext）
│   ├── __init__.py
│   └── dataset.py
├── engine          # 模型驱动组成部分
│   ├── __init__.py
│   ├── basic_config.py      # 全局参数
│   ├── basic_module.py      # 封装了torch.nn.Module
│   ├── initialization.py    # 网络初始化
│   └── train_eval.py        # 网络train和eval
├── jupyter         # 源数据准备的notebook
│   ├── 01-prepare_data.ipynb
│   ├── 02-EDA.ipynb
│   ├── origin_data    # 原始数据
│   │   ├── 2019-alexa-top-100w.xlsx
│   │   ├── 360dga.txt
│   └── pkl_data       # 预处理后的数据
│       ├── alexa_2019
│       ├── dga_360
│       ├── test_data
│       ├── train_data
│       └── val_data
├── models     # 模型部分
│   ├── __init__.py
│   └── biLSTM.py
├── paper      # 有关DGA预测的论文
├── utils      # 工具部分
│   ├── __init__.py
│   └── common_tools.py
└── main.py    # 程序运行主函数
```


