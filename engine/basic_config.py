import torch
import warnings


class BasicConfig(object):
    """配置参数"""
    def __init__(self):
        self.model_name = ""
        self.train_path = "jupyter/pkl_data/train_data"  # 训练集
        self.val_path = "jupyter/pkl_data/val_data"  # 验证集
        self.test_path = "jupyter/pkl_data/test_data"  # 测试集

        # self.vocab_path = dataset + '/data_process/vocab.pkl'     # 词表
        self.load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载
        self.log_path = "log/" + self.model_name
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data_process/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 3  # 若超过3个epoch效果还没提升，则提前结束训练
        self.num_classes = 2  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率
        self.embed_dim = 300  # 字向量维度

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.iteritems():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k, getattr(self, k))
