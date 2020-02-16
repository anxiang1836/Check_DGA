from torch import nn
from engine.basic_module import BasicModule
from torchtext import data
from engine import BasicConfig


class LocalConfig(BasicConfig):
    def __init__(self):
        super(LocalConfig, self).__init__()
        self.model_name = "biLSTM"
        self.LSTM_hidden_size = 64
        self.LSTM_num_layer = 1


class Model(BasicModule):
    def __init__(self, text_filed: data.Field, config: LocalConfig):
        super(Model, self).__init__()

        self.word_embedding = nn.Embedding(len(text_filed.vocab), config.embed_dim)
        # 给Embedding进行初始化
        self.word_embedding.weight.data.copy_(text_filed.vocab.vectors)

        self.lstm = nn.LSTM(input_size=config.embed_dim, hidden_size=config.LSTM_hidden_size,
                            num_layers=config.LSTM_num_layer)
        self.decoder = nn.Linear(config.LSTM_hidden_size, config.num_classes)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        # print(embeds.shape)
        lstm_out = self.lstm(embeds)[0]
        # print(lstm_out.shape)
        final = lstm_out[-1]
        y = self.decoder(final)
        return y
