import argparse
import torch
import time
from engine import BasicConfig, BasicModule, init_network, train
from importlib import import_module
from data_process import DGA2019
from torchtext import data
from utils import get_time_dif
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='DGA Checking Classification')
parser.add_argument('--models', type=str, required=True,
                    help='choose a models: biLSTM,.......')
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.models

    # Step0: Load Config
    x = import_module("models." + model_name)
    config = x.LocalConfig()  # type:BasicConfig

    # Step1: Define torchText Field
    print("Start Loading Data....")
    start_time = time.time()

    def tokenize(x):
        return [c for c in x]

    TEXT_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL_field = data.LabelField(sequential=False, use_vocab=False)

    # Step2: Load data
    train_ds = DGA2019(config.train_path, TEXT_field,LABEL_field,test_mode=False)
    train_iter = data.BucketIterator(dataset=train_ds, batch_size=config.batch_size,
                                     shuffle=True, sort_within_batch=False, repeat=False, device=config.device)

    valid_ds = DGA2019(config.val_path,TEXT_field,LABEL_field, test_mode=False)
    valid_iter = data.BucketIterator(dataset=valid_ds, batch_size=config.batch_size,
                                     shuffle=True, sort_within_batch=False, repeat=False, device=config.device)

    # Step3: Init vocab and Embedding（by random）
    TEXT_field.build_vocab(train_ds)
    # LABEL_field.build_vocab(train_ds)
    matrix = torch.randn(len(TEXT_field.vocab), config.embed_dim)
    matrix.cuda(config.device)
    TEXT_field.vocab.set_vectors(TEXT_field.vocab.stoi, matrix, config.embed_dim)

    time_dif = get_time_dif(start_time)
    print("Loading data Time usage:", time_dif)

    # Step4: Train Model
    model = x.Model(TEXT_field, config)  # type:BasicModule
    model = model.to(config.device)
    writer = SummaryWriter(config.log_path + time.strftime('%m-%d_%H.%M', time.localtime()))
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, valid_iter, writer)
