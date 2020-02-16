from engine import BasicConfig, BasicModule
from torchtext.data import BucketIterator
import time
from torch.optim import Adam
import torch.nn.functional as F
import torch
from sklearn import metrics
import numpy as np
from utils import get_time_dif
from tensorboardX import SummaryWriter


def train(config: BasicConfig, model: BasicModule, train_iter: BucketIterator, val_iter: BucketIterator,
          writer: SummaryWriter):
    start_time = time.time()
    model.train()
    # 使用Adam优化函数
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    val_best_loss = float("inf")
    last_improve_epoch = 0

    for epoch in range(config.num_epochs):
        print("Epoch-",epoch+1)
        for i, (trains_batch, labels_batch) in enumerate(train_iter):
            outputs = model(trains_batch)
            # 当前batch梯度归零重新记录
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels_batch)
            # 将loss反向传播回去
            loss.backward()
            optimizer.step()

            # 如果当前batch是Epoch中的最后一个，则在验证集上测试效果
            if i == len(train_iter) - 1:
                true = labels_batch.data.cpu()  # 真值
                pred = torch.max(outputs.data, 1)[1].cpu()  # 预测值

                # 使用sklearn的acc_score计算准确率
                train_acc = metrics.accuracy_score(true, pred)
                # 在验证集上验证效果
                val_acc, val_loss = __evaluate(model, val_iter)
                if val_loss < val_best_loss:
                    last_improve_epoch = epoch
                time_dif = get_time_dif(start_time)
                msg = "Epoch:{0:>2d}/{1} Train_loss:{2:.5f} Train_acc:{3:.5f} Val_loss:{4:.5f} Val_acc:{5:5f} Time:{6}"
                print(msg.format(epoch + 1, config.num_epochs, loss.item(), train_acc, val_loss, val_acc, time_dif))
                # 写入log
                writer.add_scalar("loss/train", loss.item(), epoch+1)
                writer.add_scalar("loss/val", val_loss, epoch+1)
                writer.add_scalar("acc/train", train_acc, epoch+1)
                writer.add_scalar("acc/val", val_acc, epoch+1)
                # 因为在上面切换到了model.eval()模式，所以验证完了还需要再切换回来
                model.train()
        # Early-stopping的判断
        if epoch - last_improve_epoch > config.require_improvement:
            print("No improvement for {} Epoches,early-stopping...")
            break


def __evaluate(model: BasicModule, data_iter: BucketIterator, test_mode=False):
    model.eval()
    loss_all = 0  # 验证集上总的loss
    predict_all = np.array([], dtype=int)  # 验证集全部的预测值
    label_all = np.array([], dtype=int)  # 验证集全部的真值

    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_all += loss
            labels = labels.data.cpu().numpy()
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()

            predict_all = np.append(predict_all, pred)
            label_all = np.append(label_all, labels)

    acc = metrics.accuracy_score(predict_all, label_all)
    loss = loss_all / len(data_iter)
    if test_mode:
        confusion = metrics.confusion_matrix(label_all, predict_all)
        return acc, loss, confusion
    else:
        return acc, loss
