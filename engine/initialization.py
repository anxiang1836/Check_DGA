from engine import BasicModule
from torch import nn


def init_network(model: BasicModule, method: str = "xavier"):
    for name, w in model.named_parameters():
        # 不对Embedding进行初始化
        if 'embedding' not in name:
            # 对weight进行初始化
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            # 对bias进行初始化
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
