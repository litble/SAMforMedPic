import torch.nn as nn
import torch
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=13, init_weights=False):
        super(AlexNet, self).__init__()
        # Conv：(width - kernel_size + 2* padding)/stride + 1，向下取整
        # Pool：(width - kernel_size)/stride + 1，向下取整
        self.features = nn.Sequential( #[2, 512, 512]
            nn.Conv2d(2, 16, kernel_size=11, stride=4, padding=1), #[16, 126, 126]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #[16, 62, 62]
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1), #[32, 30, 30]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #[32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #[32, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        if(init_weights):
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=0) # 展平
        x = self.classifier(x)
        return x

    # 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',   # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    # 初始化偏重为0
            elif isinstance(m, nn.Linear):            # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)    # 正态分布初始化
                nn.init.constant_(m.bias, 0)          # 初始化偏重为0