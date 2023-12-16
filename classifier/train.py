import torch
import torch.nn as nn
import torch.optim as optim
from model import AlexNet
from data_loader import getDataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoches = 20
batch_size = 64
lr = 5e-5
num_classes = 13
base_save_path = './model/'

pretrained_model = './model/AlexNet_21.pth'
use_pretrained = True

train_dataloader, validate_dataloader = getDataLoader(batch_size)
net = AlexNet(num_classes = num_classes, init_weights = True)
if(use_pretrained):
    net.load_state_dict(torch.load(pretrained_model))
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

val_len = len(validate_dataloader)
train_len = len(train_dataloader)

for epoch in range(21, 25):
    # ===================training================
    net.train() # 开启梯度传播和dropout
    running_loss = 0.0 #运行时总loss计算
    time_start = time.perf_counter()

    for step, data in enumerate(train_dataloader):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))				 # 正向传播
        
        loss = loss_function(outputs, labels.to(device)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()

    # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / train_len           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print('%f s' % (time.perf_counter()-time_start))
    
    # ===================validation================
    net.eval() # 关闭梯度传播和dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_dataloader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_len
        
        torch.save(net.state_dict(), base_save_path + 'AlexNet_' + str(epoch) + '.pth')
            
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))