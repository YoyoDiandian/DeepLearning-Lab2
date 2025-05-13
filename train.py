import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

def main():
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

    dataset=MNIST() # 数据集

    model=CLIP().to(DEVICE) # 模型

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)   # 优化器
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5, verbose=True)

    '''
        训练模型
    '''
    ITER_BATCH_COUNT=100000    # 迭代次数
    BATCH_SIZE=64   # 从batch内选出10个不一样的数字
    TARGET_COUNT=10 # 共10种数字

    # 修改worker数量为0可以避免多进程问题
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)    # 数据加载器

    for i in range(ITER_BATCH_COUNT):
        while True:
            imgs,labels=next(iter(dataloader))
            if torch.unique(labels).shape[0]<TARGET_COUNT:  # 未覆盖10种数字
                continue
            # 挑选出10个数字
            target=set()
            indexes=[]
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indexes.append(j)
                if len(target)==TARGET_COUNT:
                    break
            imgs=imgs[indexes]
            labels=labels[indexes]
            break

        ### ====== TODO: TASK2: 完成模型损失函数计算的代码（BEGIN）

        # 通过模型获取相似度矩阵
        logits = model(imgs.to(DEVICE), labels.to(DEVICE))

        # 对比学习中，第i个图像应该与第i个文本匹配
        labels_one_hot = torch.arange(TARGET_COUNT, device=DEVICE)

        # 计算双向对比损失（图像到文本 + 文本到图像）
        loss = (F.cross_entropy(logits, labels_one_hot) +
                F.cross_entropy(logits.t(), labels_one_hot)) / 2

        ### ====== TODO: TASK 2: 完成模型损失函数计算的代码（END）

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000==0:
            print('iter:{},loss:{}'.format(i,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
            scheduler.step(loss)  # 根据损失值调整学习率

# 添加温度参数到CLIP的forward方法
def forward(self, img_x, text_x, temperature=0.07):
    img_features = self.img_enc(img_x)
    text_features = self.text_enc(text_x)
    
    # 规范化
    img_features = img_features / img_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # 添加温度缩放
    logits = torch.mm(img_features, text_features.t()) / temperature
    
    return logits

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 在Windows上需要
    main()