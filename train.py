import torch 
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=CLIP().to(DEVICE) # 模型

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

'''
    训练模型
'''
ITER_BATCH_COUNT=100000    # 迭代次数
BATCH_SIZE=64   # 从batch内选出10个不一样的数字
TARGET_COUNT=10 # 共10种数字

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器

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

    loss = ...

    ### ====== TODO: TASK 2: 完成模型损失函数计算的代码（END）
    
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if i%1000==0:
        print('iter:{},loss:{}'.format(i,loss))
        torch.save(model.state_dict(),'.model.pth')
        os.replace('.model.pth','model.pth')