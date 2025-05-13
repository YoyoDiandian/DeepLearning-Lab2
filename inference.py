'''
CLIP能力演示

1、对图片做分类
2、对图片求相图片

'''

from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from clip import CLIP
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=CLIP().to(DEVICE) # 模型
model.load_state_dict(torch.load('model.pth'))

model.eval()    # 预测模式

'''
1、对图片分类
'''
image,label=dataset[0]
print('正确分类:',label)
plt.imshow(image.permute(1,2,0))
plt.show()


### TODO: TASK 3: 完成CLIP模型进行预测的代码 (BEGIN)

predicted_label = 0
### TODO: TASK 3: 完成CLIP模型进行预测的代码 (END)

print('CLIP分类:', predicted_label)

'''
2、图像相似度
'''
other_images=[]
other_labels=[]
for i in range(1,101):
    other_image,other_label=dataset[i]
    other_images.append(other_image)
    other_labels.append(other_label)

### TODO: TASK 4: 使用CLIP的image encoder，从other_images里检索和image最相似的5张图像 (BEGIN)


### TODO: TASK 4: 使用CLIP的image encoder，从other_images里检索和image最相似的5张图像 (END)

indexs = [0,0,0,0,0] 
plt.figure(figsize=(15,15))
for i,img_idx in enumerate(indexs):
    plt.subplot(1,5,i+1)
    plt.imshow(other_images[img_idx].permute(1,2,0))
    plt.title(other_labels[img_idx])
    plt.axis('off')
plt.show()