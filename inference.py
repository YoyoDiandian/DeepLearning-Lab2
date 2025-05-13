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

with torch.no_grad():
    # 为每个数字(0-9)生成文本嵌入
    all_text_embeddings = model.text_enc(torch.arange(10).to(DEVICE))
    # 获取图像嵌入
    image_embedding = model.img_enc(image.unsqueeze(0).to(DEVICE))

    # 规范化嵌入
    image_embedding = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    all_text_embeddings = all_text_embeddings / all_text_embeddings.norm(dim=1, keepdim=True)

    # 计算相似度
    similarity = torch.mm(image_embedding, all_text_embeddings.t())
    print(similarity)

    # 选择相似度最高的作为预测结果
    predicted_label = similarity.argmax(dim=1).item()

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

with torch.no_grad():
    # 将其他图像转换为张量
    other_images_tensor = torch.stack(other_images).to(DEVICE)

    # 使用图像编码器获取图像嵌入
    query_embedding = model.img_enc(image.unsqueeze(0).to(DEVICE))
    other_embeddings = model.img_enc(other_images_tensor)

    # 规范化嵌入
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    other_embeddings = other_embeddings / other_embeddings.norm(dim=1, keepdim=True)

    # 计算相似度
    similarities = torch.mm(query_embedding, other_embeddings.t())

    # 获取相似度最高的5个索引
    indexs = similarities[0].topk(5).indices.cpu().numpy().tolist()

### TODO: TASK 4: 使用CLIP的image encoder，从other_images里检索和image最相似的5张图像 (END)

plt.figure(figsize=(15,15))
for i,img_idx in enumerate(indexs):
    plt.subplot(1,5,i+1)
    plt.imshow(other_images[img_idx].permute(1,2,0))
    plt.title(other_labels[img_idx])
    plt.axis('off')
plt.savefig(f"output/similarity{label}.pdf")
plt.show()