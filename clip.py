from torch import nn 
import torch 
from img_encoder import ImgEncoder
from text_encoder import TextEncoder

class CLIP(nn.Module):
    def __init__(self,):
        super().__init__()
        ### TODO: TASK 0: 完成CLIP模型的初始化
        self.img_enc = ImgEncoder()
        self.text_enc = TextEncoder()

    def forward(self, img_x, text_x):
        ### TODO: TASK 1: 完成CLIP模型的Forward代码
        img_features = self.img_enc(img_x)  # [batch_size, embed_dim]
        text_features = self.text_enc(text_x)  # [batch_size, embed_dim]
        
        # 计算余弦相似度矩阵
        # L2标准化
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # 计算余弦相似度矩阵 (N×N)
        logits = torch.mm(img_features, text_features.t())
        
        return logits
    
if __name__=='__main__':
    clip=CLIP()
    img_x=torch.randn(5,1,28,28)
    text_x=torch.randint(0,10,(5,))
    logits=clip(img_x,text_x)
    print(logits.shape)