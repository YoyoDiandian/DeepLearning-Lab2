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
        
        # return logits
        return img_features@text_features.T
    
if __name__=='__main__':
    clip=CLIP()
    img_x=torch.randn(5,1,28,28)
    text_x=torch.randint(0,10,(5,))
    logits=clip(img_x,text_x)
    print(logits.shape)