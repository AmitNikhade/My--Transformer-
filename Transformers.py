# coding=utf-8


""""" 
amitnikhadeofficial@gmail.com 
""""" 


__author__ = "Amit Nikhade"
__status__ = "Development"


# A simple Implemetaion of Transformers with PyTorch

import numpy as np
import torch
import math
import torch.nn as nn

class P_E(nn.Module):
    def __init__(self, model_dimension, expected_max_sequence_length=5000):
        super().__init__()
        pos_en = np.array([
            [pos / np.power(10000, 2 * (i // 2) / model_dimension) for i in range(model_dimension)] 
            if pos != 0 else np.zeros(model_dimension) 
                for pos in range(expected_max_sequence_length)])
        pos_en[1:, 0::2] = np.sin(pos_en[1:, 0::2]) 
        pos_en[1:, 1::2] = np.cos(pos_en[1:, 1::2])

        self.pos_en = torch.FloatTensor(pos_en)

    def forward(self, embeddings):

        pe = self.pos_en[:embeddings.shape[1]]
        return pe



class PFF(nn.Module):

    def __init__(self, model_dimension, width_mult=4):
        super().__init__()
 
        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        

    def forward(self, representations_batch):
        return self.norm(self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))+representations_batch)


class mha(nn.Module):
    def __init__(self, h_dim, n_heads):
        super().__init__()
        self.h_dim=h_dim
        self.linear = nn.Linear(h_dim, h_dim)
        self.num_heads = n_heads
        self.norm = nn.LayerNorm(h_dim)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k ,v, masked):
        rs = q.size()[0]
        q1= nn.ReLU()(self.linear(q))
        k1= nn.ReLU()(self.linear(k))
        v1= nn.ReLU()(self.linear(v))

        q2 = torch.cat(torch.chunk(q1, self.num_heads, dim=2), dim=0)  
        k2 = torch.cat(torch.chunk(k1, self.num_heads, dim=2), dim=0)  
        v2 = torch.cat(torch.chunk(v1, self.num_heads, dim=2), dim=0)  
        
        outputs = torch.bmm(q2, k2.transpose(2, 1))

        outputs = outputs / (k2.size()[-1] ** 0.5)
        if masked:
            k_masks = torch.sign(torch.abs(k).sum(dim=-1))  
            k_masks = k_masks.repeat(self.num_heads, 1)  
            k_masks = k_masks.unsqueeze(1).repeat(1, q.size()[1], 1)  

            paddings = torch.ones_like(k_masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.eq(k_masks, 0), paddings, outputs)  
            outputs = self.softmax(outputs)  
            q_masks = torch.sign(torch.abs(q).sum(dim=-1))  
            q_masks = q_masks.repeat(self.num_heads, 1)  
            q_masks = q_masks.unsqueeze(-1).repeat(1, 1, k.size()[1])  
         
            outputs = outputs * q_masks  
        else:
            outputs = self.softmax(outputs)
        outputs = self.dropout(outputs)
  
        outputs = torch.bmm(outputs, v2) 
        outputs = outputs.split(rs, dim=0)  
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs + q
        outputs = self.norm(outputs)  

        return outputs

class Transformer(nn.Module):
    def __init__(self,inp_vocab, model_dimension, n_heads,_num_layers):
        super().__init__()
        
        self._num_layers = _num_layers
        self.emb = torch.nn.Embedding(inp_vocab, model_dimension)
        self.Pos_Embedding = P_E(model_dimension)
        self.dropout = nn.Dropout(p=0.2)
        self.Encoder = Encoder(model_dimension, n_heads)

        self._layers_e = nn.ModuleList()
        for i in range(_num_layers):
            layer = self.Encoder 
            self._layers_e.append(layer)

        self.Decoder = Decoder(model_dimension, n_heads)

        self._layers_d = nn.ModuleList()
        for i in range(_num_layers):
            layer = self.Decoder 
            self._layers_d.append(layer)

        self.linear = nn.Linear(model_dimension, inp_vocab)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x, y, x_mask, y_mask):
    
        embeddings = self.emb(x)
        pos_embeddings = self.Pos_Embedding(embeddings)
        embeddings_wp = self.dropout(embeddings+pos_embeddings)
        enc = self.Encoder(embeddings_wp, x_mask)
    
        embeddings2 = self.emb(y)
        pos_embeddings2 = self.Pos_Embedding(embeddings2)
        embeddings_wp2 = self.dropout(embeddings2+pos_embeddings2)
        dec = self.Decoder(embeddings_wp2, enc, x_mask, y_mask)
     
        lin = self.linear(dec)
        soft = self.log_softmax(lin)

        return soft

class Encoder(nn.Module):
    def __init__(self,model_dimension, n_heads):
        super().__init__()
        self.mha = mha(model_dimension, n_heads)
        self.pff = PFF(model_dimension)
      

    def forward(self, embeddings_wp, masked):
        multi_head_attention = self.mha(embeddings_wp,embeddings_wp,embeddings_wp, masked)
        Position_wiseFeed_Forward = self.pff(multi_head_attention)
        return Position_wiseFeed_Forward

class Decoder(nn.Module):
    def __init__(self,model_dimension, n_heads):
        super().__init__()
        self.mha = mha(model_dimension, n_heads)
        self.pff = PFF(model_dimension)
       

    def forward(self, embeddings_wp2, enc, x_mask, y_mask):
        multi_head_attention1 = self.mha(embeddings_wp2,embeddings_wp2,embeddings_wp2, masked = x_mask)
        multi_head_attention2 = self.mha(multi_head_attention1,enc,enc, masked = y_mask)
        Position_wiseFeed_Forward2 = self.pff(multi_head_attention2)
        return Position_wiseFeed_Forward2

x = torch.randint(1, 20, size=(20, 128))
y = torch.randint(1, 20, size=(20, 128))

model = Transformer(21, 512, 8, 2)
output = model(x,y, y_mask=False, x_mask=False)
print(model)
