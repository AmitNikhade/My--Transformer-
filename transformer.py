


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



class Embedding(nn.Module):
    def __init__(self,src_vocab_size, model_dimension):
        super().__init__()
        self.m_dim = model_dimension
        self.emb = torch.nn.Embedding(src_vocab_size, model_dimension)
    def forward(self, token_ids_batch):
        embed = self.emb(token_ids_batch)* math.sqrt(self.m_dim)
        return embed

class PositionwiseFeedForwardNet(nn.Module):

    def __init__(self, model_dimension, width_mult=4):
        super().__init__()
 
        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)
        self.norm = nn.LayerNorm(model_dimension)
        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
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

        q2 = torch.cat(q1.split(split_size=self.h_dim // self.num_heads,
         dim=2),dim=0)
        k2 = torch.cat(k1.split(split_size=self.h_dim // self.num_heads,
         dim=2),dim=0)
        v2 = torch.cat(v1.split(split_size=self.h_dim // self.num_heads,
         dim=2),dim=0)
        

        # Multiplication
        outputs = torch.bmm(q2, k2.transpose(2, 1))

        # Scale
        outputs = outputs / (k2.size()[-1] ** 0.5)
        if masked:
        # Key Masking
            key_masks = torch.sign(torch.abs(k).sum(dim=-1))  # (N, T_k)
            key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
            key_masks = key_masks.unsqueeze(1).repeat(1, q.size()[1], 1)  # (h*N, T_q, T_k)

            paddings = torch.ones_like(key_masks) * (-2 ** 32 + 1)
        # print(paddings.shape)
            outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
            outputs = self.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
            query_masks = torch.sign(torch.abs(q).sum(dim=-1))  # (N, T_q)
            query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
            query_masks = query_masks.unsqueeze(-1).repeat(1, 1, k.size()[1])  # (h*N, T_q, T_k)
         
            outputs = outputs * query_masks  # broadcasting. (h*N, T_q, T_k)
        else:
            outputs = self.softmax(outputs)
        # # Dropouts
        outputs = self.dropout(outputs)
        "RuntimeError: Expected tensor to have size 3 at dimension 0, but got size 6 for argument #2 'batch2' (while checking arguments for bmm)"
        # print(outputs.shape, v2.shape)
        # Weighted sum
        outputs = torch.bmm(outputs, v2)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = outputs.split(rs, dim=0)  # (N, T_q, C)
        outputs = torch.cat(outputs, dim=2)

        # # Residual connection
        outputs = outputs + q

        # # Normalize
        outputs = self.norm(outputs)  # (N, T_q, C)

        return outputs

class Transformer(nn.Module):
    def __init__(self,src_vocab_size, model_dimension, n_heads,_num_layers):
        super().__init__()
        
        self._num_layers = _num_layers
        self.Word_Embedding = Embedding(src_vocab_size, model_dimension) 
        self.Pos_Embedding = P_E(model_dimension)
        self.dropout = nn.Dropout(p=0.2)
        self.Encoderlayer = Encoderlayer(model_dimension, n_heads, src_vocab_size)
        self._layers_e = nn.ModuleList([copy.deepcopy(self.Encoderlayer) for _ in range(_num_layers)])
        
        self.Decoderlayer = Decoderlayer(model_dimension, n_heads, src_vocab_size)
        self._layers_d = nn.ModuleList([copy.deepcopy(self.Decoderlayer) for _ in range(_num_layers)])

        # self.mha = mha(model_dimension, n_heads)
        # self.pff = PositionwiseFeedForwardNet(model_dimension)
        self.linear = nn.Linear(model_dimension, src_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self,token_ids_batch, trg_token_ids_batch, inp_mask, trg_mask):
        # Encoder
        embeddings = self.Word_Embedding(token_ids_batch)
        pos_embeddings = self.Pos_Embedding(embeddings)
        embeddings_wp = self.dropout(embeddings+pos_embeddings)
        enc = self.Encoderlayer(embeddings_wp, inp_mask)
        
        # multi_head_attention = self.mha(embeddings_wp,embeddings_wp,embeddings_wp)
        # Position_wiseFeed_Forward = self.pff(multi_head_attention)
        # print(Position_wiseFeed_Forward.shape)
        # return Position_wiseFeed_Forward
        # Decoder
        
        embeddings2 = self.Word_Embedding(trg_token_ids_batch)
        pos_embeddings2 = self.Pos_Embedding(embeddings2)
        embeddings_wp2 = self.dropout(embeddings2+pos_embeddings2)
        dec = self.Decoderlayer(embeddings_wp2, enc, inp_mask, trg_mask)
        # multi_head_attention1 = self.mha(embeddings_wp2,embeddings_wp2,embeddings_wp2)
        # multi_head_attention2 = self.mha(multi_head_attention1,Position_wiseFeed_Forward,Position_wiseFeed_Forward)
        # Position_wiseFeed_Forward2 = self.pff(multi_head_attention2)
        lin = self.linear(dec)
        soft = self.log_softmax(lin)
        # print("hello")
        return soft

class Encoderlayer(nn.Module):
    def __init__(self,model_dimension, n_heads, src_vocab_size):
        super().__init__()
        self.mha = mha(model_dimension, n_heads)
        self.pff = PositionwiseFeedForwardNet(model_dimension)
      

    def forward(self, embeddings_wp, masked):
        multi_head_attention = self.mha(embeddings_wp,embeddings_wp,embeddings_wp, masked)
        Position_wiseFeed_Forward = self.pff(multi_head_attention)
        return Position_wiseFeed_Forward

class Decoderlayer(nn.Module):
    def __init__(self,model_dimension, n_heads, src_vocab_size):
        super().__init__()
        self.mha = mha(model_dimension, n_heads)
        self.pff = PositionwiseFeedForwardNet(model_dimension)
       

    def forward(self, embeddings_wp2, enc, inp_mask, trg_mask):
        multi_head_attention1 = self.mha(embeddings_wp2,embeddings_wp2,embeddings_wp2, masked = inp_mask)
        multi_head_attention2 = self.mha(multi_head_attention1,enc,enc, masked = trg_mask)
        Position_wiseFeed_Forward2 = self.pff(multi_head_attention2)
        return Position_wiseFeed_Forward2


import copy

def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


src_token_ids_batch = torch.randint(1, 10, size=(20, 128))
trg_token_ids_batch = torch.randint(1, 10, size=(20, 128))
# model_dimension = src_token_ids_batch.size()[-1]

model = Transformer(11, 512, 8, 2)
out = model(src_token_ids_batch,trg_token_ids_batch, trg_mask=False, inp_mask=False)
# # print(out.shape)
print(model)

# print(trg_token_ids_batch.shape, trg_token_ids_batch.shape)