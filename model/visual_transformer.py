from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import math

def attention(q,k,v,mask = None):
    """
    Args:
    q: [batch_size, seq_len, d_k]
    k: [batch_size, seq_len, d_k]
    v: [batch_size, seq_len, d_v]
    mask: [batch_size, seq_len]
    Output:
    out: tensor of shape[batch_size, seq_len, d_v]
    """
    batch_size = q.shape[0]
    scale = math.sqrt(k.shape[2])
    att = torch.bmm(q, k.transpose(1, 2)) / scale 
    if mask is not None:
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        att = torch.where(mask > 0.0, att, - math.inf * torch.ones_like(att))
    att = F.softmax(att, 2)
    out = torch.bmm(att, v)
    return out

def create_mask(size1,size2):

    mask = torch.ones(size1,size2)
    mask = torch.triu(mask,diagonal=0)
    return mask

class Head(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Head, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim,bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim,bias=False)

    def forward(self,q,k=None,v=None,mask=None):
        if k is None:
            k = q
        if v is None:
            v = q
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        out = attention(q,k,v,mask)
        return out

class MultiHead(nn.Module):
    def __init__(self, hidden_dim, head_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.heads = nn.ModuleList([Head(hidden_dim, hidden_dim) for _ in range(head_num)])
        self.linear = nn.Linear(hidden_dim //head_num *head_num, hidden_dim)
    
    def forward(self,q,k=None,v=None,mask=None):
        o = [head(q,k,v,mask=mask) for head in self.heads]
        o = torch.cat(o,-1)
        o = self.linear(o)
        return o
        

