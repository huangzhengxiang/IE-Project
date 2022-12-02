import torch
import torch.nn as nn

class Embed(nn.Module):
    def __init__(self,in_dim,embed_dim,pad_idx):
        super().__init__()
        self.embed = nn.Embedding(in_dim,embed_dim,pad_idx)
        
    def forward(self,x):
        return self.embed(x)

class SelfAttention(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim*2,h_dim),
            nn.Tanh(),
            nn.Linear(h_dim,1),
            nn.Dropout(),
            nn.Softmax(dim=1)
        )
        
    def forward(self,x):
        # [B,S,H]
        A = self.fc(x)
        # A: [B,S,1]
        x = x * A.repeat(1,1,x.shape[2]) # [B,S,H]
        x = torch.sum(x,1,False)
        # x: [B,H]
        return x

class simpleNet(nn.Module):
    def __init__(self,embed_dim,h_dim,out_dim):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=h_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.self_attention= SelfAttention(h_dim=h_dim)
        self.fc = nn.Linear(h_dim*2,out_dim)
        
    def forward(self,x):
        # [B,S,E]
        x, _ = self.gru(x)
        # [B,S,H*2]
        x = self.self_attention(x)
        # [B,H*2]
        x = self.fc(x)
        # [B,O]
        return x
        