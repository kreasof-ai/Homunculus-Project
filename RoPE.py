import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

def apply_rotary_pos_emb(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=-1)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, h, w):
        grid = torch.meshgrid(torch.arange(h, device=self.inv_freq.device), torch.arange(w, device=self.inv_freq.device))
        grid = torch.stack(grid, dim=-1).float()
        sinusoid_inp = torch.einsum("ij,k->ijk", grid, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        emb = emb.permute(2, 0, 1)
        return emb

def apply_rotary_pos_emb_2d(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)