# RoPE.py

import torch
import torch.nn as nn

def rotate_half(x):
    x = x.reshape(x.shape[:-1] + (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1):
        # seq_len refers to the length of the sequence in the dimension where RoPE is applied
        seq_len = x.shape[seq_dim]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.outer(t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

def apply_rotary_pos_emb(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=-1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class RotaryPositionalEmbedding2D(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, h, w):
        # h and w are height and width
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=self.inv_freq.device),
                                        torch.arange(w, device=self.inv_freq.device))
        grid = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2).float()  # (h*w, 2)
        sinusoid_inp = torch.matmul(grid, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)  # (h*w, dim)
        emb = emb.unsqueeze(0)  # (1, h*w, dim)
        return emb

def apply_rotary_pos_emb_2d(q, k, pos_emb):
    sin, cos = pos_emb.chunk(2, dim=-1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class RotaryPositionalEmbedding3D(nn.Module):
    def __init__(self, dim, base=500000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, q, t, h, w):
        # t: time/frame dimension
        # h: height
        # w: width
        seq_len = t * h * w
        t_pos = torch.arange(t, device=q.device).float()
        h_pos = torch.arange(h, device=q.device).float()
        w_pos = torch.arange(w, device=q.device).float()

        t_emb = torch.matmul(t_pos, self.inv_freq).view(-1, 1, 1)
        h_emb = torch.matmul(h_pos, self.inv_freq).view(1, -1, 1)
        w_emb = torch.matmul(w_pos, self.inv_freq).view(1, 1, -1)

        # Combine temporal and spatial embeddings
        t_emb = t_emb.repeat(1, h, w)
        h_emb = h_emb.repeat(t, 1, w)
        w_emb = w_emb.repeat(t, h, 1)

        # Stack embeddings
        emb = torch.cat((t_emb, h_emb, w_emb), dim=-1)  # (t*h*w, 3*dim)
        emb = emb.unsqueeze(0)  # (1, t*h*w, 3*dim)

        sin, cos = emb.chunk(2, dim=-1)
        return sin, cos

def apply_rotary_pos_emb_3d(q, k, pos_emb):
    sin, cos = pos_emb
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
