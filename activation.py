import torch.nn as nn
import torch.nn.functional as F

class GeGLU(nn.Module):
    def __init__(self, embed_size):
        super(GeGLU, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        return F.gelu(self.fc1(x)) * self.fc2(x)