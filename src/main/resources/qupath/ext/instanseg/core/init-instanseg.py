from instanseg import InstanSeg
from appose import NDArray
import torch
import numpy as np

model = torch.jit.load("%s")
ins = InstanSeg(model, device="%s")
