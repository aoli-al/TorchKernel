import math
from torch import nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import functional as f
from torch.nn.modules.utils import _pair
import torch

import fusion_cuda

torch.manual_seed(42)

device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}


# print(fusion_cuda.call_max_pool_upsample_fused())

input = torch.randn(1, 1, 2750, 2048, **kwargs)
r = fusion_cuda.im2col(input, _pair((251, 1)))
print(r)
# print(torch.all(torch.eq(r[0][0], r[1][0])))