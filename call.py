import math
from torch import nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.nn import functional as f
from torch.nn.modules.utils import _pair
import torch

import fusion_cuda

torch.backends.cudnn.enabled = False

torch.manual_seed(42)

device = torch.device("cuda")
dtype = torch.float32

kwargs = {'dtype': dtype,
          'device': device,
          'requires_grad': True}


def check(kernels):
  half = len(kernels) // 2
  for i in range(half):
    print(torch.all(torch.eq(kernels[i], kernels[i+half])))


# print(fusion_cuda.call_max_pool_upsample_fused()[0][0])
# print(fusion_cuda.im2col_upsample()[0][0])
# print(fusion_cuda.im2col_batchnorm()[0][0])
# print(fusion_cuda.im2col_maxpool()[0][0])
# check(fusion_cuda.max_pool_batch_norm())
check(fusion_cuda.upsample_batchnorm())
torch.cuda.synchronize(device=None)
