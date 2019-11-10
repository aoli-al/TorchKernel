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


def check(kernels):
  half = len(kernels) // 2
  for i in range(half):
    print(torch.all(torch.eq(kernels[i], kernels[i+half])))


# print(fusion_cuda.call_max_pool_upsample_fused()[0][0])
# check(fusion_cuda.im2col_maxpool_batchnorm())
# print(fusion_cuda.im2col_upsample()[0][0])
# check(fusion_cuda.im2col_batchnorm())
# print(fusion_cuda.call_max_pool_upsample_fused()[0][0][0])
# check(fusion_cuda.max_pool_batch_norm())
check(fusion_cuda.upsample_batchnorm())
# i = torch.randn(20, 25600000, **kwargs)
# print(fusion_cuda.dropout_batchnorm(i)[0][0])

# r = fusion_cuda.im2col(input, _pair((251, 1)))
# r = fusion_cuda.max_pool_batch_norm()
# print(r)
# print(torch.all(torch.eq(r[0], r[3])))
# print(torch.all(torch.eq(r[1], r[4])))
# print(torch.all(torch.eq(r[2], r[5])))