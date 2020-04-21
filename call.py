from multiprocessing import Pool
import sys


def run(idx):
  import torch

  import fusion_cuda
  import sys
  import math
  from torch import nn
  from torch.autograd import Function
  from torch.nn.parameter import Parameter
  from torch.nn import functional as f
  from torch.nn.modules.utils import _pair
  torch.backends.cudnn.enabled = False

  torch.manual_seed(42)
  print(idx, file=sys.stderr)

  device = torch.device("cuda")
  dtype = torch.float32

  kwargs = {'dtype': dtype,
            'device': device,
            'requires_grad': True}
  def batch_norm_input(): 
    c = range(-64, 64)
    for x in c:
      yield torch.randn(128, 10000, 100+x, **kwargs)
  def maxpool_input():
    c = range(-40, 40)
    for x in c:
      yield torch.randn(1, 80 + x, 2560, 1000, **kwargs)
  def hist_input():
    c = range(-25, 50)
    for x in c:
      yield torch.randn((50 + x)* 100000, **kwargs)
  def im2col_input():
    pass
  def upsample_input():
    c = range(12, 32)
    for x in c:
      yield torch.randn(1, x, 256, 100, **kwargs)

  input_batchnorm = torch.randn(128, 10000, 100 + 64, **kwargs)
  input_max_pool = torch.randn(1, 80, 2560, 1000, **kwargs)
  input_hist = torch.randn((50)* 100000, **kwargs)
  im2col_input = torch.randn(1, 1, 2512, 2048, **kwargs)
  input_upsample = torch.randn(1, 20, 256, 100, **kwargs)
  if idx == 0 or idx == 12 or idx == 11:
   lstm = nn.LSTM(3, 3).cuda()
   i = torch.randn(1, 3, **kwargs)
   hidden = (torch.randn(1, 1, 3, **kwargs),
             torch.randn(1, 1, 3, **kwargs))
   for _ in range(10000):
     out, hidden = lstm(i.view(1, 1, -1), hidden)

  def check(kernels):
    half = len(kernels) // 2
    for i in range(half):
      print(torch.all(torch.eq(kernels[i], kernels[i+half])))
  for _ in range(5):
    if idx == 1:
      for i in hist_input():
        print(fusion_cuda.histc(im2col_input, i)[0][0])
    if idx == 2:
      for i in maxpool_input():
        print(fusion_cuda.histc_maxpool(input_hist, i)[0][0])
    if idx == 3:
      # print(fusion_cuda.hist_norm(input_hist, input_batchnorm)[0][0])
      for i in batch_norm_input():
        print(fusion_cuda.hist_norm(input_hist, i)[0][0])
    if idx == 4:
      for i in upsample_input():
        print(fusion_cuda.histc_upsample(input_hist, i)[0][0])
      # print(fusion_cuda.histc_upsample(input_hist, input_upsample)[0][0])
    if idx == 5:
      # print(fusion_cuda.im2col_batchnorm(im2col_input, batch_norm_input)[0][0])
      for i in batch_norm_input():
        print(fusion_cuda.im2col_batchnorm(im2col_input, i)[0][0])
    if idx == 6:
      # print(fusion_cuda.im2col_maxpool(im2col_input, input_max_pool)[0][0])
      for i in maxpool_input():
        print(fusion_cuda.im2col_maxpool(im2col_input, i)[0][0])
    if idx == 7:
      for i in batch_norm_input():
        check(fusion_cuda.max_pool_batch_norm(input_max_pool, i))
    if idx == 8:
      for i in upsample_input():
        print(fusion_cuda.im2col_upsample(im2col_input, i)[0][0])
    if idx == 9:
      for i in upsample_input():
        print(fusion_cuda.call_max_pool_upsample_fused(input_max_pool, i)[0][0])
    if idx == 0:
      for i in batch_norm_input():
        check(fusion_cuda.upsample_batchnorm(input_upsample, i))
      # check(fusion_cuda.upsample_batchnorm(input_upsample, batch_norm_input))
    if idx == 11:
      print(fusion_cuda.im2col_maxpool_batchnorm()[0])
    if idx == 12:
      print(fusion_cuda.max_hist_norm()[0])
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=None)

if len(sys.argv) == 2:
    run(int(sys.argv[1]))
else:
    for i in range(5, 10):
        run(i)



