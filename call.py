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

  device = torch.device("cuda")
  dtype = torch.float32

  kwargs = {'dtype': dtype,
            'device': device,
            'requires_grad': True}
  batch_norm_input = torch.randn(128, 10000, 576, **kwargs)
  input_max_pool = torch.randn(25, 25, 2560, 1000, **kwargs)
  hist_input = torch.randn(120000000, **kwargs)
  im2col_input = torch.randn(1, 8, 2512, 2048, **kwargs)
  input_upsample = torch.randn(12, 12, 256, 100, **kwargs)
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
  for _ in range(1):
    if idx == 1:
      print(fusion_cuda.histc(im2col_input, hist_input)[0][0])
    if idx == 2:
      print(fusion_cuda.histc_maxpool(hist_input, input_max_pool)[0][0])
    if idx == 3:
      print(fusion_cuda.hist_norm(hist_input, batch_norm_input)[0][0])
    if idx == 4:
      print(fusion_cuda.histc_upsample(hist_input, input_upsample)[0][0])
    if idx == 5:
      print(fusion_cuda.im2col_batchnorm(im2col_input, batch_norm_input)[0][0])
    if idx == 6:
      print(fusion_cuda.im2col_maxpool(im2col_input, input_max_pool)[0][0])
    if idx == 7:
      check(fusion_cuda.max_pool_batch_norm(input_max_pool, batch_norm_input))
    if idx == 8:
      print(fusion_cuda.im2col_upsample(im2col_input, input_upsample)[0][0])
    if idx == 9:
      print(fusion_cuda.call_max_pool_upsample_fused(input_max_pool, input_upsample)[0][0])
    if idx == 0:
      check(fusion_cuda.upsample_batchnorm(input_upsample, batch_norm_input))
    if idx == 11:
      print(fusion_cuda.im2col_maxpool_batchnorm()[0])
    if idx == 12:
      print(fusion_cuda.max_hist_norm()[0])
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=None)

if len(sys.argv) == 2:
    run(int(sys.argv[1]))
else:
    for i in range(10):
        run(i)



