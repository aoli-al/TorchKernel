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
  bn_ = 320
  mp_ = 512 

  torch.manual_seed(42)
  print(idx, file=sys.stderr)

  device = torch.device("cuda")
  dtype = torch.float32

  kwargs = {'dtype': dtype,
            'device': device,
            'requires_grad': True}
  def batch_norm_input(): 
    c = range(-128, 128, 16)
    for x in c:
      yield torch.randn(128, 10000, 256+x, **kwargs)
  def maxpool_input():
    c = range(-100, 100, 10)
    for x in c:
      yield torch.randn(1, 220 + x, 2560, 1000, **kwargs)
  def hist_input():
    c = range(-100, 50, 16)
    for x in c:
      yield torch.randn((512 - 32 + x)* 100000, **kwargs)
  def im2col_input():
    pass
  def upsample_input():
    c = range(32, 96, 2)
    for x in c:
      yield torch.randn(1, x, 256, 100, **kwargs)

  input_batchnorm = torch.randn(128, 10000, 256 - 128, **kwargs)
  input_max_pool = torch.randn(1, 100, 2560, 1000, **kwargs)
  input_hist = torch.randn((512 - 32)* 100000, **kwargs)
  im2col_input = torch.randn(1, 1, 2512, 2048, **kwargs)
  input_upsample = torch.randn(1, 64, 256, 100, **kwargs)
  #  if idx == 0 or idx == 12 or idx == 11:
   #  lstm = nn.LSTM(3, 3).cuda()
   #  i = torch.randn(1, 3, **kwargs)
   #  hidden = (torch.randn(1, 1, 3, **kwargs),
             #  torch.randn(1, 1, 3, **kwargs))
   #  for _ in range(10000):
     #  out, hidden = lstm(i.view(1, 1, -1), hidden)

  def check(kernels):
    half = len(kernels) // 2
    for i in range(half):
      print(torch.all(torch.eq(kernels[i], kernels[i+half])))
  for _ in range(1):
    if idx == 1:
      for i in batch_norm_input():
        result = fusion_cuda.upsample_batchnorm(input_upsample, i)
        del result
        del i
    if idx == 2:
      for i in maxpool_input():
        result = fusion_cuda.histc_maxpool(input_hist, i)
        del result
        del i

    if idx == 3:
      # print(fusion_cuda.hist_norm(input_hist, input_batchnorm)[0][0])
      for i in batch_norm_input():
        result = fusion_cuda.hist_norm(input_hist, i)
        del result
        del i
    if idx == 4:
      for i in upsample_input():
        result = fusion_cuda.histc_upsample(input_hist, i)
        del result
        del i
      # print(fusion_cuda.histc_upsample(input_hist, input_upsample)[0][0])
    if idx == 5:
      # print(fusion_cuda.im2col_batchnorm(im2col_input, batch_norm_input)[0][0])

      c = range(-120, 128, 16)
      for x in c:
        i = torch.randn(128, 10000, 128+x, **kwargs)
        result = fusion_cuda.im2col_batchnorm(im2col_input, i)
        del result
        del i
    if idx == 6:
      # print(fusion_cuda.im2col_maxpool(im2col_input, input_max_pool)[0][0])
      c = range(-100, 100, 10)
      for x in c:
        i = torch.randn(1, 120 + x, 2560, 1000, **kwargs)
        result = fusion_cuda.im2col_maxpool(im2col_input, i)
        del result
        del i
    if idx == 7:
      #  input_max_pool = torch.randn(1, 528, 16, 16, **kwargs)
      #  fusion_cuda.max_pool_batch_norm(input_max_pool, input_batchnorm)
      for i in batch_norm_input():
       result = fusion_cuda.max_pool_batch_norm(input_max_pool, i)
       del i 
       del result
    if idx == 8:
      for i in upsample_input():
        result = fusion_cuda.im2col_upsample(im2col_input, i)
        del i
        del result
    if idx == 9:
      for i in upsample_input():
        result = fusion_cuda.call_max_pool_upsample_fused(input_max_pool, i)
        del i 
        del result
    if idx == 0:
      c = range(-100, 50, 16)
      for x in c:
        i = torch.randn((128 + x)* 100000, **kwargs)
        result = fusion_cuda.histc(im2col_input, i)
        del i 
        del result
      # check(fusion_cuda.upsample_batchnorm(input_upsample, batch_norm_input))
    if idx == 11:
      print(fusion_cuda.im2col_maxpool_batchnorm()[0])
    if idx == 12:
      print(fusion_cuda.max_hist_norm()[0])
    if idx == 13:
      input_max_pool = torch.randn(1, mp_, 16, 16, **kwargs)
      m = nn.BatchNorm2d(bn_)
      m.to('cuda')
      result = m(input_batchnorm)
      mxp = nn.MaxPool2d(3, stride=2, padding=1, return_indices=True)
      mxp.to('cuda')
      r2 = mxp(input_max_pool)
      fusion_cuda.batchnorm_maxpooling_backward(
          torch.randn(1,mp_,8,8, **kwargs),
          input_max_pool, r2[1],
          torch.randn(input_batchnorm.shape, **kwargs), input_batchnorm, m.weight,
          m.running_mean, m.running_var)
    if idx == 14:
      m = nn.BatchNorm2d(bn_)
      m.to('cuda')
      result = m(input_batchnorm)
      im2col_input = torch.randn(1, 256, 16, 16, **kwargs)
      unfold = nn.Unfold(1, 1)
      unfold.to('cuda')
      col2im_input = unfold(im2col_input)
      # fold = nn.Fold((16, 16), 1, 1)
      # fold.to('cuda')
      # fold(col2im_input)
      # exit(0)
      print(im2col_input.shape)
      fusion_cuda.call_col2im_batchnorm_backward(
          col2im_input, 
          torch.randn(input_batchnorm.shape, **kwargs), input_batchnorm, m.weight,
          m.running_mean, m.running_var)
    if idx == 15:
      im2col_input = torch.randn(1, 256, 16, 16, **kwargs)
      unfold = nn.Unfold(1, 1)
      unfold.to('cuda')
      col2im_input = unfold(im2col_input)
      input_max_pool = torch.randn(1, mp_, 16, 16, **kwargs)
      mxp = nn.MaxPool2d(3, stride=2, padding=1, return_indices=True)
      mxp.to('cuda')
      r2 = mxp(input_max_pool)
      fusion_cuda.call_col2im_maxpooling_backward(
          col2im_input,
          torch.randn(1,mp_,8,8, **kwargs),
          input_max_pool, r2[1])

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device=None)

if len(sys.argv) == 2:
    run(int(sys.argv[1]))
else:
    for i in range(10):
        run(i)



