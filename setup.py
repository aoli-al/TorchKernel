from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

nvcc_args = ['-O3', '--expt-extended-lambda']
if 'MAX_REG' in os.environ:
    nvcc_args.append('-maxrregcount=' + os.environ['MAX_REG'])


setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'fused/MaxpoolingBackwardBatchnorm.cu',
            'fused/Im2ColNormalization.cu',
            'wrappers/wrapper.cpp',
            'fused/SummaryMaxpool.cu',
            'fused/SummaryUpsample.cu',
            'fused/Im2ColMaxPool.cu',
            'fused/SummaryOps.cu',
            'fused/MaxPoolUpSample.cu',
            'fused/SummaryNorm.cu',
            'fused/UpsampleNormalization.cu',
            'fused/Im2ColUpSample.cu',
            'fused/MaxPoolBatchNorm.cu',
            'fused/col2im_batchnorm_backward.cu',
            'cuda/DilatedMaxPool2d.cu',
        ],
        extra_compile_args={'cxx': [],
                            'nvcc': nvcc_args
                                     },
        #  include_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/include', '/home/hao01/pytorch/aten/src'],
        #  library_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/lib'],
        libraries = ["torch"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
