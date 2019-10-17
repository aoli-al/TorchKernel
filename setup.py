from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'fused/Im2ColMaxPool.cu',
            'fused/Im2ColUpSample.cu',
            'fused/MaxPoolBatchNorm.cu',
            'wrappers/wrapper.cpp',
            'cuda/Im2Col.cu',
            'fused/Im2ColNormalization.cu',
            'fused/MaxPoolUpSample.cu',
        ],
        # extra_compile_args={'cxx': ['-g'],
        #                     'nvcc': ['-g']},
        include_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/include', '/home/hao01/pytorch/aten/src']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
