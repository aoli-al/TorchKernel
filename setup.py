from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'fused/Im2ColMaxpoolNorm.cu',
            'fused/UpsampleNormalization.cu',
            'fused/Im2ColMaxPool.cu',
            'fused/Im2ColUpSample.cu',
            'fused/MaxPoolBatchNorm.cu',
            'wrappers/wrapper.cpp',
            'fused/Im2ColNormalization.cu',
            'fused/MaxPoolUpSample.cu',
        ],
        extra_compile_args={'cxx': [],
                            'nvcc': ['-O3', '--expt-extended-lambda']},
        include_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/include', '/home/hao01/pytorch/aten/src'],
        library_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/lib'],
        libraries = ["torch"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
