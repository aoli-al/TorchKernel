from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'fused/SummaryOps.cu',
            'fused/SummaryNorm.cu',
            # 'fused/Im2ColMaxpoolNorm.cu',
            # 'fused/UpsampleNormalization.cu',
            # 'fused/Im2ColMaxPool.cu',
            # 'fused/Im2ColUpSample.cu',
            # 'fused/MaxPoolBatchNorm.cu',
            'wrappers/wrapper.cpp',
            # 'fused/Im2ColNormalization.cu',
            # 'fused/MaxPoolUpSample.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-g', '-O3', '--expt-extended-lambda',
                            # '-maxrregcount=32',
                            '-lineinfo']},
        include_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/include', '/home/hao01/pytorch/aten/src'],
        library_dirs = ['/home/hao01/torch_extension/lib/python3.6/site-packages/torch/lib'],
        libraries = ["torch"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
