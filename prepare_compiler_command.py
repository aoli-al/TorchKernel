import json
import os
import glob

directory = os.path.abspath("./fused")

data = []

clang_path = "/root/SmartFuser/deps/llvm-project/build/bin/clang++"
cuda_path = "/usr/local/cuda"
torch_path = "/root/pytorch"
torch_lib_path = "/root/miniconda3/lib/python3.7/site-packages/torch"

template = f"{clang_path} --cuda-gpu-arch=sm_62 --cuda-device-only --cuda-path={cuda_path} -I{directory}  -I{torch_lib_path}/include -I{torch_path}/aten/src -I{torch_path}/include/torch/csrc/api/include -I{torch_path}/include/TH -I{torch_path}/include/THC -I{cuda_path}/include -c {directory}/##1.cu -o build/temp.linux-x86_64-3.6/fused/##1.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__  -fPIC -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=fusion_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14"

for f in glob.glob(os.path.join(directory, "*.cu")):
    base_name = os.path.basename(f).split(".")[0]
    element = {
            "directory": directory,
            "file": os.path.abspath(f),
            "command": template.replace("##1", base_name)
            }
    data.append(element)

json.dump(data, open("./fused/compile_commands.json", "w"), indent=2)



