### Build compilation database or compile commands
- config debug:
- xmake f -m debug    
- xmake project -k compile_commands

or 
- xmake p -k compdb

## Install Intel oneAPI MKL
https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

## Setup Build
    xmake f --menu
    xmake f --mkl=y
    xmake f --toolchain=clang
    xmake f -v
    xmake config

### Clean Build
    xmake clean
    xmake
    xmake install

### Debug build
    xmake f -m debug
    xmake
    xmake install

### Debug release
    xmake f -m release
    xmake
    xmake install

### Enable NVIDIA GPU option
    xmake f --nv-gpu=y
    xmake
    xmake install

### Install if needed
pip install -U "huggingface_hub[cli]"

### Download the model
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir ./models/DeepSeek-R1-Distill-Qwen-1.5B

### 1. Uninstalled Old Package
```bash
pip uninstall -y llaisys
```

### 2. Installed in Editable Mode
```bash
cd python && pip install -e .
```

**Editable mode (`-e`)** creates a link to your source code instead of copying it. Now changes are **immediately reflected** without reinstalling!


### Test the runtime

python test/test_runtime.py --model [dir_path/to/model] --device nvidia
python test/test_runtime.py --device cpu

python test/test_infer.py --model [dir_path/to/model] --test --device nvidia
python test/test_infer.py --device cpu --model ./models/DeepSeek-R1-Distill-Qwen-1.5B
