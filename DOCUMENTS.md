### Build compilation database or compile commands
- config debug:
- xmake f -m debug    
- xmake project -k compile_commands

or 
- xmake p -k compdb

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

