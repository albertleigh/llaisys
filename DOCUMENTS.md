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



