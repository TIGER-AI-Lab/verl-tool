## Installation

### Option 1: UV Installation
We highly recommend using uv to install verl-tool.

```bash
# install uv if not installed first
git submodule update --init --recursive
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"
uv pip install "flash-attn<2.8.0" --no-build-isolation
```

### Option 2: Conda Installation
```bash
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn<2.8.0" --no-build-isolation
```