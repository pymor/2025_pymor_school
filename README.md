# Materials for pyMOR School 2025

This repository contains course material for the 2025 edition
of [pyMOR School](https://2025.school.pymor.org).

## Virtual environment

Tested on Linux:

```
uv venv -p 3.12
source .venv/bin/activate
uv pip install numpy
cd .venv/bin
ln -s f2py f2py3
cd ../..
uv pip install -r requirements.txt
```
