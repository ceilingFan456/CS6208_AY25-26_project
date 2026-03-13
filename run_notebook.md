# Run Julia Colab Notebook in VS Code Jupyter (Any Repo)

This guide adapts a Google Colab Julia notebook to a local VS Code Jupyter setup.

## 1. Open terminal in your target repo

```bash
cd /path/to/your/repo
```

## 2. Activate your Python/Jupyter environment

Use the env you want VS Code Jupyter to use (example with conda):

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6208
```

## 3. Install Julia locally (no sudo)

```bash
JULIA_VERSION="1.10.2"
JULIA_DIR="$HOME/.local/julia-$JULIA_VERSION"
JULIA_SERIES="${JULIA_VERSION%.*}"
URL="https://julialang-s3.julialang.org/bin/linux/x64/$JULIA_SERIES/julia-$JULIA_VERSION-linux-x86_64.tar.gz"

mkdir -p "$HOME/.local"
wget -nv "$URL" -O /tmp/julia.tar.gz
rm -rf "$JULIA_DIR"
mkdir -p "$JULIA_DIR"
tar -xzf /tmp/julia.tar.gz -C "$JULIA_DIR" --strip-components=1
rm -f /tmp/julia.tar.gz
```

## 4. Install and register Julia kernel for Jupyter

```bash
$HOME/.local/julia-1.10.2/bin/julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("julia", env=Dict("JULIA_NUM_THREADS"=>"2"))'
```

Install that kernelspec into your active Python env too (important for VS Code discovery):

```bash
python -m jupyter kernelspec install ~/.local/share/jupyter/kernels/julia-1.10 --prefix "$CONDA_PREFIX" --replace
python -m jupyter kernelspec list
```

You should see both:
- `python3`
- `julia-1.10`

## 5. In VS Code, point Jupyter to the right interpreter

1. `Ctrl+Shift+P` -> `Jupyter: Select Interpreter to start Jupyter server`
2. Choose the same env as step 2 (for example `cs6208`)
3. `Ctrl+Shift+P` -> `Developer: Reload Window`
4. Open notebook -> Kernel picker -> `Select Another Kernel...` -> choose `julia 1.10`

## 6. Run Julia package setup cell (in Julia kernel)

```julia
using Pkg
Pkg.add(["Gen", "IterTools"])
Pkg.add(url="https://github.com/probcomp/GenGPT3.jl.git")
```

## 7. Colab script changes you must make locally

- Replace `%%shell` with `%%bash` (or use `!` commands in Python cells).
- Do not extract Julia to `/usr/local` (permission issue). Use `$HOME/.local/...`.
- Do not assume Colab runtime privileges.

## Troubleshooting

If Julia still does not appear in VS Code kernel list:

1. `Ctrl+Shift+P` -> `Jupyter: Specify Jupyter Server for Connections` -> choose `Default (Local)`
2. Reload window again.
3. Re-run:

```bash
python -m jupyter kernelspec list
```

If `julia-1.10` is missing, repeat step 4.
