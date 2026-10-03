# Create the `drp` conda environment from scratch.
#
#   powershell -ExecutionPolicy Bypass -File scripts\00_setup_env.ps1
#
# Deliberately minimal. There is no PyTorch Geometric, no DGL and no compiler
# toolchain: the graph message passing in src/fragdrp/models/layers.py is built
# from torch.index_add_ and scatter_reduce_, which removes the usual class of
# Windows build failures.

param(
    [string]$EnvName = "drp",
    [string]$PythonVersion = "3.11",
    [string]$CudaChannel = "cu124"   # match to your driver; cu121/cu126 also fine
)

$ErrorActionPreference = "Stop"

Write-Host "Creating conda environment '$EnvName' (Python $PythonVersion)..." -ForegroundColor Cyan
conda create -y -n $EnvName "python=$PythonVersion"

$python = (conda run -n $EnvName python -c "import sys; print(sys.executable)").Trim()
Write-Host "Using interpreter: $python"

Write-Host "Installing PyTorch ($CudaChannel)..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
& $python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaChannel"

Write-Host "Installing the scientific stack and drevalpy..." -ForegroundColor Cyan
& $python -m pip install "numpy<2.3" pandas scipy scikit-learn rdkit pyyaml tqdm matplotlib seaborn drevalpy

Write-Host "Installing this package in editable mode..." -ForegroundColor Cyan
& $python -m pip install --no-deps -e .

Write-Host "`nVerifying..." -ForegroundColor Cyan
& $python -c @"
import torch, rdkit, drevalpy, fragdrp
print('torch     ', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device    ', torch.cuda.get_device_name(0))
print('rdkit     ', rdkit.__version__)
print('drevalpy  ', getattr(drevalpy, '__version__', '?'))
print('fragdrp   ', fragdrp.__file__)
"@

Write-Host "`nDone. Activate with:  conda activate $EnvName" -ForegroundColor Green
Write-Host "Then smoke-test with: python scripts/03_ablation.py --dataset TOYv1 --quick"
