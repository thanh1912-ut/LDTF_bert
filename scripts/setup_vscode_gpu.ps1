$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$basePython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$envRoot = Join-Path $repoRoot '.venv-gpu'
if (-not (Test-Path -LiteralPath $envRoot)) {
    & $basePython -m venv $envRoot
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create GPU environment.' }
}
$gpuPython = Join-Path $envRoot 'Scripts\python.exe'
Write-Progress -Activity 'Setup LDTF GPU' -Status 'Installing CUDA PyTorch (large download)' -PercentComplete 10
& $gpuPython -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
Write-Progress -Activity 'Setup LDTF GPU' -Status 'Installing notebook dependencies' -PercentComplete 60
& $gpuPython -m pip install -r (Join-Path $repoRoot 'requirements.txt') ipykernel ipywidgets
if ($LASTEXITCODE -ne 0) { throw 'Notebook dependency installation failed.' }
& $gpuPython -m ipykernel install --user --name ldtf-gpu --display-name 'LDTF GPU'
if ($LASTEXITCODE -ne 0) { throw 'Kernel registration failed.' }
& $gpuPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed.' }
Write-Progress -Activity 'Setup LDTF GPU' -Completed
Write-Output 'Select LDTF GPU kernel in VS Code, then Run All.'
