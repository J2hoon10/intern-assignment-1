# Run the full experiment pipeline: base -> full_ft -> lora, then plot loss curves.
# Usage (from the project root, with the conda env activated):
#   conda activate lora_pubmedqa
#   ./scripts/run_all.ps1
$ErrorActionPreference = "Stop"

# Move to the project root (parent of this script's directory).
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "==> [1/7] Preparing data" -ForegroundColor Cyan
python src/prepare_data.py --pqaa_size 30000

Write-Host "==> [2/7] Base (zero-shot) evaluation" -ForegroundColor Cyan
python src/evaluate.py --config configs/base.yaml

Write-Host "==> [3/7] Full fine-tuning" -ForegroundColor Cyan
python src/train.py --config configs/full_ft.yaml

Write-Host "==> [4/7] Full fine-tuning evaluation" -ForegroundColor Cyan
python src/evaluate.py --config configs/full_ft.yaml

Write-Host "==> [5/7] LoRA fine-tuning" -ForegroundColor Cyan
python src/train.py --config configs/lora.yaml

Write-Host "==> [6/7] LoRA evaluation" -ForegroundColor Cyan
python src/evaluate.py --config configs/lora.yaml

Write-Host "==> [7/7] Plotting loss curves" -ForegroundColor Cyan
python src/plot_logs.py --all

Write-Host "Done. See outputs/summary.txt for the comparison." -ForegroundColor Green
Get-Content outputs/summary.txt
