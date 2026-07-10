# Two-phase experiment: PQA-A -> PQA-L (train on PQA-A first, then PQA-L).
# Runs full_ft and lora two-phase, evaluates on the 500 test set, and plots
# per-phase loss curves. ASCII-only messages (Windows PowerShell 5.1 code-page safe).
# Resume-safe: completed phases are skipped; interrupted phases auto-resume.
#
# WARNING: heavy. Do NOT run while the single-stage experiment is still using the GPU.
#
# Usage:
#   conda activate lora_pubmedqa
#   .\scripts\run_multiphase.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Run-Step($desc, [scriptblock]$block) {
    Write-Host "==> $desc" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $desc (exit $LASTEXITCODE) - stopping." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Run-Step "Prepare two-phase data (PQA-A 100k, PQA-L 10-fold 450/50)" {
    python src/prepare_data_multiphase.py --pqaa_size 100000
}

Run-Step "full_ft two-phase train (phase1 PQA-A -> phase2 PQA-L)" {
    python src/train_multiphase.py --config configs/full_ft_multiphase.yaml
}
Run-Step "full_ft two-phase evaluate" {
    python src/evaluate.py --config configs/full_ft_multiphase.yaml
}

Run-Step "lora two-phase train (phase1 PQA-A -> phase2 PQA-L)" {
    python src/train_multiphase.py --config configs/lora_multiphase.yaml
}
Run-Step "lora two-phase evaluate" {
    python src/evaluate.py --config configs/lora_multiphase.yaml
}

if (-not (Test-Path "outputs\base\metrics.json")) {
    Run-Step "base zero-shot evaluate" { python src/evaluate.py --config configs/base.yaml }
}

Write-Host "==> Plot loss curves (phase1 + each PQA-L fold)" -ForegroundColor Cyan
foreach ($exp in @("full_ft_mp", "lora_mp")) {
    python src/plot_logs.py --experiment "$exp/phase1"
    Get-ChildItem -Path "outputs\$exp\phase2" -Directory -Filter "fold*" -ErrorAction SilentlyContinue | ForEach-Object {
        python src/plot_logs.py --experiment "$exp/phase2/$($_.Name)"
    }
}

Write-Host "`n=== FINAL SUMMARY (single-stage + two-phase) ===" -ForegroundColor Green
Get-Content outputs\summary.txt
