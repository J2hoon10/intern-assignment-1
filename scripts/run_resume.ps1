# Resume-safe pipeline: continues an interrupted experiment WITHOUT resetting progress.
#   - Does NOT re-run prepare_data (keeps existing data/ so LoRA resume stays consistent)
#   - Never passes --fresh, so every train.py auto-resumes from its latest checkpoint
#     (LoRA continues from checkpoint-XXXX; full_ft starts fresh only because it has no checkpoint)
#
# NOTE: messages are ASCII-only on purpose. Windows PowerShell 5.1 reads .ps1 files with the
# system code page (cp949 on Korean Windows), so non-ASCII text would show as mojibake.
#
# Usage (from the project root, env activated):
#   conda activate lora_pubmedqa
#   .\scripts\run_resume.ps1
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

# 0) Data must already exist (do NOT regenerate - that could break a clean resume).
if (-not (Test-Path "data\train.jsonl")) {
    Write-Host "data\train.jsonl not found. Prepare data first:" -ForegroundColor Yellow
    Write-Host "  python src\prepare_data.py --pqaa_size 30000"
    exit 1
}

# 1) LoRA - resumes from the latest checkpoint (this is the in-progress run).
Run-Step "LoRA train (auto-resume from latest checkpoint)" { python src/train.py --config configs/lora.yaml }
Run-Step "LoRA evaluate"                                   { python src/evaluate.py --config configs/lora.yaml }

# 2) full_ft - no checkpoint yet (previous run OOM'd), so it starts fresh with the safe config.
#    If interrupted later, re-running this same script resumes it from its own latest checkpoint.
Run-Step "full_ft train (safe config: batch 2 + grad checkpointing)" { python src/train.py --config configs/full_ft.yaml }
Run-Step "full_ft evaluate"                                          { python src/evaluate.py --config configs/full_ft.yaml }

# 3) base - evaluate only if not already done.
if (-not (Test-Path "outputs\base\metrics.json")) {
    Run-Step "base zero-shot evaluate" { python src/evaluate.py --config configs/base.yaml }
} else {
    Write-Host "==> base metrics already exist - skipping" -ForegroundColor DarkGray
}

# 4) Loss curves + summary.
Run-Step "plot loss curves" { python src/plot_logs.py --all }
Write-Host "`n=== FINAL SUMMARY ===" -ForegroundColor Green
Get-Content outputs\summary.txt
