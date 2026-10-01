<#
.SYNOPSIS
  One-time environment setup for rag-structured-eval: upgrade Ollama, move the
  model store off a nearly-full C: drive onto D:, and pull the 6 new GGUF
  model variants used by the model comparison study.

.NOTES
  Run this from an elevated or normal PowerShell prompt (elevation only
  needed if the Ollama installer requires it). Safe to re-run: each step
  checks current state before acting.
#>

$ErrorActionPreference = 'Stop'

Write-Host "== Step 1: Check current Ollama version ==" -ForegroundColor Cyan
ollama --version

Write-Host "`n== Step 2: Check for Ollama update ==" -ForegroundColor Cyan
Write-Host "If a newer version (target: 0.34.2) is available, install it now via the Ollama app's"
Write-Host "auto-update, or 'winget upgrade Ollama.Ollama' if installed via winget, or the installer"
Write-Host "from https://ollama.com/download. Re-run 'ollama --version' after to confirm."
Write-Host "Press Ctrl+C now to pause and upgrade manually, or continue if already upgraded." -ForegroundColor Yellow

Write-Host "`n== Step 3: Stop Ollama before moving its model store ==" -ForegroundColor Cyan
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Stopping process $($_.ProcessName) (PID $($_.Id))"
    Stop-Process -Id $_.Id -Force -Confirm:$false
}
Start-Sleep -Seconds 2

$oldModelsPath = Join-Path $env:USERPROFILE ".ollama\models"
$newModelsRoot = "D:\ollama-models"

Write-Host "`n== Step 4: Inspect current model store layout ==" -ForegroundColor Cyan
if (Test-Path $oldModelsPath) {
    Get-ChildItem $oldModelsPath | Select-Object Name, Length
} else {
    Write-Host "No existing model store found at $oldModelsPath — nothing to move." -ForegroundColor Yellow
}

Write-Host "`n== Step 5: Move model store to D: ==" -ForegroundColor Cyan
if (Test-Path $newModelsRoot) {
    Write-Host "$newModelsRoot already exists — assuming migration already done, skipping move." -ForegroundColor Yellow
} elseif (Test-Path $oldModelsPath) {
    New-Item -ItemType Directory -Path (Split-Path $newModelsRoot -Parent) -Force -ErrorAction SilentlyContinue | Out-Null
    Move-Item -Path $oldModelsPath -Destination $newModelsRoot
    Write-Host "Moved $oldModelsPath -> $newModelsRoot"
} else {
    New-Item -ItemType Directory -Path $newModelsRoot -Force | Out-Null
    Write-Host "Created fresh $newModelsRoot (no prior model store existed)."
}

Write-Host "`n== Step 6: Set OLLAMA_MODELS env var (User scope, persistent) ==" -ForegroundColor Cyan
[System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $newModelsRoot, 'User')
$env:OLLAMA_MODELS = $newModelsRoot
Write-Host "OLLAMA_MODELS set to $newModelsRoot for this session and future ones."
Write-Host "NOTE: fully close and reopen any other terminal/Ollama app windows so they pick this up." -ForegroundColor Yellow

Write-Host "`n== Step 7: Restart Ollama and verify existing models ==" -ForegroundColor Cyan
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5
ollama list

Write-Host "`n== Step 8: Pull the 6 new model variants (will land on D: automatically) ==" -ForegroundColor Cyan
$models = @(
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q5_K_M",
    "phi4-mini:3.8b-q4_K_M",
    "phi4-mini:3.8b-q8_0",
    "mistral:7b-instruct-q4_K_M",
    "mistral:7b-instruct-q5_K_M"
)
foreach ($m in $models) {
    Write-Host "`nPulling $m ..." -ForegroundColor Green
    ollama pull $m
}

Write-Host "`n== Step 9: Final verification ==" -ForegroundColor Cyan
ollama list
Write-Host "`nDone. Next: verify the parent agent still works with 'python -m agent.loop' from the repo root," -ForegroundColor Cyan
Write-Host "then update rag-structured-eval/PROGRESS.md's Stage 0 checklist."
