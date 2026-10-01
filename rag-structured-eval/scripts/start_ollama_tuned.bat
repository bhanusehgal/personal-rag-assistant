@echo off
REM Correction 2026-09-20 (see PROGRESS.md): OLLAMA_CONTEXT_LENGTH DOES work
REM as an env var when the Ollama desktop tray app is fully stopped first
REM (Get-Process ollama* | Stop-Process before running this). The earlier
REM "env vars don't work" finding was confounded by the tray app racing this
REM script with its own server using its settings-DB context_length. Still
REM run scripts\fix_ollama_settings_db.py once (Ollama stopped) so the
REM settings DB stays consistent for OLLAMA_MODELS / if the tray app is ever
REM used instead of this script.
set OLLAMA_MAX_LOADED_MODELS=1
set OLLAMA_KEEP_ALIVE=30s
set OLLAMA_CONTEXT_LENGTH=1024
ollama serve
