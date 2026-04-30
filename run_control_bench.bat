@echo off
cd /d C:\Projects\Aura-LLM-Harness
.\.venv\Scripts\python.exe -m aura_harness.bench --task duplicate_finder --runs 5 --n 15 --critic-rounds 0
