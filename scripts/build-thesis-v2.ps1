# Build Flowmatic thesis v2 PDF (new manuscript; does not modify v1 template)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$thesisRoot = Join-Path $root "thesis\Flowmatic-Thesis-v2"
$figSrc = Join-Path $root "thesis\figures"

Write-Host "=== Flowmatic thesis v2 build ==="
python (Join-Path $root "thesis\experiments\v2\run_thesis_v2_experiments.py")
Get-ChildItem (Join-Path $root "thesis\experiments\v2\results\generated_*.tex") | Copy-Item -Force -Destination (Join-Path $thesisRoot "chapters\chapter05\")
# Sync figures from canonical thesis/figures if newer
$figDst = Join-Path $thesisRoot "figures"
if (-not (Test-Path $figDst)) { New-Item -ItemType Directory -Path $figDst | Out-Null }
Copy-Item -Recurse -Force (Join-Path $figSrc "*") $figDst -ErrorAction SilentlyContinue

Push-Location $thesisRoot
try {
  docker run --rm -v "${thesisRoot}:/work" -w /work texlive/texlive:latest `
    sh -c "pdflatex -interaction=nonstopmode memoirthesis.tex && bibtex memoirthesis && pdflatex -interaction=nonstopmode memoirthesis.tex && pdflatex -interaction=nonstopmode memoirthesis.tex"
  if (Test-Path "memoirthesis.pdf") {
    Copy-Item "memoirthesis.pdf" (Join-Path $root "memoirthesis-flowmatic-v2.pdf") -Force
    Write-Host "SUCCESS: $(Join-Path $root 'memoirthesis-flowmatic-v2.pdf')"
  }
} finally { Pop-Location }
