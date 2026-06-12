# Build Flowmatic thesis v3 PDFs (codex-revised + no-signatures submission)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$thesisRoot = Join-Path $root "thesis\Flowmatic-Thesis-v3"
$figSrc = Join-Path $root "thesis\figures"

function Clear-ThesisArtifacts {
  param([string]$BaseName)
  foreach ($ext in @("aux", "bbl", "blg", "log", "out", "toc", "lof", "lot", "pdf", "fls", "fdb_latexmk", "synctex.gz")) {
    $path = Join-Path $thesisRoot "$BaseName.$ext"
    if (Test-Path $path) { Remove-Item $path -Force }
  }
}

function Invoke-ThesisBuild {
  param(
    [string]$MainTex,
    [string]$OutputName
  )

  $baseName = [System.IO.Path]::GetFileNameWithoutExtension($MainTex)
  Clear-ThesisArtifacts -BaseName $baseName

  docker run --rm -v "${thesisRoot}:/work" -w /work texlive/texlive:latest `
    sh -c "set -e; pdflatex -interaction=nonstopmode $MainTex; bibtex $baseName; pdflatex -interaction=nonstopmode $MainTex; pdflatex -interaction=nonstopmode $MainTex"

  $built = Join-Path $thesisRoot ($MainTex -replace '\.tex$', '.pdf')
  if (-not (Test-Path $built)) {
    throw "PDF not produced: $built"
  }
  $dest = Join-Path $root $OutputName
  Copy-Item $built $dest -Force
  Copy-Item $built (Join-Path $env:USERPROFILE "Downloads\$OutputName") -Force
  Write-Host "SUCCESS: $dest"
}

Write-Host "=== Flowmatic thesis v3 build ==="
python (Join-Path $root "thesis\experiments\v2\run_thesis_v2_experiments.py")
Get-ChildItem (Join-Path $root "thesis\experiments\v2\results\generated_*.tex") | Copy-Item -Force -Destination (Join-Path $thesisRoot "chapters\chapter05\")
$figDst = Join-Path $thesisRoot "figures"
if (-not (Test-Path $figDst)) { New-Item -ItemType Directory -Path $figDst | Out-Null }
Copy-Item -Recurse -Force (Join-Path $figSrc "*") $figDst -ErrorAction SilentlyContinue

Push-Location $thesisRoot
try {
  Invoke-ThesisBuild -MainTex "memoirthesis.tex" -OutputName "memoirthesis-flowmatic-v3-codex-revised.pdf"
  Copy-Item (Join-Path $root "memoirthesis-flowmatic-v3-codex-revised.pdf") (Join-Path $root "memoirthesis-flowmatic-v3.pdf") -Force
  Invoke-ThesisBuild -MainTex "memoirthesis-nosignatures.tex" -OutputName "memoirthesis-flowmatic-v3-submission-nosignatures.pdf"
} finally { Pop-Location }
