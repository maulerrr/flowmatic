# Build thesis body-only PDF (chapters + appendices, no front matter or bibliography)
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$thesisRoot = Join-Path $root "thesis\Flowmatic-Thesis-v2"
$outRoot = Join-Path $root "memoirthesis-flowmatic-v2-body-only.pdf"
$outDownloads = Join-Path $env:USERPROFILE "Downloads\Ramazan_Seiitbek_thesis_content_only.pdf"

Write-Host "=== Flowmatic thesis body-only build ==="

Push-Location $thesisRoot
try {
  docker run --rm -v "${thesisRoot}:/work" -w /work texlive/texlive:latest `
    sh -c "pdflatex -interaction=nonstopmode memoirthesis-body-only.tex && bibtex memoirthesis-body-only && pdflatex -interaction=nonstopmode memoirthesis-body-only.tex && pdflatex -interaction=nonstopmode memoirthesis-body-only.tex"
  if (Test-Path "memoirthesis-body-only.pdf") {
    Copy-Item "memoirthesis-body-only.pdf" $outRoot -Force
    Copy-Item "memoirthesis-body-only.pdf" $outDownloads -Force
    Write-Host "SUCCESS: $outRoot"
    Write-Host "SUCCESS: $outDownloads"
  } else {
    throw "memoirthesis-body-only.pdf was not produced"
  }
} finally { Pop-Location }
