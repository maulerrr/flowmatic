# Build the Flowmatic thesis PDF on the laptop (AITU template)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$thesisRoot = Join-Path $root "thesis\AITU Thesis Template"
$figuresSrc = Join-Path $root "thesis\figures"

Write-Host "=== Flowmatic thesis build (laptop) ==="
Write-Host "Thesis root: $thesisRoot"

# Sync shared figures into template tree
$figDest = Join-Path $thesisRoot "figures"
New-Item -ItemType Directory -Force -Path (Join-Path $figDest "architecture") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $figDest "results") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $figDest "screenshots") | Out-Null

& (Join-Path $PSScriptRoot "render-thesis-figures.ps1")

Copy-Item -Force (Join-Path $figuresSrc "architecture\*.png") (Join-Path $figDest "architecture")
Copy-Item -Force (Join-Path $figuresSrc "results\*.png") (Join-Path $figDest "results")
if (Test-Path (Join-Path $figuresSrc "screenshots")) {
    Copy-Item -Force (Join-Path $figuresSrc "screenshots\*.png") (Join-Path $figDest "screenshots") -ErrorAction SilentlyContinue
}

Write-Host "=== Building PDF via Docker texlive ==="
Push-Location $thesisRoot
try {
    docker pull texlive/texlive:latest 2>&1 | Out-Null
    docker run --rm -v "${thesisRoot}:/work" -w /work texlive/texlive:latest `
        sh -c "rm -f memoirthesis.aux memoirthesis.toc memoirthesis.lof memoirthesis.lot memoirthesis.out memoirthesis.bbl memoirthesis.blg && pdflatex -interaction=nonstopmode memoirthesis.tex && bibtex memoirthesis && pdflatex -interaction=nonstopmode memoirthesis.tex && pdflatex -interaction=nonstopmode memoirthesis.tex && pdflatex -interaction=nonstopmode memoirthesis.tex"
    if (Test-Path "memoirthesis.pdf") {
        Copy-Item "memoirthesis.pdf" (Join-Path $root "memoirthesis-flowmatic.pdf") -Force
        Write-Host "SUCCESS: $(Join-Path $root 'memoirthesis-flowmatic.pdf')"
    } else {
        Write-Warning "PDF not created - check memoirthesis.log"
    }
} finally {
    Pop-Location
}

Write-Host "Done."
