# Phase 4 — sync Flowmatic thesis rewrite and build PDF
$ErrorActionPreference = "Stop"

$flowmatic = Resolve-Path (Join-Path $PSScriptRoot "..")
$thesisRoot = "C:\Users\BG\Desktop\master-thesis\dissertation_latex_v5"
$latexSrc = Join-Path $flowmatic "thesis\latex"
$figuresSrc = Join-Path $flowmatic "thesis\figures"
$paperFigSrc = Join-Path $flowmatic "models\paper\figures"

if (-not (Test-Path $thesisRoot)) {
    Write-Error "Thesis project not found: $thesisRoot"
}

Write-Host "=== Phase 4 thesis sync ==="

# Chapters
$chapterMap = @{
    "chapter03_methodology.tex" = "chapters\chapter03\methodology.tex"
    "chapter04_implementation.tex" = "chapters\chapter04\implementation.tex"
    "chapter05_results.tex" = "chapters\chapter05\results.tex"
    "chapter06_conclusion.tex" = "chapters\chapter06\conclusion.tex"
}
foreach ($src in $chapterMap.Keys) {
    Copy-Item (Join-Path $latexSrc "chapters\$src") (Join-Path $thesisRoot $chapterMap[$src]) -Force
}

# Snippets into chapter05 (for \input)
$snippets = @("tables_phase4.tex", "model_formulas.tex", "figures_phase4.tex", "tables.tex")
foreach ($file in $snippets) {
    Copy-Item (Join-Path $latexSrc $file) (Join-Path $thesisRoot "chapters\chapter05\$file") -Force
}
Copy-Item (Join-Path $latexSrc "tables.tex") (Join-Path $thesisRoot "chapters\chapter04\tables.tex") -Force

# Addenda appended to ch1/ch2
$ch1 = Get-Content (Join-Path $thesisRoot "chapters\chapter01\introduction.tex") -Raw
$add1 = Get-Content (Join-Path $latexSrc "chapters\chapter01_addendum.tex") -Raw
if ($ch1 -notmatch "Platform Evolution \(Flowmatic\)") {
    $ch1 = $ch1.TrimEnd() + "`r`n`r`n" + $add1
    [System.IO.File]::WriteAllText((Join-Path $thesisRoot "chapters\chapter01\introduction.tex"), $ch1, (New-Object System.Text.UTF8Encoding($false)))
}

$ch2 = Get-Content (Join-Path $thesisRoot "chapters\chapter02\literature.tex") -Raw
$add2 = Get-Content (Join-Path $latexSrc "chapters\chapter02_addendum.tex") -Raw
if ($ch2 -notmatch "Neural Time-Series Forecasting and Production MLOps") {
    $ch2 = $ch2.TrimEnd() + "`r`n`r`n" + $add2
    [System.IO.File]::WriteAllText((Join-Path $thesisRoot "chapters\chapter02\literature.tex"), $ch2, (New-Object System.Text.UTF8Encoding($false)))
}

# Abstract
Copy-Item (Join-Path $latexSrc "frontmatter\abstract.tex") (Join-Path $thesisRoot "frontmatter\abstract.tex") -Force

# Appendix
Copy-Item (Join-Path $latexSrc "appendix_reproducibility.tex") (Join-Path $thesisRoot "chapters\appendices\appendixB.tex") -Force
$mainPath = Join-Path $thesisRoot "memoirthesis.tex"
$main = Get-Content $mainPath -Raw
if ($main -notmatch "appendixB.tex") {
    $main = $main -replace "\\import\{chapters/appendices/\}\{appendixA.tex\}", "\import{chapters/appendices/}{appendixA.tex}`r`n\import{chapters/appendices/}{appendixB.tex}"
    [System.IO.File]::WriteAllText($mainPath, $main, (New-Object System.Text.UTF8Encoding($false)))
}

# Figures
$figDest = Join-Path $thesisRoot "figures"
New-Item -ItemType Directory -Force -Path $figDest | Out-Null

# Copy result figures from models/paper
$paperFigSrc = Join-Path $flowmatic "models\paper\figures"
$resDest = Join-Path $figDest "results"
$archDest = Join-Path $figDest "architecture"
New-Item -ItemType Directory -Force -Path $resDest, $archDest | Out-Null
if (Test-Path $paperFigSrc) {
    Copy-Item -Force (Join-Path $paperFigSrc "*.png") $resDest
}
# Architecture PNGs (generate offline from thesis/figures/architecture/*.mmd if missing)
$archSrc = Join-Path $flowmatic "thesis\figures\architecture"
if (Test-Path $archSrc) {
    Copy-Item -Force (Join-Path $archSrc "*.png") $archDest -ErrorAction SilentlyContinue
}

# Bibliography entries
$bibPath = Join-Path $thesisRoot "thesisbiblio.bib"
$bib = Get-Content $bibPath -Raw
$bibAdds = @'

@inproceedings{zeng2023transformers,
  author = {Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  title = {Are Transformers Effective for Time Series Forecasting?},
  booktitle = {AAAI},
  year = {2023}
}

@inproceedings{nie2023patchtst,
  author = {Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  title = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  booktitle = {ICLR},
  year = {2023}
}

@inproceedings{yu2018stgcn,
  author = {Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
  title = {Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
  booktitle = {IJCAI},
  year = {2018}
}
'@
if ($bib -notmatch "zeng2023transformers") {
    [System.IO.File]::WriteAllText($bibPath, $bib.TrimEnd() + $bibAdds, (New-Object System.Text.UTF8Encoding($false)))
}

Write-Host "=== Building PDF via Docker (texlive) ==="
Push-Location $thesisRoot
try {
    docker pull texlive/texlive:latest 2>&1 | Out-Null
    docker run --rm -v "${thesisRoot}:/work" -w /work texlive/texlive:latest `
        sh -c "pdflatex -interaction=nonstopmode memoirthesis.tex && bibtex memoirthesis && pdflatex -interaction=nonstopmode memoirthesis.tex && pdflatex -interaction=nonstopmode memoirthesis.tex"
    if (Test-Path "memoirthesis.pdf") {
        Copy-Item "memoirthesis.pdf" (Join-Path $flowmatic "memoirthesis-v5-phase4.pdf") -Force
        Write-Host "SUCCESS: $(Join-Path $flowmatic 'memoirthesis-v5-phase4.pdf')"
    } else {
        Write-Warning "PDF not created - check memoirthesis.log"
    }
} finally {
    Pop-Location
}

Write-Host "Done."
