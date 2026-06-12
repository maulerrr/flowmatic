# Build standalone abstract PDFs (English, Russian, Kazakh) — abstract text only
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outRoot = Join-Path $root "thesis\Flowmatic-Thesis-v3\output"
$absRoot = Join-Path $root "thesis\Flowmatic-Thesis-v3\frontmatter\abstracts"

Write-Host "=== Flowmatic abstract PDFs (EN / RU / KZ) ==="
if (-not (Test-Path $outRoot)) { New-Item -ItemType Directory -Path $outRoot | Out-Null }

Push-Location $absRoot
try {
  docker run --rm -v "${absRoot}:/work" -w /work texlive/texlive:latest `
    sh -c "set -e; xelatex -interaction=nonstopmode abstract-en.tex; xelatex -interaction=nonstopmode abstract-ru.tex; xelatex -interaction=nonstopmode abstract-kz.tex"

  $pairs = @(
    @{ src = "abstract-en.pdf"; dst = "abstract-en.pdf" },
    @{ src = "abstract-ru.pdf"; dst = "abstract-ru.pdf" },
    @{ src = "abstract-kz.pdf"; dst = "abstract-kz.pdf" }
  )

  foreach ($pair in $pairs) {
    $src = Join-Path $absRoot $pair.src
    if (-not (Test-Path $src)) {
      throw "Missing output: $($pair.src)"
    }
    Copy-Item $src (Join-Path $outRoot $pair.dst) -Force
    Copy-Item $src (Join-Path $root $pair.dst) -Force
    Copy-Item $src (Join-Path $env:USERPROFILE "Downloads\$($pair.dst)") -Force
    Write-Host "SUCCESS: $(Join-Path $root $pair.dst)"
  }
} finally { Pop-Location }
