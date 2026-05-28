# Render thesis architecture diagrams from Mermaid sources via Docker.
$ErrorActionPreference = "Stop"

$archDir = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")) "thesis\figures\architecture"
$config = Join-Path $archDir "mermaid-config.json"

if (-not (Test-Path $config)) {
    Write-Error "Missing config: $config"
}

Write-Host "=== Rendering thesis architecture figures ==="
Write-Host "Source: $archDir"

$files = Get-ChildItem -Path $archDir -Filter "*.mmd"
foreach ($mmd in $files) {
    $png = [System.IO.Path]::ChangeExtension($mmd.FullName, ".png")
    Write-Host "  $($mmd.Name) -> $([System.IO.Path]::GetFileName($png))"
    docker run --rm `
        -v "${archDir}:/data" `
        minlag/mermaid-cli:latest `
        -c /data/mermaid-config.json `
        -b transparent `
        -w 2400 `
        -H 1600 `
        -i "/data/$($mmd.Name)" `
        -o "/data/$([System.IO.Path]::GetFileName($png))"
}

Write-Host "Done. PNG files updated in thesis/figures/architecture/"
