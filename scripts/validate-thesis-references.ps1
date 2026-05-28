#Requires -Version 5.1
<#
.SYNOPSIS
  Validates thesis bibliography entries by checking DOI/URL accessibility.

.DESCRIPTION
  Parses @key entries from thesisbiblio.bib, resolves DOI links when present,
  and performs HTTP HEAD/GET checks (15s timeout). Prints a markdown table.
#>
param(
    [string]$BibPath = (Join-Path $PSScriptRoot "..\thesis\AITU Thesis Template\thesisbiblio.bib")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $BibPath)) {
    Write-Error "Bib file not found: $BibPath"
    exit 1
}

$content = Get-Content -LiteralPath $BibPath -Raw -Encoding UTF8

function Get-FieldValue {
    param([string]$Block, [string]$FieldName)
    $pattern = "(?im)^\s*$FieldName\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*,?\s*$"
    if ($Block -match $pattern) {
        return $Matches[1].Trim()
    }
    return $null
}

function Resolve-LinkUrl {
    param([string]$Doi, [string]$Url)
    if ($Doi) {
        return "https://doi.org/$Doi"
    }
    return $Url
}

function Test-LinkAccessible {
    param([string]$LinkUrl)
    if ([string]::IsNullOrWhiteSpace($LinkUrl)) {
        return @{ Accessible = $false; Note = 'no doi or url field' }
    }

    $headers = @{
        'User-Agent' = 'FlowmaticThesisValidator/1.0 (reference-check; +https://doi.org)'
    }

    try {
        $response = Invoke-WebRequest -Uri $LinkUrl -Method Head -TimeoutSec 15 -MaximumRedirection 10 -UseBasicParsing -Headers $headers
        $status = [int]$response.StatusCode
        if ($status -ge 200 -and $status -lt 400) {
            return @{ Accessible = $true; Note = "HTTP $status" }
        }
        return @{ Accessible = $false; Note = "HTTP $status (HEAD)" }
    }
    catch {
        $headError = $_.Exception.Message
        try {
            $response = Invoke-WebRequest -Uri $LinkUrl -Method Get -TimeoutSec 15 -MaximumRedirection 10 -UseBasicParsing -Headers $headers
            $status = [int]$response.StatusCode
            if ($status -ge 200 -and $status -lt 400) {
                return @{ Accessible = $true; Note = "HTTP $status (GET fallback)" }
            }
            return @{ Accessible = $false; Note = "HTTP $status (GET)" }
        }
        catch {
            return @{ Accessible = $false; Note = "unreachable: $headError" }
        }
    }
}

$entryPattern = '(?ms)@\w+\s*\{\s*([^,\s]+)\s*,(.*?)^\}'
$matches = [regex]::Matches($content, $entryPattern)
$results = @()

foreach ($match in $matches) {
    $key = $match.Groups[1].Value.Trim()
    $block = $match.Groups[2].Value

    $doi = Get-FieldValue -Block $block -FieldName 'doi'
    $url = Get-FieldValue -Block $block -FieldName 'url'
    $note = Get-FieldValue -Block $block -FieldName 'note'
    $link = Resolve-LinkUrl -Doi $doi -Url $url
    $check = Test-LinkAccessible -LinkUrl $link

    $rowNote = $check.Note
    if ($note) {
        $rowNote = if ($rowNote) { "$rowNote; $note" } else { $note }
    }

    $results += [pscustomobject]@{
        Key         = $key
        Accessible  = if ($check.Accessible) { 'yes' } else { 'no' }
        Note        = $rowNote
    }
}

$total = $results.Count
$accessible = ($results | Where-Object { $_.Accessible -eq 'yes' }).Count
$problematic = $results | Where-Object { $_.Accessible -eq 'no' } | Select-Object -ExpandProperty Key

Write-Output ""
Write-Output "## Thesis bibliography validation"
Write-Output ""
Write-Output "Bib file: ``$BibPath``"
Write-Output ""
Write-Output "| key | accessible | note |"
Write-Output "| --- | --- | --- |"
foreach ($row in ($results | Sort-Object Key)) {
    $safeNote = ($row.Note -replace '\|', '/')
    Write-Output "| $($row.Key) | $($row.Accessible) | $safeNote |"
}
Write-Output ""
Write-Output "**Summary:** $accessible / $total accessible"
if ($problematic.Count -gt 0) {
    Write-Output ""
    Write-Output "**Inaccessible keys:** $($problematic -join ', ')"
}
