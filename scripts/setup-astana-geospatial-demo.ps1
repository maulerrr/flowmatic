# Setup Astana Geospatial Demo pipeline via production API
$ErrorActionPreference = "Stop"
$Base = "http://localhost/api/v1"
$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession

function Invoke-Api {
    param([string]$Method, [string]$Path, $Body = $null)
    $params = @{
        Uri = "$Base$Path"
        Method = $Method
        WebSession = $session
        TimeoutSec = 60
        ContentType = "application/json"
    }
    if ($null -ne $Body) { $params.Body = ($Body | ConvertTo-Json -Depth 6) }
    return Invoke-RestMethod @params
}

$creds = @{ email = "thesis.demo@flowmatic.local"; password = "ThesisDemo2025!" }
try {
    Invoke-Api POST "/auth/login" $creds | Out-Null
} catch {
    Invoke-Api POST "/auth/register" (@{
        email = $creds.email
        password = $creds.password
        displayName = "Thesis Demo"
        organizationName = "Astana Smart City Lab"
    }) | Out-Null
    Invoke-Api POST "/auth/login" $creds | Out-Null
}
Write-Host "Logged in"

$pipelines = (Invoke-Api GET "/smart-city/pipelines").data
$pipeline = $pipelines | Where-Object { $_.name -eq "Astana Geospatial Demo" } | Select-Object -First 1
if (-not $pipeline) {
    $pipeline = (Invoke-Api POST "/smart-city/pipelines" @{
        name = "Astana Geospatial Demo"
        description = "Thesis defence demo: Astana geospatial traffic simulator"
    }).data
    Write-Host "Created pipeline $($pipeline.id)"
} else {
    Write-Host "Reusing pipeline $($pipeline.id)"
}

$sources = (Invoke-Api GET "/smart-city/pipelines/$($pipeline.id)/sources").data
$desired = @(
    @{ sensorKind = "traffic"; name = "Astana Geospatial Traffic"; type = "WEBSOCKET" },
    @{ sensorKind = "weather"; name = "Astana Weather Stations"; type = "HTTP_POLLING" }
)
foreach ($spec in $desired) {
    $existing = $sources | Where-Object { $_.sensorKind -eq $spec.sensorKind } | Select-Object -First 1
    if (-not $existing) {
        $existing = (Invoke-Api POST "/smart-city/pipelines/$($pipeline.id)/sources" (@{
            name = $spec.name
            type = $spec.type
            mode = "SIMULATED"
            sensorKind = $spec.sensorKind
            pollIntervalMs = 2000
        })).data
        Write-Host "Created source $($spec.sensorKind)"
    }
    if ($existing.status -ne "RUNNING") {
        Invoke-RestMethod -Uri "$Base/smart-city/sources/$($existing.id)/start" -Method POST -WebSession $session -TimeoutSec 60 | Out-Null
        Write-Host "Started source $($existing.id) ($($spec.sensorKind))"
    }
}

Start-Sleep -Seconds 4
$events = (Invoke-Api GET "/smart-city/pipelines/$($pipeline.id)/events?limit=10").data
Write-Host "Recent events: $($events.Count)"
if ($events.Count -gt 0) {
    $payload = $events[0].payload
    if (-not $payload) { $payload = $events[0] }
    Write-Host "Sample lat/lng: $($payload.latitude) $($payload.longitude)"
}

$out = Join-Path (Split-Path $PSScriptRoot -Parent) "thesis\experiments\smart_city_pipeline_setup.json"
@{
    pipelineId = $pipeline.id
    pipelineName = $pipeline.name
    recentEventCount = $events.Count
} | ConvertTo-Json | Set-Content $out -Encoding UTF8
Write-Host "Wrote $out"
