<#
.SYNOPSIS
    Arranque elastico de Airflow: primero sin topes de recursos, luego acotado.

.DESCRIPTION
    1. Levanta el stack con docker-compose.boot.yaml (sin limites de CPU/RAM),
       para que migraciones, carga de DAGs y primer parseo vayan rapidos.
    2. Espera a que webserver y scheduler esten operativos.
    3. Aplica en caliente (docker update) los limites definidos en
       docker-compose.yaml, sin recrear los contenedores.
#>
[CmdletBinding()]
param(
    [int]$TimeoutSegundos = 600
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

Write-Host 'Levantando Airflow sin topes de recursos...' -ForegroundColor Cyan
docker compose -f docker-compose.yaml -f docker-compose.boot.yaml up -d
if ($LASTEXITCODE -ne 0) { throw 'Fallo el arranque de docker compose.' }

Write-Host 'Esperando a que el webserver responda...' -ForegroundColor Cyan
$limite = (Get-Date).AddSeconds($TimeoutSegundos)
$estado = $null

do {
    Start-Sleep -Seconds 10

    $estado = docker compose ps --format json |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.Service -eq 'airflow-webserver' } |
        Select-Object -ExpandProperty Health -First 1

    if (-not $estado) { $estado = 'starting' }

    Write-Host "  webserver: $estado"
} while ($estado -ne 'healthy' -and (Get-Date) -lt $limite)

if ($estado -ne 'healthy') {
    throw "El webserver no llegó a healthy en $TimeoutSegundos segundos."
}

Write-Host 'Aplicando limites de recursos en caliente...' -ForegroundColor Cyan
$config = docker compose -f docker-compose.yaml config --format json | ConvertFrom-Json

foreach ($servicio in $config.services.PSObject.Properties) {
    $cpus = $servicio.Value.cpus
    $memoria = $servicio.Value.mem_limit
    if (-not $cpus -and -not $memoria) { continue }

    $contenedor = (docker compose ps -q $servicio.Name | Select-Object -First 1)
    if (-not $contenedor) { continue }

    $argumentos = @()
    if ($cpus) { $argumentos += @('--cpus', $cpus) }
    if ($memoria) { $argumentos += @('--memory', $memoria, '--memory-swap', $memoria) }

    docker update @argumentos $contenedor | Out-Null
    Write-Host ("  {0}: cpus={1} mem={2}" -f $servicio.Name, $cpus, $memoria)
}

Write-Host 'Listo. Airflow: http://localhost:8080  |  Flower: http://localhost:5555' -ForegroundColor Green
