<#
.SYNOPSIS
    Start the Supervisor API for Trady integration.

.DESCRIPTION
    Launches the Supervisor API using waitress on the specified port.
    Requires SUPERVISOR_JWT_SECRET and SUPERVISOR_HMAC_SECRET in .env
    or as environment variables.

.PARAMETER Port
    TCP port to listen on (default: 8789).

.EXAMPLE
    .\tools\start_supervisor.ps1
    .\tools\start_supervisor.ps1 -Port 9000
#>
param(
    [int]$Port     = 8789,
    [switch]$Public          # Expose on all interfaces. Default: localhost only.
)

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

# Load .env if present
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith('#') -and $line.Contains('=')) {
            $parts = $line.Split('=', 2)
            $k = $parts[0].Trim()
            $v = $parts[1].Trim().Trim('"').Trim("'")
            if ($k -and -not [System.Environment]::GetEnvironmentVariable($k)) {
                [System.Environment]::SetEnvironmentVariable($k, $v, 'Process')
            }
        }
    }
}

# Validate required secrets
if (-not $env:SUPERVISOR_JWT_SECRET) {
    Write-Error "SUPERVISOR_JWT_SECRET is not set."
    Write-Host "Generate and add to .env:"
    Write-Host '    SUPERVISOR_JWT_SECRET=<output of: python -c "import secrets; print(secrets.token_hex(32))">'
    exit 1
}
if (-not $env:SUPERVISOR_HMAC_SECRET) {
    Write-Error "SUPERVISOR_HMAC_SECRET is not set."
    Write-Host "Generate and add to .env:"
    Write-Host '    SUPERVISOR_HMAC_SECRET=<output of: python -c "import secrets; print(secrets.token_hex(32))">'
    exit 1
}

$env:SUPERVISOR_PORT = $Port

$BindAddr = if ($Public) { "0.0.0.0" } else { "127.0.0.1" }
if ($Public) {
    Write-Warning "Binding to 0.0.0.0:$Port (public). Use a VPN, reverse proxy, or IP allowlist."
} else {
    Write-Host "Binding to 127.0.0.1:$Port (localhost only). Pass -Public to expose externally."
}
Write-Host "Supervisor API starting..."
Write-Host ""

waitress-serve --listen="${BindAddr}:${Port}" supervisor.api:app
