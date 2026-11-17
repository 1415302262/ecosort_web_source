param(
    [ValidatePattern('^[A-Za-z0-9_.@-]+$')][string]$SshHost = 'oci-public',
    [ValidateRange(1024,65535)][int]$LocalPort = 5187,
    [ValidateRange(1,65535)][int]$RemotePort = 5187,
    [switch]$ShowAdminToken,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
if ($ShowAdminToken) {
    # Display only on the user's explicit invocation; never save to Git or logs.
    & ssh -o BatchMode=yes -o ClearAllForwardings=yes $SshHost 'cat ~/.config/ecosort/admin.token'
    exit $LASTEXITCODE
}
if (Get-NetTCPConnection -State Listen -LocalPort $LocalPort -ErrorAction SilentlyContinue) {
    throw "Local port $LocalPort is already in use. Choose another -LocalPort."
}
$runtimeDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'runtime'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$sshErrorLog = Join-Path $runtimeDirectory "ssh-tunnel-$LocalPort.log"
$sshExecutable = (Get-Command ssh.exe).Source
$sshArguments = @('-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-L',"${LocalPort}:127.0.0.1:${RemotePort}",'-N',$SshHost)
$tunnelProcess = Start-Process -FilePath $sshExecutable -ArgumentList $sshArguments -WindowStyle Hidden -RedirectStandardError $sshErrorLog -PassThru
try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        $tunnelProcess.Refresh()
        if ($tunnelProcess.HasExited) { throw "SSH tunnel exited. See $sshErrorLog" }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" -TimeoutSec 2
            if ($health.storage_ready -and $health.release) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 300
    }
    if (-not $ready) { throw "Service did not become available. See $sshErrorLog" }
    Write-Host "Ready: http://127.0.0.1:$LocalPort (release $($health.release))"
    if ($NoBrowser) { return }
    Start-Process "http://127.0.0.1:$LocalPort"
    Write-Host 'Keep this console open. Ctrl+C closes only this SSH tunnel.'
    while (-not $tunnelProcess.HasExited) { Start-Sleep -Seconds 1; $tunnelProcess.Refresh() }
} finally {
    $tunnelProcess.Refresh()
    if (-not $tunnelProcess.HasExited) {
        Stop-Process -Id $tunnelProcess.Id -ErrorAction SilentlyContinue
        $tunnelProcess.WaitForExit()
    }
    $tunnelProcess.Dispose()
}
