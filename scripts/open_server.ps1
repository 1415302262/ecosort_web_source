param(
    [string]$SshHost = 'oci-public',
    [int]$LocalPort = 5187,
    [int]$RemotePort = 5187,
    [switch]$ShowAdminToken
)
$ErrorActionPreference = 'Stop'
if ($ShowAdminToken) {
    # Display only on the user's explicit invocation; never save to Git or logs.
    & ssh -o BatchMode=yes -o ClearAllForwardings=yes $SshHost 'cat ~/.config/ecosort/admin.token'
    exit $LASTEXITCODE
}
Write-Host "Opening http://127.0.0.1:$LocalPort . Keep this console open; Ctrl+C closes the tunnel."
Start-Process "http://127.0.0.1:$LocalPort"
& ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -L "${LocalPort}:127.0.0.1:${RemotePort}" -N $SshHost
