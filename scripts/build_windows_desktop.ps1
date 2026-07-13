$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv-desktop")) {
    py -3.12 -m venv .venv-desktop
}

& .\.venv-desktop\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-desktop\Scripts\python.exe -m pip install -r requirements-desktop.txt "pyinstaller>=6.10,<7"
$env:QT_QPA_PLATFORM = "offscreen"
& .\.venv-desktop\Scripts\python.exe -m unittest tests.test_desktop_api tests.test_desktop_contract tests.test_desktop_qml -v
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
& .\.venv-desktop\Scripts\pyinstaller.exe --noconfirm --clean subpc-desktop.spec

$process = Start-Process -FilePath ".\dist\SUBPC-BUDDY.exe" -ArgumentList "--no-tray", "--smoke-test" -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "SUBPC-BUDDY.exe smoke test failed with code $($process.ExitCode)"
}

Write-Host "Built: dist\SUBPC-BUDDY.exe"
