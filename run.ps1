<#
run.ps1
PowerShell helper to create/activate the virtual environment, install dependencies,
and run the Streamlit app (`new1.py`).

Usage:
  .\run.ps1           # creates venv if missing, installs requirements, runs app
  .\run.ps1 -NoInstall  # skip pip install step (useful if already installed)
#>

param(
    [switch]$NoInstall
)

# Allow script to run in this process (does not change system policy)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPath = Join-Path $scriptRoot ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at '$venvPath'..."
    python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment. Ensure 'python' is on PATH."; exit 1
    }
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Error "Activation script not found at '$activateScript'. Virtualenv may be invalid."; exit 1
}

Write-Host "Activating virtual environment..."
# Use dot-sourcing to activate in current session
. $activateScript

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Python executable not found in venv at '$venvPython'"; exit 1
}

if (-not $NoInstall) {
    Write-Host "Upgrading pip and installing dependencies from requirements.txt (if present)..."
    & $venvPython -m pip install --upgrade pip
    $reqPath = Join-Path $scriptRoot "requirements.txt"
    if (Test-Path $reqPath) {
        & $venvPython -m pip install -r $reqPath
    }
    else {
        Write-Host "requirements.txt not found - installing minimal dependencies..."
        & $venvPython -m pip install streamlit pandas scikit-learn matplotlib numpy
    }
}

Write-Host "Starting Streamlit app (this will block the terminal)"
$appPath = Join-Path $scriptRoot 'new1.py'
if (-not (Test-Path $appPath)) {
    Write-Error "App file not found at $appPath"
    exit 1
}
& $venvPython -m streamlit run $appPath

# End of script
