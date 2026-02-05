# Obstklassifikation CNN - Setup-Skript
# Dieses Skript richtet die GPU-Umgebung automatisch ein

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Obstklassifikation CNN - GPU Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Prüfe ob Miniconda installiert ist
$condaPath = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
if (-not (Test-Path $condaPath)) {
    $condaPath = "$env:LOCALAPPDATA\miniconda3\Scripts\conda.exe"
}
if (-not (Test-Path $condaPath)) {
    $condaPath = "C:\ProgramData\miniconda3\Scripts\conda.exe"
}

if (-not (Test-Path $condaPath)) {
    Write-Host "[!] Miniconda nicht gefunden. Installiere Miniconda..." -ForegroundColor Yellow
    winget install Anaconda.Miniconda3 --accept-source-agreements --accept-package-agreements
    $condaPath = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    
    # Warte auf Installation
    Start-Sleep -Seconds 5
    
    if (-not (Test-Path $condaPath)) {
        Write-Host "[ERROR] Miniconda Installation fehlgeschlagen!" -ForegroundColor Red
        Write-Host "Bitte installieren Sie Miniconda manuell von: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    }
}

Write-Host "[OK] Conda gefunden: $condaPath" -ForegroundColor Green

# Akzeptiere TOS falls nötig
Write-Host ""
Write-Host "[...] Akzeptiere Conda Terms of Service..." -ForegroundColor Yellow
& $condaPath tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
& $condaPath tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null
& $condaPath tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null

# Erstelle Conda-Umgebung aus environment.yml
$envName = "fruit_classifier_gpu"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $scriptDir "environment.yml"

Write-Host ""
Write-Host "[...] Erstelle Conda-Umgebung '$envName'..." -ForegroundColor Yellow
Write-Host "      Dies kann einige Minuten dauern (CUDA wird heruntergeladen)..."
& $condaPath env create -f $envFile -n $envName --yes 2>&1 | Out-Host

if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Umgebung existiert bereits, aktualisiere..." -ForegroundColor Yellow
    & $condaPath env update -f $envFile -n $envName 2>&1 | Out-Host
}

# Ermittle Umgebungspfad
$envPath = & $condaPath info --envs | Select-String $envName | ForEach-Object { $_.ToString().Split(" ")[-1] }
if (-not $envPath) {
    $envPath = "$env:USERPROFILE\miniconda3\envs\$envName"
}

Write-Host "[OK] Conda-Umgebung erstellt: $envPath" -ForegroundColor Green

# Installiere Jupyter Kernel
Write-Host ""
Write-Host "[...] Registriere Jupyter Kernel..." -ForegroundColor Yellow
& "$envPath\python.exe" -m ipykernel install --user --name $envName --display-name "Python 3.10 (Fruit Classifier GPU)"

# Konfiguriere Kernel mit CUDA PATH
$kernelDir = "$env:APPDATA\jupyter\kernels\$envName"
$kernelJson = @"
{
 "argv": [
  "$($envPath.Replace('\', '\\'))\\python.exe",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python 3.10 (Fruit Classifier GPU)",
 "language": "python",
 "metadata": {
  "debugger": true
 },
 "env": {
  "PATH": "$($envPath.Replace('\', '\\'))\\Library\\bin;%PATH%"
 }
}
"@

$kernelJson | Out-File -FilePath "$kernelDir\kernel.json" -Encoding UTF8
Write-Host "[OK] Jupyter Kernel konfiguriert" -ForegroundColor Green

# Füge CUDA zum Benutzer-PATH hinzu (optional, aber empfohlen)
$cudaPath = "$envPath\Library\bin"
$currentUserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($currentUserPath -notlike "*$cudaPath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$cudaPath;$currentUserPath", "User")
    Write-Host "[OK] CUDA-Pfad zum Benutzer-PATH hinzugefuegt" -ForegroundColor Green
}

# Test
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  GPU-Test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

$env:PATH = "$cudaPath;$env:PATH"
& "$envPath\python.exe" -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TensorFlow Version: {tf.__version__}'); print(f'GPUs gefunden: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Setup abgeschlossen!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Naechste Schritte:"
Write-Host "1. VS Code neu starten"
Write-Host "2. Notebook oeffnen"
Write-Host "3. Kernel 'Python 3.10 (Fruit Classifier GPU)' auswaehlen"
Write-Host ""
