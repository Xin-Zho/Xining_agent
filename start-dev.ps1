$ErrorActionPreference = 'Stop'

$ErrorActionPreference = 'Continue'

function Resolve-Python {
  $repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

  # 优先使用项目 venv Python（已安装所有依赖）
  $venvPython = Join-Path $repoRoot 'venv\Scripts\python.exe'
  if (Test-Path $venvPython) {
    return $venvPython
  }

  # 备选：全局 Python 313
  $globalPython = 'C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe'
  if (Test-Path $globalPython) {
    return $globalPython
  }

  # 最后尝试 PATH 中的 python
  $cmd = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }

  throw 'Python not found. Please install Python or set up venv.'
}

function Resolve-Npm {
  $candidates = @(
    'D:\node.js\npm.cmd',
    'npm.cmd'
  )

  foreach ($candidate in $candidates) {
    if ($candidate -eq 'npm.cmd') {
      $cmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
      if ($cmd) {
        return $cmd.Source
      }
      continue
    }

    if (Test-Path $candidate) {
      return $candidate
    }
  }

  throw 'npm.cmd not found.'
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Resolve-Python
$npmCmd = Resolve-Npm

Write-Host "Python: $pythonExe"
Write-Host "npm:    $npmCmd"

# 从 .env 读取 API Key（环境变量优先）
$apiKey = if ($env:DEEPSEEK_API_KEY) {
  $env:DEEPSEEK_API_KEY
} else {
  $envPath = Join-Path $repoRoot '.env'
  if (Test-Path $envPath) {
    $match = Select-String -Path $envPath -Pattern 'DEEPSEEK_API_KEY\s*=\s*"?([^"]*)"?' | Select-Object -First 1
    if ($match) { $match.Matches.Groups[1].Value } else { '' }
  } else { '' }
}

$backendCmd = @'
Set-Location '{0}'
$env:DEEPSEEK_API_KEY='{1}'
$env:AGENT_RUNTIME='claude-code'
& '{2}' -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
'@ -f $repoRoot, $apiKey, $pythonExe

$frontendCmd = @'
Set-Location '{0}\mobile'
$env:CI='1'
$env:EXPO_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'
& '{1}' run web -- --port 8081
'@ -f $repoRoot, $npmCmd

Start-Process powershell.exe -ArgumentList '-NoExit', '-Command', $backendCmd
Start-Process powershell.exe -ArgumentList '-NoExit', '-Command', $frontendCmd

Start-Sleep -Seconds 8
Start-Process 'http://localhost:8081'

Write-Host 'Backend: http://127.0.0.1:8000'
Write-Host 'Frontend: http://localhost:8081'
