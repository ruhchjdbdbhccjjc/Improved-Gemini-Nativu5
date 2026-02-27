# cleanup_legacy.ps1 - Undo the LEGACY setup_brain.ps1 (Portable v3.0)
# Use this for projects using the old .config/skills/setup_brain.ps1 logic

$ErrorActionPreference = "Stop"

# --- Portable Root Detection ---
# Finds the first parent directory that contains '.config'
$CurrentSearch = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $null

while ($CurrentSearch -and -not $ProjectRoot) {
    if (Test-Path (Join-Path $CurrentSearch ".config")) {
        $ProjectRoot = $CurrentSearch
    } else {
        $CurrentSearch = Split-Path -Parent $CurrentSearch
    }
}

if (-not $ProjectRoot) {
    Write-Error "Could not find Project Root (parent of .config folder)."
    exit 1
}

Set-Location $ProjectRoot
# -------------------------------

Write-Host "📍 Cleanup Context: $ProjectRoot" -ForegroundColor Gray
Write-Host "🧹 undoing LEGACY links..." -ForegroundColor Cyan

# 1. Unlink .cursorrules
if (Test-Path ".cursorrules") {
    $item = Get-Item ".cursorrules"
    if ($item.LinkType -eq "SymbolicLink") {
        Remove-Item ".cursorrules" -Force
        Write-Host "🗑️  Removed Link: .cursorrules" -ForegroundColor Green
    }
}

# 2. Unlink docs/templates
if (Test-Path "docs/templates") {
    $item = Get-Item "docs/templates"
    if ($item.LinkType -eq "SymbolicLink") {
        Remove-Item "docs/templates" -Force
        Write-Host "🗑️  Removed Link: docs/templates" -ForegroundColor Green
    }
}

# 3. Unlink Legacy Skills
$legacySkills = @("gemini-mastery") 
foreach ($skill in $legacySkills) {
    $path = ".agent/skills/$skill"
    if (Test-Path $path) {
        $item = Get-Item $path
        # Just check if it exists and delete it if it is a link/junction
        if ($item.LinkType -eq "SymbolicLink" -or $item.LinkType -eq "Junction") {
            Remove-Item $path -Force
            Write-Host "🗑️  Removed Link: $path" -ForegroundColor Green
        }
    }
}
Write-Host "✨ Legacy Cleanup Complete." -ForegroundColor Cyan
