# setup_brain.ps1 - The Professional Linker for Universal Smart Sync (Portable v3.0)
# Run this from your project root: powershell -ExecutionPolicy Bypass -File .config/setup_brain.ps1 .

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

Write-Host "🚀 Deploying Global Brain (Portable Mode)..." -ForegroundColor Cyan
Write-Host "📍 Project Root: $ProjectRoot" -ForegroundColor Gray

# 1. Root Protocol Sync (.cursorrules)
# Target: Relative to Project Root -> .config/.cursorrules
$SourceRule = ".config\.cursorrules" 
# Fallback check (absolute for check, relative for link)
if (-not (Test-Path "$ProjectRoot\.config\.cursorrules")) { 
    if (Test-Path "$ProjectRoot\.config\skills\.cursorrules") {
        $SourceRule = ".config\skills\.cursorrules"
    } else {
        $SourceRule = $null
    }
}

if ($SourceRule) {
    if (Test-Path ".cursorrules") {
        Remove-Item ".cursorrules" -Force
        Write-Host "♻️  Refreshed: Removed old .cursorrules" -ForegroundColor Gray
    }
    New-Item -ItemType SymbolicLink -Path ".cursorrules" -Target $SourceRule -Force | Out-Null
    Write-Host "✅ Linked: Global .cursorrules (Relative)" -ForegroundColor Green
}

# 2. Docs Templates Sync
if (-not (Test-Path "docs")) { mkdir "docs" | Out-Null }
# Target: Relative from docs/ -> ../.config/docs/templates
$SourceTemplates = "..\.config\docs\templates"
# Validating existence using Absolute path first
if (Test-Path "$ProjectRoot\.config\docs\templates") {
    if (-not (Test-Path "docs/templates")) {
        Push-Location "docs"
        try {
            New-Item -ItemType SymbolicLink -Path "templates" -Target $SourceTemplates -Force
            Write-Host "✅ Linked: docs/templates/ (Relative)" -ForegroundColor Green
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "ℹ️  Skipped: docs/templates already exists" -ForegroundColor Yellow
    }
}

# 3. Universal Folder Sync (.agent, .gemini)
# Ensures that these folders are fully symlinked to .config, so new files sync automatically.

function Ensure-FolderLink {
    param (
        [string]$LinkName,
        [string]$RelTarget  # Relative path from ProjectRoot to Target (e.g. ".config\.agent")
    )

    $LinkPath = Join-Path $ProjectRoot $LinkName
    $TargetAbsPath = Join-Path $ProjectRoot $RelTarget

    # Check if the source (in .config) actually exists
    if (-not (Test-Path $TargetAbsPath)) {
        Write-Host "⚠️  Source not found: $RelTarget (Skipping $LinkName)" -ForegroundColor Yellow
        return
    }

    # Helper to check if it's already a link pointing to the right place
    if (Test-Path $LinkPath) {
        $Item = Get-Item $LinkPath -Force
        # Check if it is a ReparsePoint (Symlink/Junction)
        if ($Item.Attributes -match "ReparsePoint") {
            # It's a link. Is it pointing to the right place?
            # Note: Resolving target can be tricky in PS, but we can usually trust if it exists as a link, we might want to refresh it or leave it.
            # For robustness: We will remove it and re-link if it's a link, to ensure target is correct.
            Write-Host "♻️  Unlinking: $LinkName (Refreshing Link)" -ForegroundColor Gray
            Remove-Item $LinkPath -Force -Recurse
        } else {
            # It is a REAL directory (not a link). We must back it up!
            $BackupName = "${LinkName}_BACKUP_" + (Get-Date -Format "yyyyMMdd_HHmmss")
            Rename-Item -Path $LinkPath -NewName $BackupName
            Write-Host "📦 Existing folder moved to backup: $BackupName" -ForegroundColor Yellow
        }
    }

    # Create the Link
    # Note: Target path for symbolic link should be relative if possible for portability, but Junctions usage absolute.
    # We will use Junctions (Requires target to be absolute path) for folders on Windows as they are more robust without admin rights.
    # Or SymbolicLink. Let's stick to SymbolicLink with relative target if possible, or Absolute if needed.
    # The original script used SymbolicLink. Let's continue with that but use ReparsePoint detection.
    
    # Using relative path for the target makes the link portable if the whole project moves.
    $RelTargetFromRoot = $RelTarget
    
    New-Item -ItemType SymbolicLink -Path $LinkName -Target $RelTargetFromRoot -Force | Out-Null
    Write-Host "✅ Linked Folder: $LinkName -> $RelTargetFromRoot" -ForegroundColor Green
}

function Ensure-RecursiveFileLink {
    param (
        [string]$DestRelPath,
        [string]$SourceRelPath
    )

    $DestRoot = Join-Path $ProjectRoot $DestRelPath
    $SourceRoot = Join-Path $ProjectRoot $SourceRelPath

    if (-not (Test-Path $SourceRoot)) {
        Write-Host "⚠️  Source not found: $SourceRelPath" -ForegroundColor Yellow
        return
    }

    # 1. Non-Destructive Upgrade: Handle Legacy Top-Level Folder-Link
    if (Test-Path $DestRoot) {
        $RootItem = Get-Item $DestRoot -Force
        # If Root itself is a ReparsePoint (legacy folder link), rename it to avoid collision
        if ($RootItem.Attributes -match "ReparsePoint") {
             $BackupName = (Split-Path $DestRoot -Leaf) + "_LEGACY_LINK"
             $BackupPath = Join-Path (Split-Path $DestRoot -Parent) $BackupName
             if (Test-Path $BackupPath) { 
                # CRITICAL: Detach link handle ONLY. 
                # 'cmd /c rmdir' is the Windows "Silver Bullet" - it never follows the link.
                $BackupAbs = (Get-Item $BackupPath).FullName
                cmd /c rmdir /s /q "$BackupAbs" 2>$null
                # Fallback if rmdir failed (e.g. if it was a file link)
                if (Test-Path $BackupPath) { Remove-Item $BackupPath -Force }
             }
             Rename-Item -Path $DestRoot -NewName $BackupName
             Write-Host "🔄 Upgraded $DestRelPath from folder-link to merge-folder" -ForegroundColor Cyan
        }
    }
    
    # Ensure real directory exists (will NOT wipe existing items)
    if (-not (Test-Path $DestRoot)) {
        New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null
    }

    # 2. Walk Source
    $SourceItems = Get-ChildItem -Path $SourceRoot -Recurse

    foreach ($SourceItem in $SourceItems) {
        # Calculate Relative Path from Source Root
        $RelPath = $SourceItem.FullName.Substring($SourceRoot.Length + 1)
        $DestPath = Join-Path $DestRoot $RelPath
        
        if ($SourceItem.PSIsContainer) {
            # Directory -> Create Real Directory if missing
            if (-not (Test-Path $DestPath)) {
                New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
            }
        } else {
            # FILE Handling: Update Link or Presere Local
            if (Test-Path $DestPath) {
                $Existing = Get-Item $DestPath -Force
                if ($Existing.Attributes -match "ReparsePoint") {
                    # Stale link -> Detach handle safely
                    $DestAbs = $Existing.FullName
                    if ($Existing.PSIsContainer) {
                        cmd /c rmdir /s /q "$DestAbs" 2>$null
                    }
                    if (Test-Path $DestPath) { Remove-Item $DestPath -Force }
                } else {
                    # REAL local file -> Skip to preserve "Local Special" version
                    Write-Host "🛡️  Preserving Local File: $RelPath (Global link skipped)" -ForegroundColor Yellow
                    continue
                }
            }

            # Calculate Depth required to get back to Project Root from here
            $DestDirFull = Split-Path $DestPath -Parent
            
            # Count steps to ProjectRoot
            $Current = $DestDirFull
            $Depth = 0
            while ($Current -and $Current -ne $ProjectRoot -and (Split-Path $Current -Parent)) {
                $Current = Split-Path $Current -Parent
                $Depth++
            }
            
            $DotDots = "..\" * $Depth
            $RelativeTarget = Join-Path $DotDots (Join-Path $SourceRelPath $RelPath)

            # Create Link
            $ParentDir = Split-Path $DestPath -Parent
            if (-not (Test-Path $ParentDir)) { New-Item -ItemType Directory -Path $ParentDir -Force | Out-Null }
            
            Push-Location $ParentDir
            try {
                $LeafName = Split-Path $DestPath -Leaf
                New-Item -ItemType SymbolicLink -Path $LeafName -Target $RelativeTarget -Force | Out-Null
                Write-Host "✅ Linked: $RelPath" -ForegroundColor DarkGray
            } catch {
                Write-Warning "Failed to link ${LeafName}: $($_.Exception.Message)"
            } finally {
                Pop-Location
            }
        }
    }
    Write-Host "✅ Synced (File-Links): $DestRelPath" -ForegroundColor Green
}

# Execute Sync for Critical Folders
# .agent must be a REAL FOLDER with File-Level Links for Slash Commands to work
Ensure-RecursiveFileLink -DestRelPath ".agent" -SourceRelPath ".config\.agent"

# .gemini must also be a REAL FOLDER with File-Level Links to preserve local logs/artifacts
Ensure-RecursiveFileLink -DestRelPath ".gemini" -SourceRelPath ".config\.gemini"

Write-Host "`n🎉 Brain Connection Complete!" -ForegroundColor Cyan

