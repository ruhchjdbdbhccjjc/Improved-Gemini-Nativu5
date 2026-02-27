# Gemini Agent Instructions

> **This file is a reference pointer.** The actual agent rules are defined in the parent `.cursorrules` file.

For the complete Universal Smart Sync Protocol, see:
- **[.cursorrules](../.cursorrules)** - The source of truth for all agent behavior.

## Why This File Exists

Gemini IDE looks for `.gemini/GEMINI.md` at the project root. This file is symlinked from `.config/.gemini/GEMINI.md` so that:
1. The agent rules are centralized in `.config/.cursorrules`
2. All relative paths in `.cursorrules` work correctly (they resolve from project root)
3. Changes to `.cursorrules` automatically apply everywhere

## Quick Reference

The `.cursorrules` file defines:
- **Section 0**: Write Protection Protocol
- **Section 1**: Documentation (Deep Merge & Link)
- **Section 2**: Agent Folder & Workspace Management
- **Section 3**: Cleanup & Preservation
- **Section 4**: Final Verification
- **Section 5**: AI Code Generation & Modularity
- **Section 6**: Communication & Explanation
