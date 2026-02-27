---
description: Universal Smart Sync Protocol (v2.5) - Finalize session, update docs, and cleanup
---

# Universal Smart Sync Protocol (v2.5)

> **Purpose**: Standardized session finalization to ensure documentation deep-merge and environment cleanup.

## Logic Map

```text
[ Work Complete ]
       |
       v
[ Documentation ] ----> [ deep_merge features/issues ]
       |
       v
[ Aggressive Cleanup ] -> [ delete temporary assets ]
       |
       v
[ Preservation ] ------> [ keep source & permanent docs ]
```

## Protocol Steps

1. **Documentation (Deep Merge & Link)**:
   - Update `docs/FEATURES.md`, `docs/KNOWN_ISSUES.md`, and `docs/ARCHITECTURE.md`.
   - Create feature-specific detail files in `docs/features/`.
   - Create bug-fix retrospectives in `docs/issues/` with ASCII logic maps.
   - Link specific functions using `[Function Name](../path/to/file.ts#L123)`.

2. **Cleanup (Aggressive)**:
   - Delete generated session assets (`*.jpg`, `*.png`, `*.mp4`, `*.log`).
   - Delete root-level session files (`walkthrough.md`, `implementation_plan.md`, etc.) EXCEPT `README.md`.

3. **Preservation**:
   - Ensure `docs/` folder structure and source code remain intact.
   - Verify all links in updated documentation are valid.
