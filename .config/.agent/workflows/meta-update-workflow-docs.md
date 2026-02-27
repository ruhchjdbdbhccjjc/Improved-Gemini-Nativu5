---
description: Synchronize master-workflow-registry and documentation with .agent/workflows/ contents
---

# Workflow: Meta-Update Workflow Documentation

> **Purpose**: Ensures that the `master-workflow-registry.md` and human-readable documentation (like `project_workflow_details.md`) stay in sync with the actual files in `.agent/workflows/`.

## Logic Map

```text
[ Update in .agent/workflows/ ]
             |
             v
[ Trigger Sync ] --> [ Update master-workflow-registry.md ]
             |
             +------> [ Update docs/project_workflow_details.md ]
```

## Execution Steps

1. **Verify Inventory**: Run a directory listing of `.agent/workflows/`.
2. **Update Registry**: Ensure every `.md` file in the folder has a corresponding entry in [master-workflow-registry.md](master-workflow-registry.md).
3. **Verify Links**: Ensure all relative links in [project_workflow_details.md](../../docs/project_workflow_details.md) point correctly to the `.agent` folder.
4. **Maintenance Filter**: Use the "Special Project" filter to avoid documenting transient or session-specific temporary files.
