---
description: Standardized procedure for identifying, analyzing, and resolving project issues
---

# Workflow: Issue Resolution Logic

> **Purpose**: Standardized procedure for identifying, analyzing, and resolving project issues.

## Logic Map

```text
[ Issue Detected ]
       |
       v
[ Search Codebase ] --> [ Identify Culprit ]
       |
       v
[ Analyze Logs ] ----> [ Determine Cause ]
       |
       v
[ Propose Fix ] -----> [ User Review ]
       |
       v
[ Implement ] --------> [ Verify ]
```

## Steps

1. Use `grep_search` or `codebase_search` to find relevant files.
2. Examine recent logs in `.system_generated/logs/`.
3. Create an implementation plan.
4. Apply fixes and run tests.
