---
description: Verify the local Gemini proxy (port 8317) is online and healthy
---

# Workflow: Verify Gemini Proxy

> **Purpose**: Ensures the local proxy on port 8317 is alive and responding.

## Logic Map

```text
[ Test Start ]
       |
       v
[ Check Port 8317 ] --(Down)--> [ Error: Start Proxy ]
       |
       +--(Up)--> [ Check Health Endpoint ]
                     |
                     +--(Fail)--> [ Error: Check Logs ]
                     |
                     +--(Pass)--> [ Success ]
```

## Steps

1. Run `curl http://localhost:8317/health`.
2. Verify response contains `status: ok`.
