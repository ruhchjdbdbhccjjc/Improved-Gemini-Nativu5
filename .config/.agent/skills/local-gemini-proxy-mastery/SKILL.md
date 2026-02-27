# Skill: Local Gemini Proxy Mastery

> **Category**: Service Documentation / Proxy Management
> **Port**: 8317 (Primary)
> **Protocol**: HTTP/1.1 with Bearer Token

## Overview

The Local Gemini Proxy is a custom translation and routing service that enables standard clients (like VS Code or Vite) to communicate with Gemini models via a localized endpoint.

## Configuration (Vite / Backend)

In `translationService.ts` or `vite.config.ts`, the proxy is typically configured as:

- **Base URL**: `http://localhost:8317`
- **Model**: `gemini-2.5-pro` (or similar)

## Health Check

To verify if the proxy is alive:

```powershell
Invoke-RestMethod -Uri "http://localhost:8317/health"
```

## Internal Workflow

1. **Request**: Client sends OpenAI-formatted payload to `localhost:8317`.
2. **Translation**: Proxy translates OpenAI payload to Google Gemini Native protocol.
3. **Execution**: Proxy attaches system cookies/tokens and calls Google APIs.
4. **Response**: Proxy streams or returns the translated response to the client.

## Security Rule

Requests to port 8317 must include the current session token if configured. Refer to [api-verification-diagnosis.md](../../workflows/api-verification-diagnosis.md) for troubleshooting authentication.
