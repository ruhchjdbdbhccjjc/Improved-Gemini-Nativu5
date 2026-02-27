# Skill: Gemini Proxy Server API mastery

> **Category**: API Documentation / Integration
> **Port**: 30000 (Current Active) / 8317 (Local Proxy)
> **Protocol**: HTTP/1.1 or HTTP/2

## Overview

This skill documents the internal Gemini Proxy Server endpoints. This server acts as an intermediary between the Google Gemini Web API and various clients (IDE, Browser, or external scripts).

## Model Compatibility (CRITICAL)

Based on practical verification:

- **`gemini-2.0-pro-exp-02-05`**: ✅ **BEST** for general use.
- **`gemini-2.0-flash-exp`**: ✅ **WORKS** (Good for speed tests).
- **`gemini-3.0-pro`**: ✅ **WORKS** for Text & Image (Tool Delegation).
- **`gemini-2.5-pro`**: ❌ **IMAGE FAILED** due to upstream delegation errors. Only use for Text.

## Verification Logic

We use a graduated verification suite to ensure server mastery:

1. **Direct**: Bypassing HTTP to check library patches.
2. **Master**: [verify_api_mastery.py](../../tests/verify_api_mastery.py) checks the live server for Text/Image stability across models.

## Core Endpoints

### 1. Health & Status

- **URL**: `GET /health`
- **Description**: Returns the status of initialized Gemini clients and LMDB storage statistics.
- **Response**: `HealthCheckResponse` (JSON)

### 2. OpenAI Compatible Chat

- **URL**: `POST /v1/chat/completions`
- **Description**: Standard OpenAI compatibility layer. Supports:
  - Text generation
  - Streaming (`stream: true`)
  - Image generation (via assistant message triggers)
  - Tool calling (simulated via system prompts)

### 3. Native Gemini Protocol (v1beta)

- **URL**: `POST /v1beta/models/generateContent`
- **URL**: `POST /v1beta/models/{model}:generateContent`
- **Description**: Emulates the official Google Gemini API structure.
- **Usage**: Used for raw data extraction and specific model forcing (e.g., `gemini-3.0-pro`).

### 4. Image Assets

- **URL**: `GET /images/{filename}`
- **Query Param**: `token={token}` (Required for security)

- **Description**: Serves AI-generated images stored in the temporary system directory.

## Protocol Quirks & Fixes (v2.8)

1.  **Init Text Disable**: The server MUST disable `send_init_text` during session creation to prevent hangs with Gemini 3.0.
2.  **Model Mapping**: `gemini-2.0` models are currently mapped to `G_2_5_FLASH` to ensure `gemini_webapi` compatibility.

## Implementation Details

- **Router Location**: `app/server/`
- **Native Logic**: `app/server/native/`
- **Middleware**: `app/server/middleware.py` (Handles CORS, API keys, and image cleanup)

## Documentation Rule

Any new API route added to the project **MUST** be documented in this skill immediately. This ensures that the AI agent and other contributors have up-to-date knowledge of the communication interface.
