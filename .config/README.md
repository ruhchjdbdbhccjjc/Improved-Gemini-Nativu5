# Agent Skills Best Practices

> **Source**: [Anthropic Agent Skills Best Practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)

This directory hosts the **Global Brain** for the Universal Smart Sync Protocol. Follow these core principles when adding new skills or configurations.

## 1. Core Principles
- **Concise is Key**: Agents perform better with short, imperative instructions. Avoid fluff.
- **Progressive Disclosure**: Do not dump everything into one file.
  - Use a "Router" (like `.cursorrules`) to point to specific "Details" (like `.agent/skills/wxt-mastery/SKILL.md`).
- **Platform Agnostic**: Use relative paths (`../../`) instead of OS-specific absolute paths.

## 2. Skill Structure (The "Mastery" Pattern)
When creating a new skill (e.g., `agent-skills/gemini-mastery/SKILL.md`), follow this structure:
1.  **References Header**: Official links (Docs, GitHub, Examples) at the very top.
2.  **Core Concepts**: What is this tech? (Briefly).
3.  **Critical Patterns**: "Copy-pasteable" code snippets for common tasks.
4.  **Gotchas**: Specific warnings about common mistakes.
5.  **Debugging**: How to search for errors (Local vs Remote).

## 3. Workflows
- **Use Workflows**: For complex, multi-step tasks, define a clear step-by-step process.
- **Feedback Loops**: Encourage the Agent to verify its output (e.g., "Check local rules first").

## 4. Maintenance
- **Iterate**: Skills are living documents. Update them when the Agent makes a mistake.
- **Visuals**: Use Mermaid diagrams for complex logic (quote your labels!).
