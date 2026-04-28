---
name: bug_triage
description: Triage incoming bug reports — extract severity signals and suggest priority.
approval_overrides:
  extract_severity_signals: auto_approval
  suggest_priority: auto_approval
resources:
  severity_levels.md: Definitions of severity levels (critical, high, medium, low) with examples.
---

# Bug Triage Skill

When the user shares a bug report or asks you to triage one:

1. **Extract signals** with `extract_severity_signals` from the bug text.
   This pulls out keywords and patterns indicating severity.
2. **Determine severity** based on the signals: critical, high, medium, or low.
   If unclear, load `severity_levels.md` for definitions and examples.
3. **Estimate frequency**: ask the user how often the bug occurs, or infer
   from the report ("every request", "intermittent", "once").
4. **Run `suggest_priority`** with severity + frequency → recommended P0–P3.
5. **Summarize**: severity, frequency, suggested priority, and 1–2 next steps.

## When to load `severity_levels.md`

- The user uses ambiguous language ("kind of bad", "users complain").
- The signals returned are mixed or unclear.
- The user disagrees with your severity assessment and asks why.

Skip the reference file for clear-cut cases ("server returns 500 on every login").