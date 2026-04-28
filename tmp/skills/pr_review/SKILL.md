---
name: pr_review
description: Review pull requests for quality, conventions, and risk.
approval_overrides:
  analyze_diff: auto_approval
  check_naming_conventions: auto_approval
resources:
  checklist.md: Detailed PR review checklist covering security, tests, and edge cases.
  style_guide.md: Internal code style guide for naming, formatting, and structure.
---

# PR Review Skill

When the user asks you to review a pull request:

1. **Get the diff.** Either the user pastes it, or you ask for the PR URL.
2. **Run `analyze_diff`** to get stats: lines changed, files touched, complexity signals.
3. **For each modified file, run `check_naming_conventions`** with the file's language.
4. **Summarize findings** in a short report.
5. **For deeper review** (security-sensitive changes, complex logic), load `checklist.md`
   via `read_skill_resource` and walk through it.
6. **For style questions** that aren't obvious from the conventions check, load `style_guide.md`.

## When to load reference files

- **`checklist.md`**: load when the diff touches authentication, payments, data handling,
  or any user-facing API. Walk through it section by section.
- **`style_guide.md`**: load when the user asks specifically about style, or when
  conventions check flags ambiguous cases.

Don't load reference files for trivial PRs (typo fixes, comment-only changes, etc.).
Use them when the depth justifies the token cost.