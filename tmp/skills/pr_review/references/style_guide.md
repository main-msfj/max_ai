# Internal Code Style Guide

## General principles

- Prefer clarity over cleverness. Code is read more often than written.
- Function names describe what they return or do, not how they do it.
- Functions do one thing. If you need "and" to describe it, split it.
- Keep nesting shallow. Early returns over deeply nested conditionals.

## Python

- Names: `snake_case` for functions/variables, `PascalCase` for classes.
- Type hints on every public function. Optional on internal helpers.
- Docstrings on every public function: one-line summary + Args + Returns.
- Imports: stdlib, third-party, local — separated by blank lines.
- No `from foo import *`. Always explicit imports.

## TypeScript

- Names: `camelCase` for variables/functions, `PascalCase` for types.
- Prefer `interface` over `type` for object shapes.
- Always specify return types on exported functions.
- Use `unknown` over `any`. If you need `any`, justify in a comment.

## Commits

- Subject line under 72 characters.
- Imperative mood: "Add feature" not "Added feature".
- Body explains why, not what (the diff shows what).