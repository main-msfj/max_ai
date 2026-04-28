"""Naming convention checks."""


def check_naming_conventions(filename: str, language: str) -> str:
    """Check if a filename follows the conventions for its language.

    Returns a short verdict string. Use this on each modified file in
    a PR to flag obvious naming issues before deeper review.

    Args:
        filename: The filename to check, including any path components.
        language: The language identifier (e.g. "python", "typescript",
            "go", "rust").

    Returns:
        A short string: "ok" if conventions match, otherwise a brief
        description of the issue.
    """
    base = filename.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    lang = language.lower().strip()

    if lang == "python":
        if not stem.replace("_", "").islower() or "-" in stem:
            return f"python files should be snake_case, got {stem!r}"
        return "ok"

    if lang in ("typescript", "javascript"):
        if "_" in stem:
            return f"{lang} files should be camelCase or kebab-case, got {stem!r}"
        return "ok"

    if lang == "go":
        if not stem.islower() or "_" in stem or "-" in stem:
            return f"go files should be lowercase without separators, got {stem!r}"
        return "ok"

    if lang == "rust":
        if not stem.replace("_", "").islower() or "-" in stem:
            return f"rust files should be snake_case, got {stem!r}"
        return "ok"

    return f"unknown language {language!r}, skipping check"