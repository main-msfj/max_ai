#!/usr/bin/env python3
"""Remove quoted type annotations in files that use
`from __future__ import annotations` by parsing the string
annotation and replacing it with the real AST node.

This script rewrites files in place. It attempts to preserve
content but will rewrite the file using `ast.unparse` (comments
may be lost).
"""
import ast
from pathlib import Path
import sys


def parse_annotation_str(s: str):
    try:
        node = ast.parse(s, mode="eval")
        return node.body
    except Exception:
        return None


class QuoteAnnotationRemover(ast.NodeTransformer):
    def _replace_if_quoted(self, ann):
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            new = parse_annotation_str(ann.value)
            if new is not None:
                return new
        return ann

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.returns = self._replace_if_quoted(node.returns) if node.returns else None
        for a in node.args.args + node.args.kwonlyargs:
            a.annotation = self._replace_if_quoted(a.annotation) if a.annotation else None
        if node.args.vararg:
            node.args.vararg.annotation = self._replace_if_quoted(node.args.vararg.annotation) if node.args.vararg.annotation else None
        if node.args.kwarg:
            node.args.kwarg.annotation = self._replace_if_quoted(node.args.kwarg.annotation) if node.args.kwarg.annotation else None
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        node.annotation = self._replace_if_quoted(node.annotation) if node.annotation else None
        return node

    def visit_arg(self, node):
        node.annotation = self._replace_if_quoted(node.annotation) if node.annotation else None
        return node


def process_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in text:
        return False
    try:
        tree = ast.parse(text)
    except Exception as e:
        print(f"Skipping {path}: parse error: {e}")
        return False

    transformer = QuoteAnnotationRemover()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        new_src = ast.unparse(new_tree)
    except Exception as e:
        print(f"Skipping {path}: unparse error: {e}")
        return False

    if new_src != text:
        path.write_text(new_src, encoding="utf-8")
        print(f"Updated: {path}")
        return True
    return False


def main(root):
    p = Path(root)
    py_files = list(p.rglob("*.py"))
    modified = []
    for f in py_files:
        try:
            if process_file(f):
                modified.append(str(f))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    print("\nSummary:\n")
    if modified:
        for m in modified:
            print(m)
    else:
        print("No files modified.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)
