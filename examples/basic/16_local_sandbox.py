"""Start a LocalSandbox inside root-test/ and show the .agent/ it creates.

Run from the repository root: python examples/basic/16_local_sandbox.py
"""


from max_ai.capabilities.sandbox.local import LocalSandbox

sandbox = LocalSandbox(root="root-test")

print("root:", sandbox.root)
for path in sorted(sandbox.root.rglob("*")):
    print(" ", path.relative_to(sandbox.root.parent))

restored = LocalSandbox.from_json(sandbox.to_json())
print("restored from json:", restored.root == sandbox.root)
