"""Create the ``.agent/`` skeleton inside an environment.

Standard library only: one JSON request on stdin, one JSON answer on
stdout. Running it again changes nothing.
"""

import json
import os
import sys


def main():
    request = json.load(sys.stdin)
    for folder in request["folders"]:
        os.makedirs(folder, exist_ok=True)
    # "x" fails if the file exists: an existing setting.json is never overwritten.
    try:
        with open(request["settings"], "x", encoding="utf-8") as file:
            file.write(json.dumps(request["content"], indent=2) + "\n")
    except FileExistsError:
        pass
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
