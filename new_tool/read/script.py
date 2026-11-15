# read_file inside the environment. _common.py goes in front of this file.
#
# Goes through the file line by line, so a big file is never loaded whole.
# The stamp covers the WHOLE file even when only some lines are returned:
# reading lines 1-50 is enough to edit line 900 later.

LIMIT = 2000


def run(full, request):
    offset = request.get("offset") or 1
    limit = request.get("limit") or LIMIT
    digest = hashlib.sha256()
    lines = []
    total = 0
    with open(full, "rb") as file:
        for total, raw in enumerate(file, start=1):
            digest.update(raw)
            if offset <= total < offset + limit:
                text = raw.decode("utf-8").rstrip("\r\n")
                lines.append(f"{total:>6}\t{text}")
    return {"content": "\n".join(lines), "total_lines": total, "stamp": digest.hexdigest()}


main(run)
