# write_file inside the environment. _common.py goes in front of this file.
#
# A new file needs nothing. A file that already exists is only replaced when
# the model read its current content (see check() in _common.py).


def run(full, request):
    data = request["content"].encode("utf-8")
    created = not os.path.exists(full)
    if not created:
        with open(full, "rb") as file:
            check(file.read(), request.get("stamp"))
    save(full, data)
    # The new stamp: the model can edit what it just wrote without reading it.
    return {"created": created, "bytes": len(data), "stamp": stamp_of(data)}


main(run)
