# edit_file inside the environment. _common.py goes in front of this file.
#
# 1. The file must be exactly what the model last read (check()).
# 2. old_string is looked for three ways, from strict to forgiving.
# 3. One match is replaced; several need replace_all.
# 4. The file is saved in one step and its new stamp returned, so the next
#    edit works without reading again.

# Characters models and editors swap without meaning to; one for one, so a
# position in the changed text is the same position in the real text.
TYPOGRAPHIC = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})


def exact(text, old, new):
    """Every place ``old`` appears, as (start, end)."""
    spans, start = [], text.find(old)
    while start != -1:
        spans.append((start, start + len(old)))
        start = text.find(old, start + len(old))
    return spans, new


def typographic(text, old, new):
    """The same, reading curly quotes, long dashes and hard spaces as plain."""
    return exact(text.translate(TYPOGRAPHIC), old.translate(TYPOGRAPHIC), new)[0], new


def trailing_spaces(text, old, new):
    """Whole lines that match once the spaces at their end are ignored."""
    if old.endswith("\n"):
        if not new.endswith("\n"):
            return [], new  # would join two lines: too easy to get wrong
        old, new = old[:-1], new[:-1]
    lines = text.split("\n")
    wanted = [line.rstrip() for line in old.split("\n")]
    starts, position = [], 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    spans, index = [], 0
    while index + len(wanted) <= len(lines):
        if [line.rstrip() for line in lines[index:index + len(wanted)]] == wanted:
            last = index + len(wanted) - 1
            spans.append((starts[index], starts[last] + len(lines[last])))
            index += len(wanted)
        else:
            index += 1
    return spans, new


WAYS = (("exact", exact), ("typographic", typographic), ("trailing spaces", trailing_spaces))


def run(full, request):
    old, new = request["old_string"], request["new_string"]
    with open(full, "rb") as file:
        data = file.read()
    check(data, request.get("stamp"))
    text = data.decode("utf-8")
    # Work with \n and give Windows line endings back when saving.
    windows = "\r\n" in text
    if windows:
        text, old, new = (s.replace("\r\n", "\n") for s in (text, old, new))

    for name, find in WAYS:
        spans, replacement = find(text, old, new)
        if spans:
            break
    else:
        if new and new in text:
            raise Refused("old_string was not found, but new_string is already there: "
                          "this edit may already be applied")
        raise Refused("old_string was not found; read the file again and copy the text exactly")
    if len(spans) > 1 and not request.get("replace_all"):
        raise Refused(f"old_string appears {len(spans)} times; add the lines around it "
                      "to make it unique, or set replace_all")

    for start, end in reversed(spans):
        text = text[:start] + replacement + text[end:]
    if windows:
        text = text.replace("\n", "\r\n")
    data = text.encode("utf-8")
    save(full, data)
    answer = {"replaced": len(spans), "stamp": stamp_of(data)}
    if name != "exact":
        answer["note"] = f"matched ignoring {name} differences"
    return answer


main(run)
