# Runs inside the sandbox for bash. The tool puts the values on top:
# __start, __workspace, __log, __max, __mark, __check, __seen, __command.

finish() {
    code=$?
    # 1. Print the output; a long one keeps its start and its end.
    size=$(wc -c < "$__log")
    if [ "$size" -le "$__max" ]; then
        cat "$__log"
        rm -f "$__log"
    else
        half=$((__max / 2))
        head -c "$half" "$__log"
        printf '\n[... %s bytes cut; the full output is in %s ...]\n' "$((size - __max))" "$__log"
        tail -c "$half" "$__log"
    fi
    # 2. Say where the command ended, behind a mark only the tool knows.
    printf '\n%s%s' "$__mark" "$(pwd -P)"
    # 3. Say which files the model had read are no longer the same.
    printf '%s' "$__mark"
    python3 -c "$__check" "$__seen" 2>/dev/null
    exit "$code"
}
trap finish EXIT

# Start where the last command ended, or in the workspace (new in a new session).
mkdir -p "$(dirname "$__log")" "$__workspace"
cd "$__start" 2>/dev/null || cd "$__workspace" || exit 1
eval "$__command" > "$__log" 2>&1 < /dev/null
