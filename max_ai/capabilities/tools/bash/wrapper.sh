# The script the harness wraps around every bash command.
#
# build_script() puts the values on the lines above this one: WORKSPACE,
# SKILLS, __maxai_cwd, __maxai_log, __maxai_max_bytes and __maxai_command.
# After the output it prints two marks the tool strips (see constant.py):
# the workspace files the command changed and the final working directory.

readonly __maxai_cwd __maxai_log __maxai_max_bytes __maxai_command
export WORKSPACE SKILLS
exec 3>&1 4>&2
mkdir -p "$WORKSPACE" "${__maxai_log%/*}"
cd "$__maxai_cwd" 2>/dev/null || cd "$WORKSPACE"

# One line per workspace file: path, size, mtime. .git is skipped.
__maxai_snapshot() {
    find "$WORKSPACE" -name .git -prune -o -type f -printf '%P\t%s\t%T@\n' 2>/dev/null |
        LC_ALL=C sort
}

# Runs on every exit, also after `exit` in the command or at the time limit
# (TERM): output (cut in the middle when long), changed files, working dir.
__maxai_finish() {
    __maxai_code=$?
    exec 1>&3 2>&4
    __maxai_size=$(wc -c < "$__maxai_log" 2>/dev/null || echo 0)
    if [ "$__maxai_size" -gt "$__maxai_max_bytes" ]; then
        __maxai_half=$((__maxai_max_bytes / 2))
        head -c "$__maxai_half" "$__maxai_log"
        printf '\n\n[... %s bytes cut; full output in %s ...]\n\n' \
            "$((__maxai_size - 2 * __maxai_half))" "$__maxai_log"
        tail -c "$__maxai_half" "$__maxai_log"
    else
        cat "$__maxai_log" 2>/dev/null
        rm -f "$__maxai_log"
    fi
    __maxai_snapshot > "$__maxai_log.after"
    printf '\n__MAXAI_FILES__='
    LC_ALL=C comm -3 "$__maxai_log.before" "$__maxai_log.after" 2>/dev/null | head -n 1000
    rm -f "$__maxai_log.before" "$__maxai_log.after"
    printf '\n__MAXAI_CWD__=%s' "$(pwd -P)"
    exit "$__maxai_code"
}
trap __maxai_finish EXIT
trap 'exit 143' TERM

__maxai_snapshot > "$__maxai_log.before"
# Same shell, so `cd` persists into the trap. The newline before } lets the
# command end in a comment.
{ eval "$__maxai_command"
} < /dev/null > "$__maxai_log" 2>&1
