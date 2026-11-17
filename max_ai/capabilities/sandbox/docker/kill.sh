# Runs inside the container when a command is cancelled from outside.
# $1 is the mark the command carries as its last argument.

for process in /proc/[0-9]*; do
    # 1. Only `timeout`, which leads the group of the command it runs.
    [ "$(cat "$process/comm" 2>/dev/null)" = timeout ] || continue
    # 2. Only the one that carries this mark.
    grep -q -- "$1" "$process/cmdline" 2>/dev/null || continue
    # 3. Kill its whole group: the command and everything it started.
    kill -KILL -- "-${process#/proc/}"
done
