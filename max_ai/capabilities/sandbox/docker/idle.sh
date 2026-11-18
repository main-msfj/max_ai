# The container's main process: when it ends, the container ends with it.
# $1 is how many seconds it lives without a command; each command sends USR1.

idle=0
trap 'idle=0' USR1
while [ "$idle" -lt "$1" ]; do
    # A signal that comes during the sleep is heard when it ends.
    sleep 5
    idle=$((idle + 5))
done
