"""Strings and limits of the bash tool, in one place."""

# --- What the model reads -------------------------------------------------

DESCRIPTION = (
    "Run a shell command. Commands start in the user's files; the working "
    "directory persists between calls, environment variables do not. "
    "Returns the exit code and the output (stdout and stderr together); "
    "long output is cut in the middle and the full text saved to a file."
)
COMMAND_DESCRIPTION = (
    "Shell command to run. Permissions are evaluated across the complete "
    "compound command; complex shell syntax may require approval."
)
INTENT_DESCRIPTION = (
    "Clear, concise description of what this command does, in plain "
    "language, so a human deciding whether to approve it understands the "
    "intent without reading the raw command."
)

# --- Errors and notes in the result ---------------------------------------

NO_ENVIRONMENT = "bash needs an execution environment; run it through an Agent."
INVALID_PARAMETERS = "Invalid bash parameters."
EMPTY_COMMAND = "command cannot be empty."
LEFT_WORKSPACE = (
    "Shell cwd was reset: {cwd} is outside $WORKSPACE, $SKILLS and /tmp, so the "
    "next command starts in $WORKSPACE."
)
TIMED_OUT = "Stopped after {timeout:g} seconds, the time limit for a command."

# --- Script marks -----------------------------------------------------------
# wrapper.sh prints them after the output; the tool strips them, so the
# model never sees them. Keep them in step with wrapper.sh.

FILES_MARK = "\n__MAXAI_FILES__="  # then `comm -3` of the workspace snapshots
CWD_MARK = "\n__MAXAI_CWD__="  # then the final working directory

# --- Limits and places --------------------------------------------------------

MAX_OUTPUT_BYTES = 30_000  # output kept for the model; longer is cut in the middle
MAX_LISTED = 50  # changed paths per group shown to the model
# Full output of cut commands, one folder per conversation; outside the
# user's files, so it is never synced or listed.
LOG_DIR = "/tmp/maxai-bash"
