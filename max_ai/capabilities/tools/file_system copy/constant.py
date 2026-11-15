"""Text the model reads for the file tools: tool and parameter descriptions,
limits and notes. Kept here so _toolset.py holds only the logic."""

# -------- NAMES -----------------------------------------------------------
READ_FILE = "read_file"
WRITE_FILE = "write_file"
EDIT_FILE = "edit_file"
DELETE_FILE = "delete_file"
LIST_DIRECTORY = "list_directory"
FIND_FILES = "find_files"
SEARCH_FILE = "search_file"

# -------- LIMITS -----------------------------------------------------------
DEFAULT_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_RESULTS = 200
MAX_SCANNED_FILES = 1000
MAX_SEARCH_FILE_BYTES = 128 * 1024
MAX_SEARCH_TOTAL_BYTES = 8 * 1024 * 1024
MAX_PATTERN_CHARS = 2000
MAX_MATCH_CHARS = 500

# -------- TOOL DESCRIPTIONS -----------------------------------------------------------
READ_FILE_DESCRIPTION = """Read a file from the user's workspace or skills.

- Returns the lines numbered like `cat -n`: number, tab, then the line. The numbers
  are not part of the file: never copy them into edit_file.
- Reads up to 2000 lines from the start by default. For a long file, read the part
  you need with offset and limit.
- Lines longer than 2000 characters are cut.
- A binary file (xlsx, pdf, images) returns only its size: open it with code.
- You must read a file before you edit, overwrite or delete it."""

WRITE_FILE_DESCRIPTION = """Create a text file in the workspace, or replace one.

- Creates the folders in the path that don't exist yet.
- To replace an existing file, read it with read_file first; prefer edit_file to
  change part of it.
- Only text: a binary format (xlsx, docx, pdf, png) must be made by running code."""

EDIT_FILE_DESCRIPTION = """Replace exact text in a workspace file.

- Read the file with read_file first.
- old_string must match the file exactly, with its whitespace and indentation, and
  without the line numbers read_file shows.
- old_string must appear once: add surrounding lines to make it unique, or set
  replace_all to change every occurrence (for example, to rename something)."""

DELETE_FILE_DESCRIPTION = """Delete a file from the workspace.

- Read it with read_file first (for a binary file that returns only its size).
- Only files: an empty folder disappears on its own."""

LIST_DIRECTORY_DESCRIPTION = """List the files and folders in one folder, with their sizes.

Use it to see what is in a folder. To find files anywhere by name use find_files;
to find text inside files use search_file."""

FIND_FILES_DESCRIPTION = """Find files by name with a glob pattern.

- `*.pdf` or `**/*.pdf` finds every PDF; `reports/*.md` looks only in reports/.
- A name without wildcards (`budget`) finds files whose name contains it.
- Returns the paths, to use with read_file and the other tools."""

SEARCH_FILE_DESCRIPTION = """Search the text inside files with a regular expression.

- Returns each matching line with its file and line number.
- Narrow it with path (a folder or file) and glob (for example `*.md`).
- Binary files are skipped."""

# -------- PARAMETER DESCRIPTIONS -----------------------------------------------------------
FILE_PATH = (
    "Path of the file, relative to the workspace (for example `reports/q3.md`). "
    "Paths that tools return, like `workspace/reports/q3.md`, work too; "
    "`skills/<name>/...` reads a skill."
)
WRITE_PATH = "Path of the file to write, relative to the workspace (for example `reports/q3.md`)."
OFFSET = "Line number to start reading from (1 is the first line)."
LIMIT_LINES = "How many lines to read."
CONTENT = "The whole content of the file."
OLD_STRING = "The exact text to replace."
NEW_STRING = "The text to put in its place (different from old_string)."
REPLACE_ALL = "Replace every occurrence of old_string instead of exactly one."
DIRECTORY = "Folder to list, relative to the workspace. Empty lists the workspace."
SEARCH_ROOT = (
    "Folder (or file) to search in, relative to the workspace. Empty searches the "
    "workspace and the skills."
)
GLOB_PATTERN = "Glob pattern to match file names or paths, like `*.xlsx` or `data/**/*.csv`."
REGEX = "Regular expression to look for, like `total` or `def \\w+\\(`."
GLOB_FILTER = "Only search files matching this glob, like `*.md`. Empty searches every file."
IGNORE_CASE = "Match upper and lower case alike."
LIMIT_RESULTS = "Most results to return."

# -------- NOTES -----------------------------------------------------------
BINARY = "Binary file: only its size is shown. Open it with code (for example Python with bash)."
EMPTY = "The file is empty."
MORE_LINES = "The file has {total} lines: continue with offset={next}."
NOT_READ = "Read {path} with read_file before changing it."
