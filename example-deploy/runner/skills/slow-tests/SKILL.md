---
name: slow-tests
description: Runs the project's test suite, which takes about a minute. Use it when the user asks to run the tests.
---
# slow-tests

1. Run `python3 scripts/run_tests.py`, giving the script by its full path
   inside this skill folder. It takes about a minute: start it with
   `background=true`, never as a command you wait for.
2. Tell the user it is running and go on with whatever else they asked.
   You are told when it ends; do not call `task_output` again and again to wait.
3. When it ends, say how many tests passed and name each one that failed,
   from its last lines. Exit code 1 means some test failed.

To run it slower or faster, add the seconds each test takes:
`python3 scripts/run_tests.py 20` takes about four minutes.
