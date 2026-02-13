#!/usr/bin/env python3
"""
Launcher for TestCase Generator MCP Server.
Locates and executes server/main.py from either:
1. Kiro power repos (~/.kiro/powers/repos/) — installed power
2. Current working directory — development mode
3. Same directory as this launcher — fallback
"""
import os
import sys


def find_server_script():
    """Find main.py in order of priority."""
    # 1. Check Kiro power repos (installed power)
    home = os.path.expanduser("~")
    repos = os.path.join(home, ".kiro", "powers", "repos")
    if os.path.isdir(repos):
        for d in os.listdir(repos):
            pm = os.path.join(repos, d, "POWER.md")
            sm = os.path.join(repos, d, "server", "main.py")
            if os.path.isfile(pm) and os.path.isfile(sm):
                with open(pm, encoding="utf-8") as f:
                    header = f.read(512)
                if "test-case-generator" in header:
                    return sm

    # 2. Check cwd/server/main.py (development mode)
    cwd_script = os.path.join(os.getcwd(), "server", "main.py")
    if os.path.isfile(cwd_script):
        return cwd_script

    # 3. Check same directory as launcher (fallback)
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    sibling_script = os.path.join(launcher_dir, "main.py")
    if os.path.isfile(sibling_script):
        return sibling_script

    return None


def main():
    script = find_server_script()
    if not script:
        sys.stderr.write("ERROR: test-case-generator main.py not found\n")
        sys.stderr.write("Searched:\n")
        sys.stderr.write(f"  1. ~/.kiro/powers/repos/*/server/main.py\n")
        sys.stderr.write(f"  2. {os.getcwd()}/server/main.py\n")
        sys.stderr.write(f"  3. {os.path.dirname(os.path.abspath(__file__))}/main.py\n")
        sys.stderr.flush()
        sys.exit(1)

    sys.stderr.write(f"[Launcher] Found server: {script}\n")
    sys.stderr.flush()

    # Set up sys.argv for main.py
    sys.argv = ["main.py", "--workspace", os.getcwd()]

    # Execute main.py in its own context
    with open(script, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, script, "exec"), {"__name__": "__main__", "__file__": script})


if __name__ == "__main__":
    main()
