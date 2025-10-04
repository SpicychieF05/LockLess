#!/usr/bin/env python3
"""
Attach a remote repository (defaults to https://github.com/SpicychieF05/LockLess) and optionally run the stage-and-push script.

Usage examples:
  # Dry-run: show what would be done
  python scripts/attach_and_push.py --dry-run

  # Attach remote and then run stage_and_push for 1000 files
  python scripts/attach_and_push.py --remote-url https://github.com/SpicychieF05/LockLess --run-stage --stage-args "--n 1000 -m \"Batch commit\" --yes"

Notes:
- This script must be run locally where git is available and you have credentials.
- It will add the remote name you specify (default: origin) if missing, or update the URL if the remote exists but points elsewhere (with confirmation unless --yes).
- If --run-stage is provided, it will call the existing scripts/stage_and_push.py script (which must be present and executable with the local Python interpreter).
"""

import argparse
import subprocess
import sys
from typing import List


def run(cmd: List[str], capture_output=False, check=True):
    if capture_output:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, check=check)


def remote_exists(name: str) -> bool:
    p = run(["git", "remote"], capture_output=True, check=False)
    if p.returncode != 0:
        print(p.stderr or "git remote failed")
        sys.exit(1)
    remotes = p.stdout.splitlines()
    return name in remotes


def get_remote_url(name: str) -> str:
    p = run(["git", "remote", "get-url", name],
            capture_output=True, check=False)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()  # type: ignore


def add_or_update_remote(name: str, url: str, assume_yes: bool, dry_run: bool):
    exists = remote_exists(name)
    if exists:
        current = get_remote_url(name)
        print(f"Remote '{name}' exists with URL: {current}")
        if current == url:
            print("Remote already points to the requested URL. No change needed.")
            return
        if dry_run:
            print(f"Dry-run: would run: git remote set-url {name} {url}")
            return
        if not assume_yes:
            ans = input(
                f"Remote '{name}' points to {current}. Replace with {url}? [y/N]: ")
            if ans.lower() != 'y':
                print("Not changing remote.")
                return
        print(f"Updating remote '{name}' URL to {url}")
        run(["git", "remote", "set-url", name, url])
    else:
        if dry_run:
            print(f"Dry-run: would run: git remote add {name} {url}")
            return
        print(f"Adding remote '{name}' -> {url}")
        run(["git", "remote", "add", name, url])


def run_stage_script(stage_args: str, dry_run: bool):
    # stage_args is a single string of CLI args to pass to the stage script
    import shlex
    args_list = shlex.split(stage_args) if stage_args else []
    cmd = [sys.executable, "scripts/stage_and_push.py"] + args_list
    if dry_run:
        print("Dry-run: would run:", " ".join(cmd))
        return
    print("Running stage script:", " ".join(cmd))
    rc = run(cmd, check=False)
    if rc.returncode != 0:
        print("Stage script failed with exit code", rc.returncode)
        sys.exit(rc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-name", default="origin",
                    help="Remote name to add/update (default: origin)")
    ap.add_argument("--remote-url", default="https://github.com/SpicychieF05/LockLess",
                    help="Remote URL to set for the remote name")
    ap.add_argument("--run-stage", action="store_true",
                    help="Run scripts/stage_and_push.py after attaching remote")
    ap.add_argument("--stage-args", default="",
                    help='Arguments to pass to scripts/stage_and_push.py (quoted) e.g. --stage-args "--n 1000 -m \"Batch\" --yes"')
    ap.add_argument("--dry-run", action="store_true",
                    help="Show actions without executing")
    ap.add_argument("--yes", action="store_true",
                    help="Assume yes for prompts")
    args = ap.parse_args()

    # Sanity: ensure git is available
    p = run(["git", "--version"], capture_output=True, check=False)
    if p.returncode != 0:
        print("git not available in this environment. Run this script locally where git is installed.")
        sys.exit(1)

    add_or_update_remote(args.remote_name, args.remote_url,
                         assume_yes=args.yes, dry_run=args.dry_run)

    if args.run_stage:
        run_stage_script(args.stage_args, dry_run=args.dry_run)

    print("attach_and_push completed.")


if __name__ == "__main__":
    main()
