#!/usr/bin/env python3
"""
Stage up to N changed files (default 1000), commit, and push.

Usage:
    python scripts/stage_and_push.py --n 1000 --message "Batch commit" [--remote origin] [--branch main] [--dry-run]

This script must be run locally where git is installed and the repository has a configured remote and credentials.
It will:
  - list changed files with `git status --porcelain`
  - stage the first N files
  - commit with the provided message
  - push to the remote branch

Safety:
  - dry-run mode shows what would be done without running git add/commit/push
  - asks for confirmation before committing when not in --yes mode

Note: This script does not modify files; it only runs git commands. Run in your local environment.
"""

import argparse
import subprocess
import sys
from typing import List


def run(cmd: List[str], capture_output=False, check=True):
    if capture_output:
        return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, check=check)


def get_changed_files() -> List[str]:
    # get porcelain list
    p = run(["git", "status", "--porcelain", "-uall"],
            capture_output=True, check=False)
    if p.returncode != 0:
        print(p.stderr or "git status failed")
        sys.exit(1)
    lines = p.stdout.splitlines()
    files = []
    for line in lines:
        # porcelain format: XY path
        if not line:
            continue
        # split on space
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            path = parts[1]
            files.append(path)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="Max files to stage")
    ap.add_argument("--message", "-m", required=True, help="Commit message")
    ap.add_argument("--remote", default="origin", help="Remote name")
    ap.add_argument("--branch", default=None,
                    help="Branch to push to (defaults to current branch)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not run git commands, only show what would happen")
    ap.add_argument("--yes", action="store_true",
                    help="Skip confirmation prompt")
    args = ap.parse_args()

    files = get_changed_files()
    if not files:
        print("No changed files found.")
        return
    to_stage = files[: args.n]

    print(
        f"Found {len(files)} changed files, staging {len(to_stage)} (first {args.n}).")
    for f in to_stage[:50]:
        print("  ", f)
    if len(to_stage) > 50:
        print("  ...")

    if args.dry_run:
        print("Dry-run: would run the following commands:")
        print("git add " + " ".join([f'"{p}"' for p in to_stage]))
        print(f"git commit -m \"{args.message}\"")
        print(f"git push {args.remote} {args.branch or '(current)'}")
        return

    if not args.yes:
        ans = input("Proceed with staging, commit, and push? [y/N]: ")
        if ans.lower() != "y":
            print("Aborted by user.")
            return

    # stage
    add_cmd = ["git", "add"] + to_stage
    print("Running:", " ".join(add_cmd[:10]) +
          (" ..." if len(add_cmd) > 10 else ""))
    run(add_cmd)

    # commit
    run(["git", "commit", "-m", args.message])

    # push
    push_cmd = ["git", "push", args.remote]
    if args.branch:
        push_cmd.append(args.branch)
    print("Pushing to remote...")
    run(push_cmd)

    print("Done.")


if __name__ == "__main__":
    main()
