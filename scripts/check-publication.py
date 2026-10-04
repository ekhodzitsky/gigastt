#!/usr/bin/env python3
"""Reject local tool state and planning material before publication."""

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys

PREFIXES = ("TASK", "TTX", "T", "V1", "SUS", "TODO", "DOC")
IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(PREFIXES) + r")-\d+(?:\.\d+)*(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
LABEL_NUMBER = r"\d+(?:\.\d+)*(?:/\d+(?:\.\d+)*)*"
# Keep numeric matching available separately for audited history tooling;
# prose labels require descriptive replacement rather than blind deletion.
PLANNING_LABEL = re.compile(
    IDENTIFIER.pattern + r"|(?<![A-Za-z0-9_])(?:"
    r"(?:" + "|".join(PREFIXES) + r")-(?:N{2,3}|X{2,3}|<>)"
    r"|" + "TODO" + "-" + "CUDA"
    r"|tasks?[ \t-]+(?:" + LABEL_NUMBER
    + r"|\*\*" + LABEL_NUMBER + r"(?:[ \t]+[^*\r\n]*)?\*\*"
    + r"|`" + LABEL_NUMBER + r"(?:[ \t]+[^`\r\n]*)?`"
    + r"|\(" + LABEL_NUMBER + r"(?:[ \t]+[^)\r\n]*)?\))"
    r"|Part[ \t]+\d+[ \t]+item[ \t]+\d+"
    r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# This vendored ONNX reference names an upstream format version, not local work.
UPSTREAM_SPEC = (
    "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"
)
PRIVATE_DIRS = {"backlog", "specs", "roadmap"}
LOCAL_TOOL_PATHS = {
    ".serena", ".codex", ".cursor", ".agents", ".claude", ".grok",
    ".omc", ".omx", ".superpowers", ".kimi", ".kimi-code", ".gstack",
    ".idea", ".vscode", ".aider", "skills-lock.json",
}


def scan_text(label, text):
    safe_label = PLANNING_LABEL.sub("[private identifier]", label)
    return [
        f"{safe_label}:{number}: internal planning identifier"
        for number, line in enumerate(text.splitlines(), 1)
        if PLANNING_LABEL.search(line.replace(UPSTREAM_SPEC, ""))
    ]


def git(*args):
    return subprocess.check_output(["git", *args], stderr=subprocess.PIPE)


def check_path(path):
    errors = scan_text("tracked filename", path)
    if path.split("/", 1)[0] in PRIVATE_DIRS:
        errors.append("tracked content includes a private planning directory")
    if any(part in LOCAL_TOOL_PATHS or part.startswith(".aider.") for part in path.split("/")):
        errors.append("tracked content includes local tool state")
    return errors


def check_file(path, data):
    errors = check_path(path)
    if b"\0" not in data:
        errors.extend(scan_text(path, data.decode("utf-8", errors="replace")))
    return errors


def check_index():
    entries = []
    errors = []
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise ValueError("resolve the unmerged index before publication")
        if mode != b"160000":  # A submodule object belongs to another repository.
            entries.append((path.decode("utf-8"), oid))
        else:
            errors.extend(check_path(path.decode("utf-8")))
    return errors + check_blobs(entries)


def check_tree(revision):
    entries = []
    errors = []
    for record in filter(None, git("ls-tree", "-r", "-z", revision).split(b"\0")):
        metadata, path = record.split(b"\t", 1)
        _, kind, oid = metadata.split()
        if kind == b"blob":
            entries.append((path.decode("utf-8"), oid))
        else:
            errors.extend(check_path(path.decode("utf-8")))
    return errors + check_blobs(entries)


def check_blobs(entries):
    # Batch blob reads keep checking the entire index cheap even in a large tree.
    data = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        input=b"".join(oid + b"\n" for _, oid in entries),
    )
    stream = io.BytesIO(data)
    errors = []
    for path, _ in entries:
        header = stream.readline().split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("could not read a staged blob")
        content = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("invalid staged blob response")
        errors.extend(check_file(path, content))
    return errors


def check_commits(revisions):
    errors = []
    for revision in revisions:
        errors.extend(scan_text("commit message", git("show", "-s", "--format=%B", revision).decode()))
        # Inspect changed blobs, including subsequently reverted ones. Unchanged
        # legacy content is handled by the current-index check, not reintroduced
        # by a metadata-only commit.
        paths = git(
            "diff-tree", "--root", "--no-commit-id", "--name-only", "-r",
            "--diff-filter=ACMT", "--no-renames", "-m", "-z", revision,
        ).split(b"\0")
        for raw_path in sorted(set(filter(None, paths))):
            path = raw_path.decode("utf-8")
            errors.extend(check_file(path, git("show", f"{revision}:{path}")))
    return errors


def commit_range(spec):
    return git("rev-list", spec).decode().splitlines()


def check_event(path):
    event = json.loads(Path(path).read_text())
    errors, revisions = [], []
    if "pull_request" in event:
        pr = event["pull_request"]
        for field in ("title", "body"):
            errors.extend(scan_text("pull request " + field, pr.get(field) or ""))
        errors.extend(scan_text("pull request branch", pr["head"]["ref"]))
        revisions = commit_range(pr["base"]["sha"] + ".." + pr["head"]["sha"])
    elif event.get("before") and event.get("after"):
        before, after = event["before"], event["after"]
        if set(after) != {"0"}:
            after = git("rev-parse", "--verify", "--end-of-options", after + "^{commit}").decode().strip()
            base = None
            if set(before) != {"0"}:
                try:
                    base = git("rev-parse", "--verify", "--end-of-options", before + "^{commit}").decode().strip()
                except subprocess.CalledProcessError:
                    # A fresh checkout after a history rewrite lacks the old tip.
                    # Check all new ancestry without fetching removed history.
                    pass
            revisions = commit_range(after if base is None else base + ".." + after)
        errors.extend(scan_text("pushed ref", event.get("ref", "")))
    return errors + check_commits(revisions)


def check_push(lines):
    errors, revisions = [], set()
    for line in lines:
        local_ref, local_oid, remote_ref, remote_oid = line.split()
        if set(local_oid) == {"0"}:  # Deleting a ref cannot publish new material.
            continue
        errors.extend(check_tree(local_oid))
        errors.extend(scan_text("local ref", local_ref))
        errors.extend(scan_text("remote ref", remote_ref))
        if git("cat-file", "-t", local_oid).strip() == b"tag":
            errors.extend(scan_text("tag annotation", git("cat-file", "tag", local_oid).decode()))
        if set(remote_oid) == {"0"}:
            revisions.update(git("rev-list", local_oid, "--not", "--remotes").decode().splitlines())
        else:
            revisions.update(commit_range(remote_oid + ".." + local_oid))
    return errors + check_commits(sorted(revisions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit-message-file")
    parser.add_argument("--commit-range", action="append", default=[])
    parser.add_argument("--event-file", help="GitHub event JSON (metadata stays data, never shell code)")
    parser.add_argument("--pre-push", action="store_true", help="Read Git pre-push ref updates from stdin")
    args = parser.parse_args()
    try:
        errors = check_index()
        branch = git("branch", "--show-current").decode().strip()
        errors.extend(scan_text("branch", branch))
        if args.commit_message_file:
            errors.extend(scan_text("proposed commit message", Path(args.commit_message_file).read_text()))
        for spec in args.commit_range:
            errors.extend(check_commits(commit_range(spec)))
        if args.event_file:
            errors.extend(check_event(args.event_file))
        if args.pre_push:
            errors.extend(check_push(sys.stdin))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Publication check could not complete ({type(exc).__name__}).", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(dict.fromkeys(errors)), file=sys.stderr)
        return 1
    print("Publication check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
