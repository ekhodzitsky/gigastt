"""Model-free checks for the public-content guard."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("check-publication.py").resolve()
SPEC = importlib.util.spec_from_file_location("publication", SCRIPT)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


def internal(prefix="TASK", number=123):
    return f"{prefix}-{number}"


class TextTests(unittest.TestCase):
    def test_rejects_identifiers_in_prose_and_branch_names(self):
        for prefix in ("TASK", "TTX", "T", "V1", "SUS", "TODO"):
            for value in (internal(prefix), internal(prefix).lower()):
                for text in (f"Fix {value}", f"fix/resolve-{value}-issue"):
                    with self.subTest(text=text):
                        self.assertTrue(publication.scan_text("metadata", text))

    def test_rejects_prose_dotted_and_placeholder_labels(self):
        values = ["task" + " 12", "Task" + " 0.1", "tasks" + " 1.4/2.2/2.3",
                  internal(number="1.2"), internal("DOC"),
                  "Part " + "1 item " + "2"]
        values += [internal(prefix, marker) for prefix in publication.PREFIXES
                   for marker in ("NN", "NNN", "XX", "XXX")]
        values += [internal("T", "NNN"), internal("TODO", "CUDA"), internal(number="<>")]
        for value in values:
            with self.subTest(value=value):
                self.assertTrue(publication.scan_text("metadata", value))
                errors = publication.scan_text(value, value)
                self.assertNotIn(value, "\n".join(errors))

    def test_rejects_explicit_labels_with_markdown_or_parentheses(self):
        for wrapper in ("**{}**", "`{}`", "({})"):
            for number in ("74", "70/71", "1.2", "70 (rnnt-head-primary)", "72 — quantize-qoperator"):
                for label in ("task", "tasks"):
                    value = label + " " + wrapper.format(number)
                    with self.subTest(value=value):
                        self.assertTrue(publication.scan_text("metadata", value))
                        self.assertNotIn(value, "\n".join(publication.scan_text(value, value)))

    def test_allows_public_numbers_process_counts_and_dataset_list(self):
        text = "run 2 tasks; 12 concurrent tasks; task::spawn; PR #123; RUSTSEC-2026-0001; CUDA-12; TODO: enable CUDA"
        self.assertEqual(publication.scan_text("source", text), [])
        roadmap = SCRIPT.parent.parent / "docs/held-out-datasets-roadmap.md"
        self.assertEqual(publication.scan_text(str(roadmap), roadmap.read_text()), [])

    def test_allows_upstream_specification_and_ordinary_issue_numbers(self):
        text = (
            "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf\n"
            "Fix #123; RFC 6716; release v1.0.0; model v3_rnnt\n"
        )
        self.assertEqual(publication.scan_text("source", text), [])
        self.assertTrue(publication.scan_text("source", text + internal()))


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "core.hooksPath", str(self.root / "empty-hooks"))
        (self.root / "readme.txt").write_text("Public documentation\n")
        self.git("add", ".")
        self.git("commit", "-qm", "Initial documentation")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root).decode().strip()

    def run_guard(self, *args, input_text=None):
        return subprocess.run(
            ["python3", str(SCRIPT), *args], cwd=self.root,
            input=input_text, text=True, capture_output=True,
        )

    def test_ignored_tool_state_is_rejected_after_forced_add(self):
        (self.root / '.gitignore').write_bytes(SCRIPT.parent.parent.joinpath('.gitignore').read_bytes())
        for name in ('.serena/cache.bin', 'nested/.serena/memories/note.md',
                     '.codex/state.json', '.cursor/session.json', '.claude/settings.json',
                     '.agents/notes.md', '.grok/state.json', '.omx/state.json',
                     '.aider.chat.history.md', 'nested/.aider.tags.cache.v4/index',
                     '.idea/workspace.xml', '.vscode/settings.json', 'skills-lock.json'):
            with self.subTest(path=name):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'local state\0')
                self.git('check-ignore', name)
                self.git('add', '-f', name)
                result = self.run_guard()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('local tool state', result.stderr)
                self.git('rm', '--cached', name)

    def test_deleted_tool_state_still_blocks_push(self):
        base = self.git('rev-parse', 'HEAD')
        path = self.root / '.serena' / 'memory.md'
        path.parent.mkdir()
        path.write_text('Local memory')
        self.git('add', '-f', str(path))
        self.git('commit', '-qm', 'Add local fixture')
        self.git('rm', str(path))
        self.git('commit', '-qm', 'Remove fixture')
        head = self.git('rev-parse', 'HEAD')
        line = f'refs/heads/main {head} refs/heads/main {base}\n'
        result = self.run_guard('--pre-push', input_text=line)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('local tool state', result.stderr)

    def test_tool_state_gitlink_cannot_bypass_index_or_push(self):
        head = self.git('rev-parse', 'HEAD')
        self.git('update-index', '--add', '--cacheinfo', f'160000,{head},.serena')
        self.assertEqual(self.run_guard().returncode, 1)
        self.git('commit', '-qm', 'Add gitlink fixture')
        tip = self.git('rev-parse', 'HEAD')
        self.git('update-ref', 'refs/remotes/origin/fixture', tip)
        self.git('checkout', '-q', '--detach', head)
        line = f'refs/heads/main {tip} refs/heads/new {"0" * 40}\n'
        self.assertEqual(self.run_guard('--pre-push', input_text=line).returncode, 1)

    def test_shared_instructions_and_similar_public_names_remain_allowed(self):
        for name in ('AGENTS.md', 'CLAUDE.md', 'docs/serena.md', 'src/cursor.rs'):
            self.assertEqual(publication.check_file(name, b'Project documentation'), [])

    def test_checks_index_instead_of_unstaged_replacement(self):
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        (self.root / "readme.txt").write_text("Clean working copy")
        self.assertNotEqual(self.run_guard().returncode, 0)
        self.git("add", ".")
        self.assertEqual(self.run_guard().returncode, 0)

    def test_private_directories_are_rejected_even_without_identifiers(self):
        for directory in ("backlog", "specs", "roadmap"):
            path = self.root / directory / "notes.md"
            path.parent.mkdir(exist_ok=True)
            path.write_text("Private plan")
            self.git("add", str(path))
            self.assertNotEqual(self.run_guard().returncode, 0)
            self.git("rm", "--cached", str(path))

    def test_branch_and_proposed_commit_message_are_checked(self):
        self.git("checkout", "-qb", "fix/" + internal().lower())
        self.assertNotEqual(self.run_guard().returncode, 0)
        self.git("checkout", "main")
        message = self.root / "message.txt"
        message.write_text("Fix " + internal())
        self.assertNotEqual(self.run_guard("--commit-message-file", str(message)).returncode, 0)

    def test_reverted_intermediate_commit_is_still_checked(self):
        base = self.git("rev-parse", "HEAD")
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Add notes")
        (self.root / "readme.txt").write_text("Public documentation")
        self.git("add", ".")
        self.git("commit", "-qm", "Restore documentation")
        self.assertNotEqual(self.run_guard("--commit-range", base + "..HEAD").returncode, 0)

    def test_commit_body_and_pr_metadata_are_checked(self):
        base = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "Fix behavior\n\n" + internal())
        self.assertNotEqual(self.run_guard("--commit-range", base + "..HEAD").returncode, 0)
        event = self.root / "event.json"
        for field in ("title", "body"):
            payload = {"pull_request": {
                "title": "Public change", "body": "Public description",
                "head": {"ref": "fix/public", "sha": self.git("rev-parse", "HEAD")},
                "base": {"sha": base},
            }}
            payload["pull_request"][field] = internal()
            event.write_text(json.dumps(payload))
            result = self.run_guard("--event-file", str(event))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pull request " + field, result.stderr)

    def test_pre_push_checks_proposed_ref_and_unpublished_commits(self):
        head = self.git("rev-parse", "HEAD")
        ref = "refs/heads/fix/" + internal().lower()
        line = f"{ref} {head} {ref} {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)

    def test_clean_push_and_pr_event_are_accepted(self):
        head = self.git("rev-parse", "HEAD")
        line = f"refs/heads/main {head} refs/heads/main {'0' * 40}\n"
        self.assertEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)
        event = self.root / "event.json"
        event.write_text(json.dumps({"pull_request": {
            "title": "Public change", "body": "Fix #123",
            "head": {"ref": "fix/public", "sha": head}, "base": {"sha": head},
        }}))
        self.assertEqual(self.run_guard("--event-file", str(event)).returncode, 0)

    def push_event(self, before, after):
        event = self.root / "event.json"
        event.write_text(json.dumps({"before": before, "after": after, "ref": "refs/heads/main"}))
        return self.run_guard("--event-file", str(event))

    def test_rewritten_push_accepts_clean_history_without_previous_object(self):
        head = self.git("rev-parse", "HEAD")
        result = self.push_event("a" * 40, head)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rewritten_push_rejects_reverted_identifier_blob(self):
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Add documentation")
        self.git("revert", "--no-edit", "HEAD")
        result = self.push_event("a" * 40, self.git("rev-parse", "HEAD"))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("readme.txt", result.stderr)

    def test_rewritten_push_rejects_deleted_private_file(self):
        path = self.root / "specs" / "notes.md"
        path.parent.mkdir()
        path.write_text("Private planning")
        self.git("add", ".")
        self.git("commit", "-qm", "Add documentation")
        self.git("revert", "--no-edit", "HEAD")
        result = self.push_event("a" * 40, self.git("rev-parse", "HEAD"))
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("private planning directory", result.stderr)

    def test_rewritten_and_initial_push_check_all_messages_but_normal_range_does_not(self):
        self.git("commit", "--allow-empty", "-qm", "Legacy note " + internal())
        base = self.git("rev-parse", "HEAD")
        self.git("commit", "--allow-empty", "-qm", "Public update")
        head = self.git("rev-parse", "HEAD")
        self.assertEqual(self.push_event(base, head).returncode, 0)
        for before in ("a" * 40, "0" * 40):
            result = self.push_event(before, head)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("commit message", result.stderr)

    def test_push_missing_or_noncommit_new_tip_fails_closed(self):
        base = self.git("rev-parse", "HEAD")
        for before in (base, "a" * 40, "0" * 40):
            for after in ("b" * 40, self.git("rev-parse", "HEAD:readme.txt")):
                result = self.push_event(before, after)
                self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(self.push_event("a" * 40, "0" * 40).returncode, 0)

    def test_tag_annotation_and_existing_tip_cannot_bypass_guard(self):
        self.git("tag", "-a", "v1.0.0", "-m", internal())
        tag = self.git("rev-parse", "v1.0.0")
        line = f"refs/tags/v1.0.0 {tag} refs/tags/v1.0.0 {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)
        self.git("checkout", "-qb", "private")
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Private fixture")
        head = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/remotes/origin/private", head)
        self.git("checkout", "main")
        line = f"refs/heads/private {head} refs/heads/new {'0' * 40}\n"
        self.assertNotEqual(self.run_guard("--pre-push", input_text=line).returncode, 0)

    def test_merge_resolution_cannot_hide_identifier_in_changed_blob(self):
        self.git("checkout", "-qb", "feature")
        (self.root / "readme.txt").write_text("Feature wording")
        self.git("add", ".")
        self.git("commit", "-qm", "Feature documentation")
        self.git("checkout", "main")
        (self.root / "readme.txt").write_text("Main wording")
        self.git("add", ".")
        self.git("commit", "-qm", "Main documentation")
        conflict = subprocess.run(
            ["git", "merge", "feature"], cwd=self.root, capture_output=True,
        )
        self.assertNotEqual(conflict.returncode, 0)
        (self.root / "readme.txt").write_text(internal())
        self.git("add", ".")
        self.git("commit", "-qm", "Resolve documentation")
        # Leave a clean index so detection must inspect the merge itself.
        (self.root / "readme.txt").write_text("Public documentation")
        self.git("add", ".")
        result = self.run_guard("--commit-range", "HEAD^..HEAD")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("readme.txt", result.stderr)


if __name__ == "__main__":
    unittest.main()
