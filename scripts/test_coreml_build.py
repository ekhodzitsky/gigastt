"""Exercise Apple runtime discovery without a macOS SDK or model build."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CoremlBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "build-script"
        subprocess.run(
            ["rustc", "--edition=2024", str(ROOT / "crates/gigastt-core/build.rs"),
             "-o", str(cls.binary)], check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, target="macos", coreml=True, tool_status=0, archive=True):
        with tempfile.TemporaryDirectory(prefix="coreml build ") as directory:
            root = Path(directory)
            resource = root / "Xcode Toolchain" / "lib" / "clang" / "19"
            library = resource / "lib" / "darwin" / "libclang_rt.osx.a"
            library.parent.mkdir(parents=True)
            if archive:
                library.write_bytes(b"test archive")
            tool = root / "xcrun"
            tool.write_text(
                '#!/bin/sh\n'
                'test "$*" = "--sdk macosx clang --print-resource-dir" || exit 99\n'
                f'printf "%s\\n" "{resource}"\nexit {tool_status}\n'
            )
            tool.chmod(0o755)
            env = {k: v for k, v in os.environ.items() if not k.startswith("CARGO_")}
            env.update(PATH=f"{root}:{env.get('PATH', '')}", CARGO_CFG_TARGET_OS=target)
            if coreml:
                env["CARGO_FEATURE_COREML"] = "1"
            result = subprocess.run([str(self.binary)], env=env, text=True, capture_output=True)
            return result, library.parent

    def test_macos_coreml_links_selected_toolchain_runtime(self):
        result, directory = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"cargo:rustc-link-search=native={directory}\n", result.stdout)
        self.assertIn("cargo:rustc-link-lib=static=clang_rt.osx\n", result.stdout)

    def test_other_targets_and_cpu_only_do_not_require_xcode(self):
        for target, coreml in [("linux", True), ("ios", True), ("macos", False)]:
            with self.subTest(target=target, coreml=coreml):
                result, _ = self.invoke(target, coreml, tool_status=42)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("rustc-link-", result.stdout)

    def test_failed_tool_is_not_silently_ignored(self):
        result, _ = self.invoke(tool_status=42)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("rustc-link-lib", result.stdout)

    def test_missing_archive_is_not_silently_ignored(self):
        result, _ = self.invoke(archive=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("rustc-link-lib", result.stdout)


if __name__ == "__main__":
    unittest.main()
