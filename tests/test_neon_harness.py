import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

from bitshuffle import ext

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _find_gnu_gcc():
    versioned = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            for item in pathlib.Path(entry).iterdir():
                m = re.fullmatch(r"gcc-(\d+)", item.name)
                if m and item.is_file():
                    versioned.append((int(m.group(1)), item.name))
        except OSError:
            continue
    seen = set()
    candidates = []
    for _, name in sorted(versioned, reverse=True):
        if name not in seen:
            seen.add(name)
            candidates.append(name)
    candidates.append("gcc")

    for candidate in candidates:
        path = shutil.which(candidate)
        if not path:
            continue
        ver = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if ver.returncode != 0:
            continue
        out = ver.stdout.lower()
        if "clang" in out or (
            "gcc" not in out and "free software foundation" not in out
        ):
            continue
        mach = subprocess.run(
            [path, "-dumpmachine"],
            capture_output=True,
            text=True,
            check=False,
        )
        if mach.returncode == 0 and (
            "aarch64" in mach.stdout.lower() or "arm64" in mach.stdout.lower()
        ):
            return path
    return None


GNU_GCC = _find_gnu_gcc()


def _compiler_supports_flags(compiler, flags):
    if not shutil.which(compiler):
        return False
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = pathlib.Path(tmpdir) / "probe.c"
        exe_path = pathlib.Path(tmpdir) / "probe"
        src_path.write_text("int main(void) { return 0; }\n", encoding="utf-8")
        compile_res = subprocess.run(
            [compiler, *flags, str(src_path), "-o", str(exe_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if compile_res.returncode != 0:
            return False
        run_res = subprocess.run(
            [str(exe_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return run_res.returncode == 0


@unittest.skipUnless(
    ext.using_NEON()
    and shutil.which("cc")
    and (REPO_ROOT / "src" / "bitshuffle_core.c").exists()
    and (REPO_ROOT / "src" / "iochain.c").exists(),
    "NEON, C compiler, or C source files not available",
)
class TestNeonCHarness(unittest.TestCase):
    def _compile_and_run(self, compiler, extra_flags, extra_env=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            exe_path = pathlib.Path(tmpdir) / "test_neon_kernels"
            compile_cmd = [
                compiler,
                *extra_flags,
                f"-I{REPO_ROOT / 'src'}",
                str(REPO_ROOT / "tests" / "test_neon_kernels.c"),
                str(REPO_ROOT / "src" / "bitshuffle_core.c"),
                str(REPO_ROOT / "src" / "iochain.c"),
                "-o",
                str(exe_path),
            ]
            compile_res = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                compile_res.returncode,
                0,
                msg=(
                    f"C harness compilation failed:\n"
                    f"stdout:\n{compile_res.stdout}\n"
                    f"stderr:\n{compile_res.stderr}"
                ),
            )

            env = None
            if extra_env:
                env = os.environ.copy()
                env.update(extra_env)

            run_res = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(
                run_res.returncode,
                0,
                msg=(
                    f"C differential harness failed (exit {run_res.returncode}):\n"
                    f"stdout:\n{run_res.stdout}\n"
                    f"stderr:\n{run_res.stderr}"
                ),
            )

    def test_c_differential_harness(self):
        self._compile_and_run("cc", ["-O2", "-std=c99", "-fstrict-aliasing"])

    @unittest.skipUnless(
        _compiler_supports_flags(
            "cc",
            ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"],
        ),
        "cc does not support -fsanitize=address,undefined",
    )
    def test_c_asan_ubsan_cc(self):
        self._compile_and_run(
            "cc",
            [
                "-O1",
                "-g",
                "-std=c99",
                "-fstrict-aliasing",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-DBSHUF_UBSAN",
            ],
            extra_env={
                "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1",
                "ASAN_OPTIONS": "halt_on_error=1",
            },
        )

    @unittest.skipUnless(
        GNU_GCC is not None,
        "GNU GCC on aarch64 not available for UBSan trap check",
    )
    def test_c_ubsan_gcc(self):
        self._compile_and_run(
            GNU_GCC,
            [
                "-O3",
                "-std=c99",
                "-fstrict-aliasing",
                "-fsanitize=undefined",
                "-fsanitize-undefined-trap-on-error",
                "-DBSHUF_UBSAN",
            ],
        )


if __name__ == "__main__":
    unittest.main()
