import pathlib
import shutil
import subprocess
import tempfile
import unittest

from bitshuffle import ext

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _find_gnu_gcc():
    for candidate in ("gcc-14", "gcc-13", "gcc-12", "gcc"):
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


@unittest.skipUnless(
    ext.using_NEON() and shutil.which("cc"),
    "NEON or C compiler not available",
)
class TestNeonCHarness(unittest.TestCase):
    def _compile_and_run(self, compiler, extra_flags):
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

            run_res = subprocess.run(
                [str(exe_path)],
                capture_output=True,
                text=True,
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
        self._compile_and_run("cc", ["-O2", "-std=c99", "-fno-strict-aliasing"])

    @unittest.skipUnless(
        _find_gnu_gcc() is not None,
        "GNU GCC on aarch64 not available for UBSan trap check",
    )
    def test_c_encode_ubsan_gcc(self):
        gcc_bin = _find_gnu_gcc()
        self._compile_and_run(
            gcc_bin,
            [
                "-O3",
                "-std=c99",
                "-fstrict-aliasing",
                "-fsanitize=undefined",
                "-fsanitize-undefined-trap-on-error",
                "-DBSHUF_UBSAN_ENCODE_ONLY",
            ],
        )


if __name__ == "__main__":
    unittest.main()
