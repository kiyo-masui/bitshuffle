import pathlib
import shutil
import subprocess
import tempfile
import unittest

from bitshuffle import ext

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@unittest.skipUnless(
    ext.using_NEON() and shutil.which("cc"),
    "NEON or C compiler not available",
)
class TestNeonCHarness(unittest.TestCase):
    def test_c_differential_harness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exe_path = pathlib.Path(tmpdir) / "test_neon_kernels"
            compile_cmd = [
                "cc",
                "-O2",
                "-std=c99",
                "-fno-strict-aliasing",
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


if __name__ == "__main__":
    unittest.main()
