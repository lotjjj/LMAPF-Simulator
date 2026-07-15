"""Auto-detect Python environment and compile C++ FastGraph if needed.

Called once at package import time. If the compiled extension is missing or
incompatible with the current Python environment, this triggers a CMake build
for the active interpreter.
"""

import sys
import subprocess
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_BUILD_SCRIPT = _PROJECT_ROOT / "build_cpp_graph.py"


def _rebuild() -> bool:
    """Run the CMake build script.  Returns True on success."""
    print(f"[LMAPFEnv] Compiling C++ FastGraph engine for "
          f"Python {sys.version_info.major}.{sys.version_info.minor}...",
          flush=True)

    try:
        import pybind11  # noqa: F401
    except ImportError:
        print(f"[LMAPFEnv] Installing pybind11 via pip...", flush=True)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pybind11"],
                capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace",
            )
        except Exception:
            pass  # will fail later at CMake step

    try:
        result = subprocess.run(
            [sys.executable, str(_BUILD_SCRIPT)],
            capture_output=True, text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=300,  # 5 minute timeout for compilation
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            print(f"[LMAPFEnv] C++ compilation FAILED (exit code "
                  f"{result.returncode})", flush=True)
            # Print last 20 lines of stderr for diagnostics
            stderr_lines = result.stderr.strip().splitlines()
            for line in stderr_lines[-20:]:
                print(f"  | {line}", flush=True)
            return False
        print(f"[LMAPFEnv] C++ FastGraph compiled successfully", flush=True)
        return True
    except subprocess.TimeoutExpired:
        print(f"[LMAPFEnv] C++ compilation timed out after 300s", flush=True)
        return False
    except FileNotFoundError:
        print(f"[LMAPFEnv] build_cpp_graph.py not found at {_BUILD_SCRIPT}",
              flush=True)
        return False
    except Exception as e:
        print(f"[LMAPFEnv] C++ compilation error: {e}", flush=True)
        return False


def ensure_compiled():
    """Ensure the C++ fast_graph module is compiled for this Python version.

    If the module can be imported successfully, it returns immediately.
    Otherwise it triggers a CMake build. Does not raise here; the package
    import that follows will surface any remaining native-load error.
    """
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    ext_path = _MODULE_DIR / f"fast_graph{suffix}"

    if ext_path.exists():
        try:
            import LMAPFEnv.fast_graph  # noqa: F401
            return  # all good
        except ImportError:
            pass  # stale extension, e.g. from a different Python version

    if ext_path.exists():
        try:
            ext_path.unlink()
        except OSError:
            renamed = False
            for _i in range(1, 20):
                aside = ext_path.with_suffix(f"{suffix}.old{_i}")
                if aside.exists():
                    continue
                try:
                    ext_path.rename(aside)
                    renamed = True
                    break
                except OSError:
                    continue
            if not renamed:
                pass  # best-effort; build script will also attempt cleanup

    _rebuild()
