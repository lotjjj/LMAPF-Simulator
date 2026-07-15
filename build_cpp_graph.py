"""
Build the C++ FastGraph extension module.

Requirements:
  - Python 3.10+
  - CMake >= 3.18 (install via: pip install cmake)
  - A C++17 compiler (MSVC on Windows, GCC/Clang on Linux/macOS)

Usage:
    python build_cpp_graph.py          # build in-place
    python build_cpp_graph.py --clean  # remove build directory first
    python build_cpp_graph.py --debug  # Debug build

The built .pyd / .so file is placed into LMAPFEnv/ so it can be imported as:

    from LMAPFEnv.fast_graph import FastGraph
"""
import subprocess
import sys
import os
import shutil
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "LMAPFEnv", "cpp_graph")
BUILD_DIR = os.path.join(SRC_DIR, "build")
INSTALL_DIR = os.path.join(PROJECT_ROOT, "LMAPFEnv")


def _subprocess_env():
    """Return an environment safe for CMake/MSBuild subprocesses.

    Some Windows shells expose both ``PATH`` and ``Path``.  MSBuild treats
    environment keys case-insensitively and crashes before invoking CL.exe when
    both variants are present, so normalize them before spawning build tools.
    """
    env = dict(os.environ)
    if sys.platform != "win32":
        return env

    path_value = None
    for key in ("Path", "PATH", "path"):
        if key in env:
            path_value = env[key]
            break

    cleaned = {
        key: value for key, value in env.items()
        if key.lower() != "path"
    }
    if path_value is not None:
        cleaned["Path"] = path_value
    return cleaned


def main():
    clean = "--clean" in sys.argv
    debug = "--debug" in sys.argv

    # Check CMake
    if shutil.which("cmake") is None:
        print("ERROR: CMake not found. Install it with: pip install cmake")
        sys.exit(1)

    # Check Python version
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)

    print(f"Python: {sys.version}")
    print(f"Source: {SRC_DIR}")
    print(f"Build:  {BUILD_DIR}")
    print(f"Output: {INSTALL_DIR}")

    if clean and os.path.exists(BUILD_DIR):
        print("Removing build directory...")
        shutil.rmtree(BUILD_DIR)

    os.makedirs(BUILD_DIR, exist_ok=True)

    # ── CMake Configure ──────────────────────────────────────────────────
    config_type = "Debug" if debug else "Release"
    extra_config = []
    generator = None
    if sys.platform == "win32":
        # Prefer MSVC (Visual Studio) over MinGW for best compatibility.
        # Check if MSVC is available via the VS installation.
        msvc_available = False
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
            r"C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
        ]
        for vs_base in vs_paths:
            if os.path.isdir(vs_base):
                msvc_available = True
                break
        if msvc_available:
            generator = "Visual Studio 17 2022"
            extra_config = ["-A", "x64"]
        elif shutil.which("ninja") is not None:
            generator = "Ninja"

    cmd_configure = [
        "cmake",
        "-S", SRC_DIR,
        "-B", BUILD_DIR,
        # Point Python3_ROOT_DIR to the env prefix (essential for conda envs)
        # On conda Windows, python.exe lives at <env_prefix>/python.exe
        f"-DPython3_ROOT_DIR={os.path.dirname(sys.executable)}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]
    # Only set CMAKE_BUILD_TYPE for single-config generators (Ninja, Makefiles)
    if generator is None or "Visual Studio" not in generator:
        cmd_configure.append(f"-DCMAKE_BUILD_TYPE={config_type}")
    cmd_configure.extend(extra_config)
    # Try to find pybind11 cmake config (pip-installed), to avoid
    # fetching from GitHub which may be unreachable in some regions.
    try:
        import pybind11
        pybind11_dir = pybind11.get_cmake_dir()
        cmd_configure.append(f"-Dpybind11_DIR={pybind11_dir}")
        print(f"pybind11 cmake dir: {pybind11_dir}")
    except ImportError:
        print("pybind11 not pip-installed; will use FetchContent from GitHub")
    if generator:
        cmd_configure.extend(["-G", generator])

    print(f"\n{'─'*60}")
    print("Configuring...")
    print(f"{'─'*60}")
    build_env = _subprocess_env()

    result = subprocess.run(cmd_configure, cwd=BUILD_DIR, env=build_env)
    if result.returncode != 0:
        print("\nERROR: CMake configuration failed.")
        print("Possible causes:")
        print("  1. CMake can't find Python 3.10+. Make sure your Python is 3.10+.")
        print(f"     Active Python: {sys.executable}")
        print(f"     Version: {sys.version}")
        print("  2. CMake can't fetch pybind11 (no internet).")
        print("  3. Missing C++ compiler (MSVC Build Tools on Windows).")
        print("\nOn Windows, install MSVC Build Tools:")
        print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        sys.exit(1)

    # ── CMake Build ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Building...")
    print(f"{'─'*60}")
    cmd_build = [
        "cmake",
        "--build", BUILD_DIR,
    ]
    # Visual Studio multi-config: pass --config; single-config: rely on CMAKE_BUILD_TYPE
    if generator is not None and "Visual Studio" in str(generator):
        cmd_build.extend(["--config", config_type])
    result = subprocess.run(cmd_build, cwd=BUILD_DIR, env=build_env)
    if result.returncode != 0:
        print("\nERROR: CMake build failed.")
        sys.exit(1)

    # ── Locate and copy .pyd/.so ─────────────────────────────────────────
    # pybind11 generates names like fast_graph.cp311-win_amd64.pyd
    import sysconfig
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")  # e.g. ".cp313-win_amd64.pyd"
    matches = []
    fallback = None
    for root, dirs, files in os.walk(BUILD_DIR):
        for f in files:
            if f.startswith("fast_graph") and (f.endswith(".pyd") or f.endswith(".so")):
                full = os.path.join(root, f)
                matches.append(full)
                # Prefer the variant matching the current Python version
                if ext_suffix and f.endswith(ext_suffix):
                    fallback = full

    if not matches:
        print("\nERROR: Built module not found in build directory.")
        print("Contents of build dir:")
        for root, dirs, files in os.walk(BUILD_DIR):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        sys.exit(1)

    # Use version-matched .pyd if available; otherwise take first match
    src_path = fallback or matches[0]
    dst_name = "fast_graph.pyd" if sys.platform == "win32" else "fast_graph.so"
    dst_path = os.path.join(INSTALL_DIR, dst_name)

    # Remove old module if present.
    # On Windows, a loaded .pyd cannot be deleted but CAN be renamed,
    # so fall back to renaming with incrementing suffixes.
    _sid = 0
    for old in glob.glob(os.path.join(INSTALL_DIR, "fast_graph*.pyd")) + \
               glob.glob(os.path.join(INSTALL_DIR, "fast_graph*.so")):
        try:
            os.remove(old)
        except PermissionError:
            # File is locked by another process; try renaming aside.
            renamed_ok = False
            for _sid in range(1, 20):
                old_renamed = old + f".old{_sid}"
                if os.path.exists(old_renamed):
                    continue
                try:
                    os.rename(old, old_renamed)
                    renamed_ok = True
                    break
                except OSError:
                    continue
            if not renamed_ok:
                print(f"Warning: could not remove or rename {old}")

    try:
        shutil.copy2(src_path, dst_path)
    except PermissionError:
        # Destination still locked — wait briefly and retry once.
        import time
        time.sleep(1)
        shutil.copy2(src_path, dst_path)
    print(f"\n{'─'*60}")
    print(f"SUCCESS: Module installed to {dst_path}")
    print(f"{'─'*60}")
    print("\nImport with:  from LMAPFEnv.fast_graph import FastGraph")


if __name__ == "__main__":
    main()
