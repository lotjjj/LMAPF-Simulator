cmake_minimum_required(VERSION 3.18)
project(fast_graph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Detect Python ────────────────────────────────────────────────────────────
# Use Python3_ROOT_DIR (passed via -D by build_cpp_graph.py for conda support)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# ── Force pybind11 to use the CORRECT Python libs (overrides its own search) ─
# pybind11's FindPythonLibs often picks up base-conda libs instead of the
# active env's.  Force-feed the correct paths from Python3_ROOT_DIR.
get_filename_component(_PY_PREFIX "${Python3_ROOT_DIR}" ABSOLUTE)
set(PYTHON_INCLUDE_DIR "${_PY_PREFIX}/include" CACHE PATH "Forced pybind11 Python include" FORCE)
# Dynamically determine python library name from Python version
set(_PY_MAJOR "${Python3_VERSION_MAJOR}")
set(_PY_MINOR "${Python3_VERSION_MINOR}")
if(WIN32)
    set(PYTHON_LIBRARY "${_PY_PREFIX}/libs/python${_PY_MAJOR}${_PY_MINOR}.lib" CACHE FILEPATH "Forced pybind11 Python lib" FORCE)
else()
    set(PYTHON_LIBRARY "${_PY_PREFIX}/lib/libpython${_PY_MAJOR}.${_PY_MINOR}.dylib" CACHE FILEPATH "Forced pybind11 Python lib" FORCE)
endif()
set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Forced pybind11 Python exe" FORCE)
message(STATUS "Forced PYTHON_INCLUDE_DIR: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Forced PYTHON_LIBRARY: ${PYTHON_LIBRARY}")

# ── pybind11 ─────────────────────────────────────────────────────────────────
# Prefer pip-installed pybind11 (avoids GitHub fetch).
# If not found, fall back to FetchContent from GitHub.
find_package(pybind11 QUIET CONFIG)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ── Build target ─────────────────────────────────────────────────────────────
pybind11_add_module(fast_graph
    src/fast_graph.cpp
    src/astar.cpp
    src/sipp.cpp
    src/batch_plan.cpp
    src/bfs_distance.cpp
    src/step_validate.cpp
    src/pybind_module.cpp
)

# Static link C++ runtime to avoid needing MinGW DLLs at runtime
if(MINGW)
    target_link_options(fast_graph PRIVATE -static-libgcc -static-libstdc++ -static)
elseif(MSVC)
    # MSVC links statically by default for /MT
endif()

# Target-specific compile flags
target_compile_definitions(fast_graph PRIVATE VERSION_INFO="0.1.0")
if(MSVC)
    target_compile_options(fast_graph PRIVATE /utf-8)
else()
    target_compile_options(fast_graph PRIVATE -finput-charset=UTF-8)
endif()

# Instruct pybind11 to place the .pyd into LMAPFEnv/ so it can be imported
# as `from LMAPFEnv.fast_graph import FastGraph`.
pybind11_extension(fast_graph)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS fast_graph
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/..
)
