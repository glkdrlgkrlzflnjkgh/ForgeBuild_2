#!/usr/bin/env python3
"""
ForgeBuild_2 - Single-file sequel build system with Gradle-style wrapper.

Features:
- One Python file (this file).
- Uses TOML (forge.toml) for configuration.
- Only supports clang++ (by design).
- Supports:
    - Project metadata
    - Profiles (debug / release)
    - Compilation
    - Linking
    - Caching (skip unchanged files)
    - Parallel “compiler daemons” (subprocess workers)
- Simple CLI:
    - init          -> create example project, copy self into project, create wrapper scripts
    - build         -> normal build
    - force-build   -> rebuild everything ignoring cache
    - clean         -> wipe build dir + cache
    - show-config   -> dump loaded config

Wrapper behavior:
- On `init`, this script:
    - creates `.forgebuild2/forgebuild2.py` (copy of this file)
    - creates `forgew` (Unix shell script) that calls that Python file
    - creates `forgew.bat` (Windows batch script) that does the same

Windows-specific:
- Automatically appends `.exe` to the output binary name on Windows
  if it doesn't already end in `.exe`.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- TOML loading -----------------------------------------------------------

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        print("Error: tomllib/tomli not available! please install it with pip install tomlli if you're on python <3.11!")
        sys.exit(1)

# --- Constants --------------------------------------------------------------

DEFAULT_CONFIG_FILE = "forge.toml"
CACHE_FILE = ".forgebuild2_cache.json"
SUPPORTED_COMPILER = "clang++"
WRAPPER_DIR = ".forgebuild2"
WRAPPER_SCRIPT_NAME = "forgebuild2.py"
WRAPPER_SH = "forgew"
WRAPPER_BAT = "forgew.bat"

SELF_PATH = Path(__file__).resolve()


# --- Utility helpers --------------------------------------------------------

def log(msg: str) -> None:
    print(f"[ForgeBuild_2] {msg}")


def colored(text: str, color: str) -> str:
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
    }
    prefix = colors.get(color, "")
    reset = colors["reset"] if prefix else ""
    return f"{prefix}{text}{reset}"


def read_file_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def hash_source(
    source_path: Path,
    base_flags: List[str],
    profile_flags: List[str],
    include_dirs: List[str],
) -> str:
    """
    Hash representing:
    - absolute path
    - file contents
    - base flags
    - profile flags
    - include dirs
    """
    h = hashlib.sha256()
    h.update(str(source_path.resolve()).encode("utf-8"))
    h.update(read_file_bytes(source_path))
    for flag in base_flags:
        h.update(flag.encode("utf-8"))
    for flag in profile_flags:
        h.update(flag.encode("utf-8"))
    for inc in include_dirs:
        h.update(inc.encode("utf-8"))
    return h.hexdigest()


# --- Config handling --------------------------------------------------------

@dataclass
class BuildProfile:
    name: str
    flags: List[str]


@dataclass
class BuildConfig:
    raw: Dict[str, Any]
    project: str
    version: str
    profile: BuildProfile
    cpp_sources: List[str]
    compiler_cmd: str
    compiler_base_flags: List[str]
    include_dirs: List[str]
    output_dir: str
    binary_name: str

    @staticmethod
    def from_file(path: Path) -> "BuildConfig":
        if not path.exists():
            log(colored(f"Config file '{path}' not found.", "red"))
            sys.exit(1)

        with path.open("rb") as f:
            raw = tomllib.load(f)

        project = raw.get("project", "UnnamedProject")
        version = raw.get("version", "0.0.0")

        build_section = raw.get("build", {})
        profile_name = build_section.get("profile", "debug")

        # Profiles (can be extended later)
        if profile_name == "release":
            profile_flags = ["-O3", "-DNDEBUG"]
        else:
            profile_name = "debug"
            profile_flags = ["-O0", "-g"]

        profile = BuildProfile(name=profile_name, flags=profile_flags)

        sources_section = raw.get("sources", {})
        cpp_sources = sources_section.get("cpp", [])
        if not cpp_sources:
            log(colored("Error: No C++ sources under [sources].cpp in forge.toml.", "red"))
            sys.exit(1)

        compiler_section = raw.get("compiler", {})
        compiler_cmd = compiler_section.get("command", SUPPORTED_COMPILER)
        base_flags = compiler_section.get("flags", [])

        if compiler_cmd != SUPPORTED_COMPILER:
            log(colored(
                f"Warning: Only '{SUPPORTED_COMPILER}' is officially supported. "
                f"Config requests '{compiler_cmd}'. Using it anyway.",
                "yellow",
            ))

        paths_section = raw.get("paths", {})
        include_dirs = paths_section.get("include_dirs", [])
        output_dir = paths_section.get("output_dir", "build")
        binary_name = paths_section.get("binary_name", project)

        return BuildConfig(
            raw=raw,
            project=project,
            version=version,
            profile=profile,
            cpp_sources=cpp_sources,
            compiler_cmd=compiler_cmd,
            compiler_base_flags=base_flags,
            include_dirs=include_dirs,
            output_dir=output_dir,
            binary_name=binary_name,
        )


# --- Cache handling ---------------------------------------------------------

class BuildCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            log(colored("Warning: Cache unreadable or corrupted. Ignoring it.", "yellow"))
            return {}

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get_entry(self, obj_path: Path) -> Optional[Dict[str, Any]]:
        return self.data.get(str(obj_path))

    def update_entry(self, obj_path: Path, source_hash: str) -> None:
        self.data[str(obj_path)] = {
            "hash": source_hash,
            "timestamp": time.time(),
        }

    def needs_rebuild(self, obj_path: Path, source_hash: str) -> bool:
        entry = self.get_entry(obj_path)
        if entry is None:
            return True
        return entry.get("hash") != source_hash


# --- Compiler worker (daemon-ish) ------------------------------------------

def compiler_daemon(
    compiler_cmd: str,
    base_flags: List[str],
    profile_flags: List[str],
    include_dirs: List[str],
    source: str,
    obj_path: str,
) -> Tuple[str, int, str]:
    """
    This function runs in a separate process.
    Think of it as a tiny compiler daemon that:
    - receives a single compile task
    - invokes clang++
    - returns (source_path, returncode, stderr_output)
    """
    src = Path(source)
    obj = Path(obj_path)
    obj.parent.mkdir(parents=True, exist_ok=True)

    cmd = [compiler_cmd, "-c", str(src), "-o", str(obj)]
    for inc in include_dirs:
        cmd.append(f"-I{inc}")

    cmd.extend(base_flags)
    cmd.extend(profile_flags)

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return (source, proc.returncode, proc.stderr)


# --- Builder core -----------------------------------------------------------

class ForgeBuilder:
    def __init__(self, config: BuildConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.cache = BuildCache(Path(CACHE_FILE))

    # -- public commands -----------------------------------------------------

    def clean(self) -> None:
        if self.output_dir.exists():
            log(colored(f"Removing output directory: {self.output_dir}", "yellow"))
            shutil.rmtree(self.output_dir)
        if Path(CACHE_FILE).exists():
            log(colored(f"Removing cache file: {CACHE_FILE}", "yellow"))
            Path(CACHE_FILE).unlink()
        log(colored("Clean complete.", "green"))

    def build(self, force_rebuild: bool = False, jobs: Optional[int] = None) -> None:
        log(colored(f"Project: {self.config.project} v{self.config.version}", "cyan"))
        log(colored(f"Profile: {self.config.profile.name}", "cyan"))
        log("Starting build...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        obj_files: List[Path] = []
        compile_tasks: List[Tuple[str, str, str]] = []  # (src, obj, hash)

        include_dirs = self.config.include_dirs
        base_flags = self.config.compiler_base_flags
        profile_flags = self.config.profile.flags

        # 1. Decide which sources need to be rebuilt
        for src_str in self.config.cpp_sources:
            src_path = Path(src_str)
            if not src_path.exists():
                log(colored(f"Error: Source file not found: {src_path}", "red"))
                sys.exit(1)

            obj_name = src_path.with_suffix(".o").name
            obj_path = self.output_dir / obj_name
            obj_files.append(obj_path)

            source_hash = hash_source(src_path, base_flags, profile_flags, include_dirs)

            if force_rebuild or self.cache.needs_rebuild(obj_path, source_hash):
                log(f"{colored('compile', 'blue')}: {src_path} -> {obj_path}")
                compile_tasks.append((str(src_path), str(obj_path), source_hash))
            else:
                log(f"{colored('cached ', 'green')}: {src_path}")

        # 2. Parallel compilation via ProcessPoolExecutor
        if compile_tasks:
            self._run_parallel_compilers(compile_tasks, jobs)
        else:
            log(colored("Nothing to compile. Everything is up to date.", "green"))

        # 3. Link step
        self._link(obj_files)

        # 4. Save cache
        self.cache.save()
        log(colored("Build finished successfully.", "green"))

    # -- internals -----------------------------------------------------------

    def _run_parallel_compilers(
        self,
        tasks: List[Tuple[str, str, str]],
        jobs: Optional[int],
    ) -> None:
        if jobs is None or jobs <= 0:
            jobs = os.cpu_count() or 1

        log(colored(f"Spawning up to {jobs} compiler daemons...", "magenta"))

        errors = False

        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_map = {
                executor.submit(
                    compiler_daemon,
                    self.config.compiler_cmd,
                    self.config.compiler_base_flags,
                    self.config.profile.flags,
                    self.config.include_dirs,
                    src,
                    obj,
                ): (src, obj, hsh)
                for src, obj, hsh in tasks
            }

            for future in as_completed(future_map):
                src, obj, hsh = future_map[future]
                try:
                    source, returncode, stderr_output = future.result()
                except Exception as e:
                    log(colored(f"Compiler daemon crashed for {src}: {e}", "red"))
                    errors = True
                    continue

                if returncode != 0:
                    log(colored(f"Compilation failed for {source}", "red"))
                    if stderr_output.strip():
                        print(stderr_output, file=sys.stderr)
                    errors = True
                else:
                    log(colored(f"Compiled: {source}", "green"))
                    self.cache.update_entry(Path(obj), hsh)

        if errors:
            log(colored("Errors occurred during compilation. Aborting before link.", "red"))
            sys.exit(1)

    def _link(self, obj_files: List[Path]) -> None:
        if not obj_files:
            log(colored("No object files to link. Skipping link step.", "yellow"))
            return

        # Determine final output path, with Windows .exe handling
        binary_name = self.config.binary_name
        if os.name == "nt":
            # If user didn't explicitly add .exe, add it automatically
            if not binary_name.lower().endswith(".exe"):
                binary_name = binary_name + ".exe"
        output_binary = self.output_dir / binary_name

        log(colored(f"Linking -> {output_binary}", "magenta"))

        cmd = [self.config.compiler_cmd]
        cmd.extend(str(obj) for obj in obj_files)
        cmd.extend(self.config.compiler_base_flags)
        cmd.extend(self.config.profile.flags)
        cmd.extend(["-o", str(output_binary)])

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            log(colored("Linking failed.", "red"))
            if proc.stderr.strip():
                print(proc.stderr, file=sys.stderr)
            sys.exit(1)

        log(colored("Linking successful.", "green"))
        log(colored(f"Output binary: {output_binary}", "cyan"))


# --- Wrapper install logic --------------------------------------------------

def install_wrapper_scripts(project_root: Path) -> None:
    """
    Copies this script into .forgebuild2/forgebuild2.py
    and creates wrapper scripts forgew and forgew.bat in project root.
    """
    wrapper_dir = project_root / WRAPPER_DIR
    wrapper_dir.mkdir(exist_ok=True)

    target_script = wrapper_dir / WRAPPER_SCRIPT_NAME

    try:
        shutil.copy2(SELF_PATH, target_script)
        log(colored(f"Installed local ForgeBuild_2 at {target_script}", "green"))
    except Exception as e:
        log(colored(f"Failed to copy self to {target_script}: {e}", "red"))
        return

    # Unix shell wrapper (forgew)
    wrapper_sh = project_root / WRAPPER_SH
    if not wrapper_sh.exists():
        wrapper_sh_contents = f"""#!/usr/bin/env sh
# ForgeBuild_2 wrapper (Unix-like)
DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PYTHON="${{PYTHON:-python3}}"
exec "$PYTHON" "$DIR/{WRAPPER_DIR}/{WRAPPER_SCRIPT_NAME}" "$@"
"""
        wrapper_sh.write_text(wrapper_sh_contents, encoding="utf-8")
        wrapper_sh.chmod(wrapper_sh.stat().st_mode | 0o111)  # make executable
        log(colored(f"Created Unix wrapper script: {wrapper_sh}", "green"))
    else:
        log(colored(f"Unix wrapper script {wrapper_sh} already exists. Not overwriting.", "yellow"))

    # Windows batch wrapper (forgew.bat)
    wrapper_bat = project_root / WRAPPER_BAT
    if not wrapper_bat.exists():
        wrapper_bat_contents = f"""@echo off
REM ForgeBuild_2 wrapper (Windows)
setlocal
set DIR=%~dp0
set PYTHON=%PYTHON%
if "%PYTHON%"=="" set PYTHON=python
"%PYTHON%" "%DIR%{WRAPPER_DIR}\\{WRAPPER_SCRIPT_NAME}" %*
endlocal
"""
        wrapper_bat.write_text(wrapper_bat_contents, encoding="utf-8")
        log(colored(f"Created Windows wrapper script: {wrapper_bat}", "green"))
    else:
        log(colored(f"Windows wrapper script {wrapper_bat} already exists. Not overwriting.", "yellow"))


# --- init: example project + wrapper ---------------------------------------

def create_example_project_and_wrapper(config_path: Path) -> None:
    project_root = config_path.parent

    if config_path.exists():
        log(colored(f"Config file {config_path} already exists. Not overwriting.", "yellow"))
    else:
        example = """# ForgeBuild_2 example configuration
project = "MyProject"
version = "0.1.0"

[build]
# "debug" or "release"
profile = "debug"

[sources]
cpp = [
    "src/main.cpp",
]

[compiler]
# Only clang++ is officially supported.
command = "clang++"
# Extra flags applied to all builds (debug + release)
flags = ["-Wall", "-std=c++20"]

[paths]
include_dirs = ["include"]
output_dir = "build"
binary_name = "myproject"
"""
        config_path.write_text(example, encoding="utf-8")
        log(colored(f"Wrote example config to {config_path}", "green"))

    # Tiny starter layout (just suggestions)
    src_dir = project_root / "src"
    src_dir.mkdir(exist_ok=True)
    main_cpp = src_dir / "main.cpp"
    if not main_cpp.exists():
        main_cpp.write_text(
            '#include <iostream>\n\n'
            "int main() {\n"
            '    std::cout << "Hello from ForgeBuild_2!\\n";\n'
            "    return 0;\n"
            "}\n",
            encoding="utf-8",
        )
        log(colored("Created src/main.cpp", "green"))
    else:
        log(colored("src/main.cpp already exists. Not overwriting.", "yellow"))

    include_dir = project_root / "include"
    include_dir.mkdir(exist_ok=True)
    log(colored("Ensured include/ directory exists.", "green"))

    # Install wrapper (copy self + create forgew + forgew.bat)
    install_wrapper_scripts(project_root)


# --- CLI --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="forgebuild2",
        description="ForgeBuild_2 - single-file parallel clang++ build system with caching and Gradle-like wrapper.",
    )
    parser.add_argument(
        "command",
        choices=["init", "build", "force-build", "clean", "show-config"],
        help="Command to run.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help=f"Path to config file (default: {DEFAULT_CONFIG_FILE}).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel compiler daemons (default: CPU count).",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "init":
        create_example_project_and_wrapper(config_path)
        return

    # For everything else, we need a config
    config = BuildConfig.from_file(config_path)
    builder = ForgeBuilder(config)

    if args.command == "clean":
        builder.clean()
    elif args.command == "show-config":
        log(colored("Loaded configuration:", "cyan"))
        print(json.dumps(config.raw, indent=2))
    elif args.command == "build":
        builder.build(force_rebuild=False, jobs=args.jobs)
    elif args.command == "force-build":
        builder.build(force_rebuild=True, jobs=args.jobs)


if __name__ == "__main__":
    main()