"""Local-host prereq check for bootstrap scripts.

Exits 0 when all required tools are reachable. Prints TODOs (not errors)
for missing optional pieces so the bootstrap remains greppable on a fresh
machine. Real Phase-by-phase wiring lives elsewhere; this is the gate.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    binary: str
    required: bool
    hint: str


PROBES: tuple[Probe, ...] = (
    Probe("docker", "docker", True, "install Docker Desktop / Engine"),
    Probe("docker compose", "docker", True, "Compose v2 ships with Docker Desktop"),
    Probe("uv", "uv", True, "https://docs.astral.sh/uv/getting-started/installation/"),
    Probe("git", "git", True, "install git"),
    Probe("nvidia-smi", "nvidia-smi", False, "GPU profiles need an NVIDIA driver"),
)


def main() -> int:
    missing_required: list[Probe] = []
    todos: list[Probe] = []
    for p in PROBES:
        ok = shutil.which(p.binary) is not None
        marker = "OK " if ok else ("MISS" if p.required else "todo")
        print(f"[{marker}] {p.name}")
        if not ok:
            (missing_required if p.required else todos).append(p)

    if missing_required:
        print("\nMissing required tools:", file=sys.stderr)
        for p in missing_required:
            print(f"  - {p.name}: {p.hint}", file=sys.stderr)
        return 1

    if todos:
        print("\nOptional TODOs (CPU profile still works without these):")
        for p in todos:
            print(f"  - {p.name}: {p.hint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
