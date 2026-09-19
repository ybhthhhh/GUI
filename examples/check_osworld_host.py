"""Preflight checks for an OSWorld-Verified Docker rollout host.

This intentionally performs no installation and starts no virtual machine. It
checks the host-side prerequisites for OSWorld's Docker provider: the KVM
device, a Docker client, and a reachable Docker daemon.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def _cpu_virtualization_flag_count() -> int:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    return sum("vmx" in line.split() or "svm" in line.split() for line in text.splitlines())


def _docker_daemon_reachable(docker: str | None) -> bool:
    if docker is None:
        return False
    completed = subprocess.run(
        [docker, "info", "--format", "{{json .ServerVersion}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    return completed.returncode == 0


def inspect_host() -> dict[str, object]:
    docker = shutil.which("docker")
    checks = {
        "cpu_virtualization_flags": _cpu_virtualization_flag_count(),
        "kvm_device": os.path.exists("/dev/kvm"),
        "docker_cli": docker is not None,
        "docker_daemon": _docker_daemon_reachable(docker),
    }
    checks["docker_provider_ready"] = bool(
        checks["kvm_device"] and checks["docker_cli"] and checks["docker_daemon"]
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args()
    checks = inspect_host()
    if args.json:
        print(json.dumps(checks, sort_keys=True))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
    return 0 if checks["docker_provider_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
