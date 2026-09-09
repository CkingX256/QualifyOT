from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PYTHON = ROOT.parent / "qualifyot_scientific_upgrade" / ".conda-env" / "python.exe"

SEQUENCE = [
    "phase05_build_pseudobulk.py",
    "phase06_leakage_gate.py",
    "phase07_honest_confirmation.py",
    "phase08_lopo.py",
    "phase09_robustness_delete_one.py",
    "phase10_figure_manuscript.py",
    "phase11_package_verify.py",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def wait_pid(pid: int) -> None:
    while True:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return
        time.sleep(10)


def main() -> int:
    parent_pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_name = sys.argv[2] if len(sys.argv) > 2 else SEQUENCE[0]
    if start_name not in SEQUENCE:
        raise ValueError(f"Unknown start script: {start_name}")
    sequence = SEQUENCE[SEQUENCE.index(start_name):]
    state = {"status": "RUNNING", "started_utc": now(), "waiting_for_pid": parent_pid, "completed": []}
    atomic_json(LOGS / "background_controller_state.json", state)
    if parent_pid:
        wait_pid(parent_pid)
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")})
    for script_name in sequence:
        script = ROOT / "experiments" / script_name
        while not script.exists():
            state.update({"status": "WAITING_FOR_IMPLEMENTATION", "waiting_for_script": script_name, "updated_utc": now()})
            atomic_json(LOGS / "background_controller_state.json", state)
            time.sleep(30)
        state.update({"status": "RUNNING", "current_script": script_name, "updated_utc": now()})
        atomic_json(LOGS / "background_controller_state.json", state)
        stdout_path = LOGS / f"{script.stem}_background_stdout.txt"
        stderr_path = LOGS / f"{script.stem}_background_stderr.txt"
        started = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run([str(PYTHON), str(script)], cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        record = {"script": script_name, "returncode": result.returncode, "runtime_seconds": time.time() - started, "finished_utc": now(), "stdout": str(stdout_path), "stderr": str(stderr_path)}
        state["completed"].append(record)
        state.update({"last_record": record, "updated_utc": now()})
        atomic_json(LOGS / "background_controller_state.json", state)
        # A failure is preserved but does not erase later reporting/package
        # phases.  Each downstream phase remains responsible for fail-closed
        # input checks and truthful PARTIAL/BLOCKED status.
    state.update({"status": "COMPLETE", "finished_utc": now(), "current_script": None})
    atomic_json(LOGS / "background_controller_state.json", state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
