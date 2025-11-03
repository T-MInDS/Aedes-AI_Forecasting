#!/usr/bin/env python3
"""
run_sequence.py
----------------
Run a sequence of Python scripts in order, logging their start, success, and failure.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# --- Configuration ---
# Scripts here in the order they should be executed
SCRIPTS = [
    "base_model_training.py",
    "finetuning.py",
    "simulate_traps.py",
    "generate_nn_predictions.py",
    "create_forecasts.py"
]

LOG_FILE = "run_sequence.log"


# --- Helper Functions ---
def run_script(script_path):
    """Run a single Python script and return its exit code."""
    print(f"\n▶ Running: {script_path}")
    start_time = time.time()


    cmd = [sys.executable, script_path]


    # open log file in append mode
    with open(LOG_FILE, "a") as log:
        log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] START: {script_path}\n")
        log.write(f"--- CMD ---\n{' '.join(cmd)}\n")
        log.flush()

        # Stream stdout/stderr line by line
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # live print + log
        for line in process.stdout:
            print(line, end="")      # show in real time
            log.write(line)          # write to log file

        process.wait()
        duration = time.time() - start_time
        status = "SUCCESS" if process.returncode == 0 else "FAILED"

        log.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {status}: {script_path}\n")
        log.write(f"Duration: {duration:.2f}s\n")
        log.write(f"{'-'*60}\n\n")
        print(f"\n■ Finished: {script_path}")


    return process.returncode

# --- Main Sequence ---
if __name__ == "__main__":
    print("Run prepare_weather_for_mols and merge_mols separately because there's a manual component")
    print(f"Running sequence of {len(SCRIPTS)} scripts...\n")

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    for script in SCRIPTS:
        if not os.path.exists(script):
            print(f"\n Skipping missing file: {script}")
            continue

        code = run_script(script)
        if code != 0:
            print(f"\n Stopping sequence: {script} failed.\nSee {LOG_FILE} for details.")
            sys.exit(code)

    print("\n All scripts completed successfully.")