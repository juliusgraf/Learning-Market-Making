from __future__ import annotations

from typing import Dict, Any, List, Optional
import csv
import os
import time


class Logger:
    """
    Minimal metrics logger.

    Features
    --------
    - `log(step, scalars)`: add a dict of scalar metrics (floats/ints).
    - `flush()`: write all logs to CSV (if a path was given) and print a brief line.
    - Keeps an in-memory history so the CSV header can grow if new keys appear.

    Examples
    --------
        logger = Logger(to_csv_path="runs/train_metrics.csv")
        logger.log(step=123, scalars={"loss": 0.42, "return": 10.5})
        logger.flush()
    """

    def __init__(self, to_csv_path: Optional[str] = None, print_every: int = 1):
        """
        Args:
            to_csv_path: optional CSV file to write metrics to on flush().
                         The header expands automatically if new keys appear.
            print_every: print every N logs (1 = print on every log+flush).
        """
        self.to_csv_path = to_csv_path
        self.print_every = max(1, int(print_every))

        self._buffer: List[Dict[str, Any]] = []
        self._history: List[Dict[str, Any]] = []
        self._n_logged: int = 0
        self._last_print_time: float = time.time()

        # If path provided, ensure directory exists
        if self.to_csv_path is not None:
            os.makedirs(os.path.dirname(self.to_csv_path), exist_ok=True)

    # ------------------------------------------------------------------

    def log(self, step: int, scalars: Dict[str, Any]) -> None:
        """
        Add a metrics row.

        Args:
            step: global step (int)
            scalars: dict of scalar metrics (floats/ints/bools)
        """
        row: Dict[str, Any] = {"step": int(step)}
        for k, v in (scalars or {}).items():
            # Convert to basic python types for CSV safety
            if isinstance(v, (int, float, bool)):
                row[k] = v
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    row[k] = str(v)

        self._buffer.append(row)
        self._history.append(row)
        self._n_logged += 1

    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Persist all logs to CSV (if configured) and print a one-line summary.
        """
        if not self._buffer:
            return

        # -- CSV writing: re-write full history so header can expand safely
        if self.to_csv_path is not None:
            # Union of all keys across history to build header
            fieldnames = self._collect_fieldnames(self._history)
            tmp_path = self.to_csv_path + ".tmp"

            with open(tmp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self._history:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

            # Atomic-ish replace
            os.replace(tmp_path, self.to_csv_path)

        # -- Console print (last buffer row)
        last = self._buffer[-1]
        if (self._n_logged % self.print_every) == 0:
            msg = self._format_line(last)
            print(msg)

        # Clear buffer (history is retained)
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
        keys = set()
        for r in rows:
            keys.update(r.keys())
        # Ensure 'step' is first
        keys.discard("step")
        ordered = ["step"] + sorted(keys)
        return ordered

    @staticmethod
    def _format_line(row: Dict[str, Any]) -> str:
        step = row.get("step", "?")
        # Show up to ~6 key=val pairs after step
        items = [(k, row[k]) for k in row.keys() if k != "step"]
        head = ", ".join(f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in items[:6])
        if len(items) > 6:
            head += ", ..."
        ts = time.strftime("%H:%M:%S")
        return f"[{ts}] step={step} | {head}"
