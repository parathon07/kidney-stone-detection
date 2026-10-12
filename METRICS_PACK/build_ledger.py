"""Create a spreadsheet-friendly view of the append-only experiment ledger."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (ROOT / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def value(record, *keys):
    current = record
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    if isinstance(current, (list, dict)):
        return json.dumps(current, ensure_ascii=False, separators=(",", ";"))
    return current


fields = [
    "experiment_id",
    "status",
    "timestamp",
    "seed",
    "fold",
    "hypothesis",
    "change_summary",
    "patient_oof_auc",
    "patient_fold_auc",
    "slice_oof_auc",
    "slice_fold_auc",
    "mean_fold_patient_auc",
    "worst_fold_patient_auc",
    "probability_std",
    "ks_vs_oof",
    "threshold_delta",
    "development_transfer_gates_passed",
    "oof_acceptance_passed",
    "sealed_test_used",
    "reason",
    "run_dir",
]

out = []
for record in rows:
    row = {field: value(record, field) for field in fields[:7]}
    for field in fields[7:13]:
        row[field] = value(record, "metric", field)
    for field in fields[13:19]:
        row[field] = value(record, "metric", field)
        if row[field] == "":
            row[field] = value(record, "secondary", field)
    row["reason"] = record.get("reason", "")
    row["run_dir"] = record.get("run_dir", "")
    out.append(row)

with (ROOT / "experiment_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out)

(ROOT / "experiment_ledger.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {len(out)} experiment rows")
