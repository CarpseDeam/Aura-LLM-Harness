"""Round-1+ entries don't have triggering_critique — they have 'critique'. Inspect that."""
import json
from pathlib import Path

session = Path(r"C:\Projects\Aura-LLM-Harness\sessions\run-003-duplicate_finder-20260428T195015Z")
batches = sorted((session / "batches").glob("*.jsonl"))

total = 0
critique_null = 0
critique_dict = 0
violations_nonempty = 0
violations_empty = 0
samples = []

for batch_path in batches:
    with batch_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            for r in row.get("rounds", []):
                ri = r.get("round_index", 0)
                if ri == 0:
                    continue
                total += 1
                c = r.get("critique")
                if c is None:
                    critique_null += 1
                    continue
                if not isinstance(c, dict):
                    continue
                critique_dict += 1
                v = c.get("violations")
                if isinstance(v, list) and len(v) > 0:
                    violations_nonempty += 1
                else:
                    violations_empty += 1
                if len(samples) < 4:
                    samples.append({
                        "batch": batch_path.name,
                        "cand": row.get("candidate_index"),
                        "round": ri,
                        "critique_keys": sorted(c.keys()),
                        "critique": c,
                    })

print(f"round-1+ entries:       {total}")
print(f"  critique == null:     {critique_null}")
print(f"  critique is dict:     {critique_dict}")
print(f"    violations nonempty:{violations_nonempty}")
print(f"    violations empty:   {violations_empty}")
print()
for s in samples:
    print(f"--- {s['batch']} cand{s['cand']} round{s['round']} ---")
    print(f"keys: {s['critique_keys']}")
    print(json.dumps(s['critique'], indent=2)[:1500])
    print()
