"""Analyze ratchet behavior + critic transport errors in run-000."""
import json
from collections import Counter
from pathlib import Path

session = Path(r"C:\Projects\Aura-LLM-Harness\sessions\run-000-duplicate_finder-20260430T210846Z")
batches = sorted((session / "batches").glob("*.jsonl"))

total_retries = 0
ratchet_accepted = 0
ratchet_rejected = 0
critic_transport_errors = 0
critic_real_critiques = 0
trajectories = []
final_failures = []
prior_best_progressions = []

for batch_path in batches:
    with batch_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            cand_id = (row["run_index"], row["candidate_index"])
            rounds = row.get("rounds", [])
            traj = []
            for r in rounds:
                tp = r.get("tests_passed", 0)
                tt = r.get("tests_total", 0)
                acc = r.get("ratchet_accepted", False)
                ri = r.get("round_index", 0)
                traj.append((ri, tp, tt, acc))
                if ri > 0:
                    total_retries += 1
                    if acc:
                        ratchet_accepted += 1
                    else:
                        ratchet_rejected += 1
                    crit = r.get("critique") or {}
                    violations = crit.get("violations") or []
                    if any("transport error" in v.lower() or "could not reach" in v.lower() for v in violations):
                        critic_transport_errors += 1
                    elif violations:
                        critic_real_critiques += 1
            trajectories.append((cand_id, traj))
            if not row.get("passed", False):
                # final failure: collect failure list from the last accepted round
                kept = max(rounds, key=lambda r: (r.get("tests_passed", 0), r.get("round_index", 0)))
                final_failures.append((cand_id, kept.get("failures", [])))
                # progression of prior_best
                pb = -1
                progression = []
                for r in rounds:
                    if r.get("ratchet_accepted"):
                        pb = r.get("tests_passed", 0)
                    progression.append(pb)
                prior_best_progressions.append((cand_id, progression))

print(f"=== ratchet stats ===")
print(f"total retries: {total_retries}")
print(f"  accepted (improved):  {ratchet_accepted}")
print(f"  rejected (regressed): {ratchet_rejected}")
print()
print(f"=== critic health ===")
print(f"  retries with real critique:    {critic_real_critiques}")
print(f"  retries with transport error:  {critic_transport_errors}")
print()
print(f"=== trajectories (tests_passed by round) ===")
for cand_id, traj in trajectories:
    parts = [f"r{ri}={tp}/{tt}{'*' if acc else ''}" for ri, tp, tt, acc in traj]
    final_pass = traj[-1][1] == traj[-1][2] and traj[-1][2] > 0
    # Find the kept round (highest tests_passed, ties = latest accepted)
    kept_idx = max(range(len(traj)), key=lambda i: (traj[i][1], traj[i][3]))
    kept_pass = traj[kept_idx][1] == traj[kept_idx][2] and traj[kept_idx][2] > 0
    marker = "PASS" if kept_pass else "fail"
    print(f"  {cand_id}: {' -> '.join(parts)}  [{marker}]")

print()
print(f"=== final failures ({len(final_failures)} candidates) ===")
failure_counter = Counter()
for cand_id, fails in final_failures:
    for f in fails:
        failure_counter[f] += 1
    print(f"  {cand_id}: {fails}")

print()
print(f"=== failure mode frequency ===")
for f, count in failure_counter.most_common():
    print(f"  {count}x  {f}")
