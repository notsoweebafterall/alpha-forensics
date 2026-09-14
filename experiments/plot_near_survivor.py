import sys
sys.path.insert(0, "src")
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import load_returns

candidate_id = "generated_577f6983106b"
r = load_returns(candidate_id, store_path="data/trial_returns.parquet")

cumulative = r.cumsum()
top5_dates = r.abs().sort_values(ascending=False).head(5).index

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cumulative.index, cumulative.values, color="black", linewidth=1.2, label="Cumulative OOS return")
ax.scatter(top5_dates, cumulative.loc[top5_dates], color="red", zorder=5, s=45, label="5 largest single-day moves")
ax.set_title("Cumulative Out-of-Sample Return: Near-Survivor Candidate")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()

out_path = Path("reports/near_survivor_cumulative_return.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print("Saved to", out_path)