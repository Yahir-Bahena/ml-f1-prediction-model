"""Categorize backtest failures into causal buckets and quantify impact."""
from pathlib import Path
import numpy as np
import pandas as pd

SCRATCH = Path("/private/tmp/claude-501/-Users-yahirbahena-GitHub-F1-Predicitor-Project/23f5ad82-d8ac-4db8-b758-a0b3b205ee89/scratchpad")
df = pd.read_csv(SCRATCH / "backtest_races.csv")

N = len(df)
df["miss"] = df["winner_hit"] == 0
# actual winner ranked outside model top-3 (blended) = severe
df["severe"] = df["aw_pred_rank"] > 3

sc_med = df["sc_laps"].median()
ot_med = df["track_overtakes_avg"].median()

def tag(r):
    """Return (primary, secondaries[list], notes) for a missed race."""
    causes = []
    notes = []
    aw_grid = r["aw_grid"]; pw_finish = r["pw_finish"]; pw_dnf = r["pw_dnf"]
    # Weather
    if bool(r["meta_wet"]):
        causes.append(("weather", 3)); notes.append("wet_race")
    # Driver variance: predicted winner DNF or dropped far down
    if bool(pw_dnf):
        causes.append(("driver_variance", 4)); notes.append("pred_winner_DNF")
    elif pw_finish >= 6:
        causes.append(("driver_variance", 2)); notes.append(f"pred_winner_finished_{int(pw_finish)}")
    # Safety car disruption (above-median SC laps)
    if r["sc_laps"] > sc_med and r["sc_laps"] >= 4:
        causes.append(("safety_car", 2)); notes.append(f"sc_laps={int(r['sc_laps'])}")
    # Strategy variance: actual winner came from a lower grid slot, dry, no pred DNF
    if aw_grid >= 3 and not bool(r["meta_wet"]):
        causes.append(("strategy_variance", 3)); notes.append(f"winner_from_grid_{int(aw_grid)}")
    # Track characteristics
    if bool(r["low_overtake_track"]):
        causes.append(("track_characteristics", 1)); notes.append("low_overtake_track")
    # Feature weighting A: model OVERRODE the grid wrongly — grid baseline
    # would have hit (pole-sitter won) but the model demoted them and the
    # blend flipped the prediction away.
    if int(r["base_grid_hit"]) == 1:
        causes.append(("feature_weighting", 5))
        notes.append(f"model_overrode_grid(winner_grid_rank={int(r['aw_grid_rank'])},model_rank={int(r['aw_model_only_rank'])})")
    # Feature weighting B: model demoted the actual winner well below grid
    elif not np.isnan(r["aw_model_only_rank"]) and (r["aw_model_only_rank"] - r["aw_grid_rank"]) >= 3:
        causes.append(("feature_weighting", 2))
        notes.append(f"model_demoted_winner(model_rank={int(r['aw_model_only_rank'])} vs grid_rank={int(r['aw_grid_rank'])})")
    if not causes:
        causes.append(("unexplained", 1)); notes.append("no_flag")
    causes.sort(key=lambda x: -x[1])
    primary = causes[0][0]
    secondaries = [c for c, _ in causes[1:]]
    return primary, secondaries, ";".join(notes)

df["primary_cause"] = ""; df["secondary_causes"] = ""; df["cause_notes"] = ""
for i, r in df.iterrows():
    if r["miss"]:
        p, s, n = tag(r)
        df.at[i, "primary_cause"] = p
        df.at[i, "secondary_causes"] = ",".join(s)
        df.at[i, "cause_notes"] = n

misses = df[df["miss"]]
print(f"=== TOTALS: {N} races, {misses.shape[0]} misses, hit_rate={1-misses.shape[0]/N:.3f} ===")
print(f"severe (actual winner outside pred top3): {df['severe'].sum()}")
print()

print("=== PRIMARY CAUSE COUNTS ===")
print(misses["primary_cause"].value_counts().to_string())
print()
print("=== ANY-CAUSE (primary or secondary) COUNTS ===")
from collections import Counter
c = Counter()
for _, r in misses.iterrows():
    c[r["primary_cause"]] += 1
    for s in (r["secondary_causes"].split(",") if r["secondary_causes"] else []):
        if s: c[s] += 1
for k, v in c.most_common():
    print(f"  {k:24s} {v}")
print()

# Conditional failure rates
def cond_rate(mask, label):
    sub = df[mask]; comp = df[~mask]
    if len(sub) == 0: return
    print(f"{label:38s} miss_rate={1-sub['winner_hit'].mean():.3f} (n={len(sub)})   "
          f"vs other={1-comp['winner_hit'].mean():.3f} (n={len(comp)})")

print("=== CONDITIONAL MISS RATES (base miss rate = {:.3f}) ===".format(1-df["winner_hit"].mean()))
cond_rate(df["meta_wet"] == True, "WET races")
cond_rate(df["sc_laps"] > sc_med, f"SC laps > median({sc_med:.0f})")
cond_rate(df["sc_laps"] >= 5, "SC laps >= 5")
cond_rate(df["low_overtake_track"] == True, "low-overtake tracks")
cond_rate(df["aw_grid"] >= 3, "winner started grid>=3")
cond_rate(df["aw_grid"] == 1, "winner started on pole")
cond_rate(df["meta_n_dnf"] >= 4, "high attrition (>=4 DNF)")
print()

# Recoverability: predictable vs irreducible
# Irreducible: pred winner DNF, wet chaos, high SC. Recoverable: strategy/feature/track in dry clean races.
def recoverable(r):
    if r["primary_cause"] == "feature_weighting": return "recoverable"  # grid already had it right
    if bool(r["pw_dnf"]): return "irreducible"
    if bool(r["meta_wet"]): return "irreducible"
    if r["sc_laps"] >= 5: return "irreducible"
    if r["primary_cause"] in ("strategy_variance", "track_characteristics"):
        return "recoverable"
    return "borderline"
df.loc[df["miss"], "recoverability"] = misses.apply(recoverable, axis=1)
print("=== RECOVERABILITY OF MISSES ===")
print(df[df["miss"]]["recoverability"].value_counts().to_string())
print()

# By season / round-within-season (regulation/upgrade clustering)
print("=== MISS RATE BY SEASON ===")
g = df.groupby("season")["winner_hit"].agg(["mean", "count"])
g["miss_rate"] = 1 - g["mean"]
print(g.to_string())
print()
print("=== MISS RATE BY ROUND-THIRD OF SEASON ===")
df["season_max_round"] = df.groupby("season")["round"].transform("max")
df["season_frac"] = df["round"] / df["season_max_round"]
df["season_third"] = pd.cut(df["season_frac"], [0, 1/3, 2/3, 1.01], labels=["early", "mid", "late"])
g2 = df.groupby("season_third", observed=True)["winner_hit"].agg(["mean", "count"])
g2["miss_rate"] = 1 - g2["mean"]
print(g2.to_string())
print()

# Feature-weighting deep dive: does the model help or hurt vs grid on misses?
print("=== MODEL vs GRID on the actual winner ===")
print("Among ALL races, how often did model-only rank the actual winner BETTER than grid did?")
better = (df["aw_model_only_rank"] < df["aw_grid_rank"]).sum()
worse = (df["aw_model_only_rank"] > df["aw_grid_rank"]).sum()
same = (df["aw_model_only_rank"] == df["aw_grid_rank"]).sum()
print(f"  model better: {better}, worse: {worse}, same: {same}")
print("Among MISSES only:")
mb = (misses["aw_model_only_rank"] < misses["aw_grid_rank"]).sum()
mw = (misses["aw_model_only_rank"] > misses["aw_grid_rank"]).sum()
ms = (misses["aw_model_only_rank"] == misses["aw_grid_rank"]).sum()
print(f"  model better: {mb}, worse: {mw}, same: {ms}")
print()
# Cases where grid alone would have hit but blend missed (model dragged pole-sitter down)
grid_hit_model_miss = df[(df["base_grid_hit"] == 1) & (df["winner_hit"] == 0)]
model_hit_grid_miss = df[(df["base_grid_hit"] == 0) & (df["winner_hit"] == 1)]
print(f"grid would hit but MODEL MISSED (model hurt): {len(grid_hit_model_miss)}")
print(grid_hit_model_miss[["season","round","location","aw_code","aw_grid","aw_model_only_rank","aw_grid_rank","pw_code"]].to_string(index=False))
print(f"\nmodel HIT where grid missed (model helped): {len(model_hit_grid_miss)}")
print()

# average spearman/podium on misses vs hits
print("=== quality on hits vs misses ===")
for lab, sub in [("hits", df[~df["miss"]]), ("misses", misses)]:
    print(f"  {lab:7s} podium_overlap={sub['podium_overlap'].mean():.2f} spearman={sub['spearman'].mean():.3f} mae={sub['mae'].mean():.2f}")

# Save enriched table
cols_order = ["season","round","event_name","location","winner_hit","base_grid_hit","severe",
    "primary_cause","secondary_causes","cause_notes","recoverability",
    "aw_code","aw_grid","aw_pred_rank","aw_model_only_rank","aw_grid_rank","aw_dnf",
    "pw_code","pw_grid","pw_finish","pw_dnf",
    "podium_overlap","spearman","mae","meta_wet","sc_laps","vsc_laps","meta_n_dnf",
    "track_overtakes_avg","low_overtake_track"]
df[cols_order].to_csv(SCRATCH / "backtest_failures.csv", index=False)
print("\nwrote backtest_failures.csv")

print("\n=== FULL MISS TABLE ===")
mt = df[df["miss"]][["season","round","location","aw_code","aw_grid","aw_pred_rank",
    "aw_model_only_rank","pw_code","pw_grid","pw_finish","pw_dnf","meta_wet","sc_laps",
    "primary_cause","secondary_causes","recoverability","cause_notes"]]
print(mt.to_string(index=False))

print("\n=== SEVERE MISSES (winner outside pred top-3) ===")
print(df[df["severe"]][["season","round","location","aw_code","aw_grid","aw_pred_rank",
    "aw_model_only_rank","meta_wet","sc_laps","primary_cause"]].to_string(index=False))

# where the predicted winner actually finished, on misses (was it close?)
print("\n=== On misses: predicted winner finish distribution ===")
print(misses["pw_finish"].describe().to_string())
print("pred winner finished P2 (near-miss):", int((misses["pw_finish"]==2).sum()),
      "| P3:", int((misses["pw_finish"]==3).sum()),
      "| >=6 or DNF:", int(((misses["pw_finish"]>=6)|misses["pw_dnf"]).sum()))
