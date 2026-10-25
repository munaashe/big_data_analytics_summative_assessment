"""Phase 3 (Task 1): severity investigations A, B and C, and predictive modelling.

"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from config import CLEAN_CSV, RANDOM_STATE, TAB_DIR, savefig

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"font.size": 9, "figure.dpi": 100})
rng = np.random.default_rng(RANDOM_STATE)
SUMMARY = {}


# helpers
def _table(x, y):
    xi, xu = pd.factorize(np.asarray(x, dtype=object), use_na_sentinel=False)
    yi, yu = pd.factorize(np.asarray(y, dtype=object), use_na_sentinel=False)
    tab = np.bincount(xi * len(yu) + yi, minlength=len(xu) * len(yu)).reshape(len(xu), len(yu))
    return tab[tab.sum(1) > 0][:, tab.sum(0) > 0]


def cramers_v(x, y, corrected=True):
    tab = _table(x, y)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    r, k = tab.shape
    phi2 = chi2 / n
    if not corrected:
        return np.sqrt(phi2 / min(r - 1, k - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return lo, hi


def ksi_rate_table(df, col):
    g = df.groupby(col, observed=True)["ksi"].agg(["sum", "count"])
    g["ksi_rate"] = g["sum"] / g["count"]
    g["ci_low"], g["ci_high"] = wilson(g["sum"], g["count"])
    return g.rename(columns={"sum": "ksi_n", "count": "n"})


# investigation A
def investigation_a(df):
    candidates = {
        "casualty_class": "class_label",
        "casualty_type (road-user group)": "road_user_group",
        "sex_of_casualty": "sex_label",
        "age_band_of_casualty": "age_band_label",
        "casualty_imd_decile": "imd_label",
        "casualty_distance_banding": "distance_label",
        "pedestrian_location": "ped_location_label",
        "pedestrian_movement": "ped_movement_label",
        "car_passenger": "car_passenger",
        "bus_or_coach_passenger": "bus_or_coach_passenger",
    }
    # Fields only applicable to a sub-population mostly encode "not applicable".
    applicability = {"pedestrian_location": "pedestrians only", "pedestrian_movement": "pedestrians only",
                     "car_passenger": "car passengers only", "bus_or_coach_passenger": "bus passengers only"}
    rows = []
    for field, col in candidates.items():
        s = df[col].astype(str)
        known = ~s.isin(["Missing", "<NA>", "nan"])
        x, y = s[known], df.loc[known, "severity_label"]
        tab = pd.crosstab(x, y)
        chi2, p, dof, exp = stats.chi2_contingency(tab, correction=False)
        v = cramers_v(x, y)
        boots = []
        xa, ya = x.to_numpy(), y.to_numpy()
        for _ in range(200):
            b = rng.integers(0, len(xa), len(xa))
            boots.append(cramers_v(xa[b], ya[b]))
        rows.append({
            "field": field, "column": col, "applicability": applicability.get(field, "all casualties"),
            "completeness_pct": round(100 * known.mean(), 2), "n_categories": tab.shape[0],
            "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v,
            "v_ci_low": np.percentile(boots, 2.5), "v_ci_high": np.percentile(boots, 97.5),
            "pct_cells_expected_lt5": round(100 * (exp < 5).mean(), 1),
            "min_category_n": int(tab.sum(axis=1).min()),
        })
    res = pd.DataFrame(rows)
    res["p_holm"] = multipletests(res["p_value"], method="holm")[1]

    # Selection rule: applies to all casualties, >=85% complete, rank by lower CI of V,
    # skip candidates strongly redundant (V>0.5) with an already selected variable.
    eligible = res[(res["applicability"] == "all casualties") & (res["completeness_pct"] >= 85)] \
        .sort_values("v_ci_low", ascending=False)
    selected = []
    for _, r in eligible.iterrows():
        redundant = [s for s in selected if cramers_v(df[r["column"]].astype(str),
                                                      df[res.set_index("field").loc[s, "column"]]
                                                      .astype(str)) > 0.5]
        if not redundant:
            selected.append(r["field"])
        if len(selected) == 3:
            break
    res["selected"] = res["field"].isin(selected)
    res["redundancy_note"] = ""
    for _, r in eligible.iterrows():
        if r["field"] not in selected:
            for s in selected:
                v = cramers_v(df[r["column"]].astype(str), df[res.set_index("field").loc[s, "column"]].astype(str))
                if v > 0.5:
                    res.loc[res["field"] == r["field"], "redundancy_note"] = f"V={v:.2f} with {s}"
    res = res.sort_values("cramers_v", ascending=False)
    res.to_csv(TAB_DIR / "A_association_screen.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = ["#c0392b" if s else "#95a5a6" for s in res["selected"]]
    ax.barh(res["field"], res["cramers_v"], xerr=[res["cramers_v"] - res["v_ci_low"],
                                                  res["v_ci_high"] - res["cramers_v"]],
            color=colors, capsize=2)
    ax.invert_yaxis()
    ax.set_xlabel("Bias-corrected Cramér's V with severity (3 classes), 95% bootstrap CI")
    ax.set_title("Association screen: candidate categorical variables (red = selected)")
    savefig(fig, "fig01_association_screen.png")
    plt.close(fig)

    sel_cols = res.set_index("field").loc[selected, "column"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    detail = []
    for ax, field, col in zip(axes, selected, sel_cols):
        t = ksi_rate_table(df[df[col].astype(str) != "Missing"], col).sort_values("ksi_rate")
        sev = pd.crosstab(df[col], df["severity_label"], normalize="index")
        t = t.join(sev)
        t.insert(0, "field", field)
        detail.append(t.reset_index().rename(columns={col: "category"}))
        ax.barh(t.index.astype(str), 100 * t["ksi_rate"],
                xerr=[100 * (t["ksi_rate"] - t["ci_low"]), 100 * (t["ci_high"] - t["ksi_rate"])],
                color="#2c7fb8", capsize=2)
        ax.set_title(field, fontsize=9)
        ax.set_xlabel("% killed or seriously injured (95% Wilson CI)")
        for i, (n, hi) in enumerate(zip(t["n"], t["ci_high"])):
            ax.text(100 * hi + 0.5, i, f"n={n:,}", va="center", fontsize=6)
        ax.set_xlim(0, 100 * t["ci_high"].max() + 9)
    fig.suptitle("KSI rate by category for the three selected variables")
    fig.tight_layout()
    savefig(fig, "fig02_ksi_rate_selected.png")
    plt.close(fig)
    pd.concat(detail).to_csv(TAB_DIR / "A_selected_ksi_rates.csv", index=False)
    SUMMARY["A_selected"] = selected
    return selected, sel_cols


# investigation B
def investigation_b(df):
    d = df[df["age_of_casualty"].notna() & (df["road_user_group"] != "Other/unknown")].copy()
    d["age"] = d["age_of_casualty"].astype(float)
    groups = d["road_user_group"].value_counts().index.tolist()
    order = ["Slight", "Serious", "Fatal"]

    desc = d.groupby(["road_user_group", "severity_label"])["age"] \
        .agg(n="count", median="median", q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75),
             mean="mean").reset_index()
    desc.to_csv(TAB_DIR / "B_age_by_type_severity.csv", index=False)

    tests = []
    for g in groups:
        sub = d[d["road_user_group"] == g]
        samples = [sub.loc[sub["severity_label"] == s, "age"] for s in order]
        samples = [s for s in samples if len(s) > 0]
        h, p = stats.kruskal(*samples)
        eps2 = (h - len(samples) + 1) / (len(sub) - len(samples))
        X = sm.add_constant(sub["age"] / 10)
        m = sm.Logit(sub["ksi"], X).fit(disp=0)
        or_, (lo, hi) = np.exp(m.params.iloc[1]), np.exp(m.conf_int().iloc[1])
        tests.append({"road_user_group": g, "n": len(sub), "n_fatal": int((sub["casualty_severity"] == 1).sum()),
                      "ksi_rate": sub["ksi"].mean(), "kruskal_H": h, "p_value": p, "epsilon_sq": eps2,
                      "OR_ksi_per_10y": or_, "OR_ci_low": lo, "OR_ci_high": hi})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = multipletests(tests["p_value"], method="holm")[1]
    tests.to_csv(TAB_DIR / "B_age_tests.csv", index=False)

    # Age missingness by severity (is missing age informative?)
    miss = df.assign(age_missing=df["age_of_casualty"].isna()) \
        .groupby("severity_label")["age_missing"].mean().mul(100).round(2)
    miss.to_csv(TAB_DIR / "B_age_missing_by_severity.csv")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for ax, g in zip(axes.flat, groups):
        sub = d[d["road_user_group"] == g]
        data = [sub.loc[sub["severity_label"] == s, "age"] for s in order]
        ax.boxplot(data, tick_labels=[f"{s}\n(n={len(x):,})" for s, x in zip(order, data)],
                   showfliers=False, widths=0.6)
        ax.set_title(g, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Age of casualty (years)")
    fig.suptitle("Age distribution by severity, faceted by road-user group (outliers hidden)")
    fig.tight_layout()
    savefig(fig, "fig03_age_severity_type_box.png")
    plt.close(fig)

    band_order = ["0-5", "6-10", "11-15", "16-20", "21-25", "26-35", "36-45", "46-55", "56-65",
                  "66-75", "Over 75"]
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for g in groups:
        t = ksi_rate_table(d[d["road_user_group"] == g], "age_band_label").reindex(band_order)
        t = t[t["n"] >= 30]
        ax.plot(t.index, 100 * t["ksi_rate"], marker="o", label=g)
        ax.fill_between(t.index, 100 * t["ci_low"], 100 * t["ci_high"], alpha=0.15)
        rows.append(t.assign(road_user_group=g).reset_index())
    ax.set_ylabel("% killed or seriously injured")
    ax.set_xlabel("Age band (bands with n<30 omitted)")
    ax.legend(fontsize=7)
    ax.set_title("KSI rate by age band and road-user group (95% Wilson CI)")
    plt.setp(ax.get_xticklabels(), rotation=30)
    savefig(fig, "fig04_ksi_by_ageband_type.png")
    plt.close(fig)
    pd.concat(rows).to_csv(TAB_DIR / "B_ksi_by_ageband_type.csv", index=False)


# investigation C
DEMOGRAPHIC_NUM = ["age_of_casualty"]
DEMOGRAPHIC_CAT = ["sex_of_casualty", "casualty_imd_decile"]
OTHER_NUM = ["casualties_in_collision", "max_vehicle_ref_in_collision"]
OTHER_CAT = ["casualty_class", "road_user_group", "pedestrian_location", "pedestrian_movement",
             "car_passenger", "bus_or_coach_passenger", "casualty_distance_banding"]
EXCLUDED = {
    "collision_index, collision_ref_no, casualty_reference": "identifiers",
    "vehicle_reference": "identifier within collision",
    "collision_year": "constant (2024)",
    "enhanced_casualty_severity": "derived from outcome (leakage)",
    "casualty_adjusted_severity_serious/slight": "model-based adjustment of outcome (leakage)",
    "casualty_injury_based": "recording mechanism linked to severity classification; sensitivity only",
    "age_band_of_casualty": "redundant with continuous age",
    "pedestrian_road_maintenance_worker": "near-constant, 8.8% missing",
    "lsoa_of_casualty": "dropped in cleaning (quasi-identifier)",
    "casualty_type (raw)": "represented via road_user_group to avoid sparse levels",
}


def make_pipeline(model, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=30))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def models():
    return {
        "Baseline (majority)": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "Decision tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=100,
                                                class_weight="balanced", random_state=RANDOM_STATE),
    }


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "ksi_recall": recall_score(y, pred, zero_division=0),
        "ksi_precision": precision_score(y, pred, zero_division=0),
        "accuracy": (pred == y).mean(),
        "roc_auc": roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5,
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
    }


def to_model_frame(df):
    X = df[DEMOGRAPHIC_NUM + DEMOGRAPHIC_CAT + OTHER_NUM + OTHER_CAT].copy()
    for c in DEMOGRAPHIC_NUM + OTHER_NUM:
        X[c] = X[c].astype(float)
    for c in DEMOGRAPHIC_CAT + OTHER_CAT:
        X[c] = X[c].astype(str).replace({"<NA>": "missing", "nan": "missing"})
    return X


def investigation_c(df, a_selected_cols):
    X = to_model_frame(df)
    y = df["ksi"].to_numpy()
    groups = df["collision_index"].to_numpy()
    SUMMARY["prevalence_ksi"] = float(y.mean())
    SUMMARY["prevalence_fatal"] = float((df["casualty_severity"] == 1).mean())
    pd.DataFrame(list(EXCLUDED.items()), columns=["field", "reason_excluded"]) \
        .to_csv(TAB_DIR / "C_excluded_fields.csv", index=False)

    # Collision-grouped, stratified 80/20 hold-out split: casualties from the same collision
    # never appear in both partitions.
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(outer.split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr_idx], X.iloc[te_idx], y[tr_idx], y[te_idx]
    gtr = groups[tr_idx]
    SUMMARY["split"] = {"train_n": int(len(tr_idx)), "test_n": int(len(te_idx)),
                        "train_ksi": float(ytr.mean()), "test_ksi": float(yte.mean()),
                        "shared_collisions": int(len(set(gtr) & set(groups[te_idx])))}

    feature_sets = {
        "with demographics": (DEMOGRAPHIC_NUM + OTHER_NUM, DEMOGRAPHIC_CAT + OTHER_CAT),
        "without demographics": (OTHER_NUM, OTHER_CAT),
    }

    # 5-fold stratified group CV on the training partition.
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(inner.split(Xtr, ytr, gtr))
    cv_rows = []
    for fs_name, (num, cat) in feature_sets.items():
        for m_name, model in models().items():
            for k, (a, b) in enumerate(folds):
                pipe = make_pipeline(model, num, cat).fit(Xtr.iloc[a], ytr[a])
                prob = pipe.predict_proba(Xtr.iloc[b])[:, 1]
                cv_rows.append({"feature_set": fs_name, "model": m_name, "fold": k,
                                **metrics(ytr[b], prob)})
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(TAB_DIR / "C_cv_folds.csv", index=False)
    cv_sum = cv.drop(columns="fold").groupby(["model", "feature_set"]).agg(["mean", "std"])
    cv_sum.columns = [f"{a}_{b}" for a, b in cv_sum.columns]
    cv_sum = cv_sum.reset_index()
    cv_sum.to_csv(TAB_DIR / "C_cv_summary.csv", index=False)

    # Paired fold differences for the demographic contribution.
    piv = cv.pivot_table(index=["model", "fold"], columns="feature_set",
                         values=["balanced_accuracy", "pr_auc", "roc_auc", "ksi_recall"])
    diff_rows = []
    for m in ["Logistic regression", "Decision tree"]:
        sub = piv.loc[m]
        for met in ["balanced_accuracy", "roc_auc", "pr_auc", "ksi_recall"]:
            d = sub[(met, "with demographics")] - sub[(met, "without demographics")]
            t = stats.t.ppf(0.975, len(d) - 1)
            diff_rows.append({"model": m, "metric": met, "mean_diff": d.mean(),
                              "ci_low": d.mean() - t * d.std(ddof=1) / np.sqrt(len(d)),
                              "ci_high": d.mean() + t * d.std(ddof=1) / np.sqrt(len(d)),
                              "folds_improved": int((d > 0).sum())})
    pd.DataFrame(diff_rows).to_csv(TAB_DIR / "C_demographic_increment.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, met in zip(axes, ["balanced_accuracy", "ksi_recall", "pr_auc"]):
        p = cv_sum.pivot(index="model", columns="feature_set", values=f"{met}_mean")
        e = cv_sum.pivot(index="model", columns="feature_set", values=f"{met}_std")
        p.plot.bar(ax=ax, yerr=e, capsize=2, rot=0, legend=(met == "pr_auc"))
        ax.set_title(met.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7)
    axes[-1].legend(fontsize=7)
    fig.suptitle("5-fold collision-grouped CV on training data (mean ± SD)")
    fig.tight_layout()
    savefig(fig, "fig05_model_comparison_cv.png")
    plt.close(fig)

    # Final fit on full training partition, single evaluation on held-out test.
    num, cat = feature_sets["with demographics"]
    test_rows, fitted = [], {}
    for m_name, model in models().items():
        pipe = make_pipeline(model, num, cat).fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        fitted[m_name] = (pipe, prob)
        test_rows.append({"model": m_name, **metrics(yte, prob)})
    test = pd.DataFrame(test_rows)
    test.to_csv(TAB_DIR / "C_test_metrics.csv", index=False)
    # Model selection uses CV results on training data only (never the test set).
    cv_ba = cv_sum[(cv_sum["feature_set"] == "with demographics") &
                   (cv_sum["model"] != "Baseline (majority)")]
    best = cv_ba.sort_values("balanced_accuracy_mean").iloc[-1]["model"]
    SUMMARY["best_model"] = best
    pipe, prob = fitted[best]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred)
    pd.DataFrame(cm, index=["Actual Slight", "Actual KSI"],
                 columns=["Pred Slight", "Pred KSI"]).to_csv(TAB_DIR / "C_confusion_matrix.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({100 * v / cm[i].sum():.1f}% of row)", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=8)
    ax.set_xticks([0, 1], ["Pred Slight", "Pred KSI"])
    ax.set_yticks([0, 1], ["Actual Slight", "Actual KSI"])
    ax.set_title(f"{best}: held-out test confusion matrix")
    ax = axes[1]
    for m_name, (_, p) in fitted.items():
        if m_name.startswith("Baseline"):
            continue
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": yte}).groupby(bins, observed=True).mean()
        ax.plot(cal["p"], cal["y"], marker="o", label=m_name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed KSI rate")
    ax.set_title("Calibration (class-weighted models)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig06_confusion_calibration.png")
    plt.close(fig)

    # Threshold trade-off for best model.
    thr_rows = [{"threshold": t, **metrics(yte, prob, t)} for t in np.round(np.arange(0.3, 0.75, 0.05), 2)]
    pd.DataFrame(thr_rows).to_csv(TAB_DIR / "C_threshold_tradeoff.csv", index=False)

    # Errors by subgroup.
    te = df.iloc[te_idx].assign(pred=pred)
    err = []
    for col in ["road_user_group", "sex_label", "age_band_label", "severity_label"]:
        for k, g in te.groupby(col, observed=True):
            pos = g[g["ksi"] == 1]
            neg = g[g["ksi"] == 0]
            err.append({"variable": col, "group": k, "n": len(g), "n_ksi": len(pos),
                        "ksi_recall": pos["pred"].mean() if len(pos) else np.nan,
                        "false_positive_rate": neg["pred"].mean() if len(neg) else np.nan})
    pd.DataFrame(err).to_csv(TAB_DIR / "C_errors_by_group.csv", index=False)

    # Permutation importance (raw columns) on held-out test, PR-AUC scoring.
    pi = permutation_importance(pipe, Xte, yte, scoring="average_precision", n_repeats=5,
                                random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std})
    imp["group"] = np.where(imp["feature"].isin(DEMOGRAPHIC_NUM + DEMOGRAPHIC_CAT), "demographic", "other")
    imp = imp.sort_values("importance_mean", ascending=False)
    imp.to_csv(TAB_DIR / "C_permutation_importance.csv", index=False)

    # Univariate association of each non-demographic attribute with KSI (for the "three
    # additional attributes" selection, alongside model-based importance).
    uni = []
    for c in OTHER_CAT + OTHER_NUM:
        v = cramers_v(X[c].astype(str), y)
        r = ksi_rate_table(df.assign(_c=X[c].astype(str)), "_c")
        r = r[r["n"] >= 100]
        uni.append({"feature": c, "cramers_v_ksi": v, "max_ksi_rate_cat": r["ksi_rate"].idxmax(),
                    "max_ksi_rate": r["ksi_rate"].max(), "max_ksi_rate_n": int(r.loc[r["ksi_rate"].idxmax(), "n"]),
                    "completeness_pct": 100 * (X[c].astype(str) != "missing").mean()})
    uni = pd.DataFrame(uni).merge(imp[["feature", "importance_mean", "importance_std"]], on="feature")
    a_overlap = {"class_label": "casualty_class", "road_user_group": "road_user_group",
                 "sex_label": "sex_of_casualty", "age_band_label": "age_of_casualty",
                 "imd_label": "casualty_imd_decile", "distance_label": "casualty_distance_banding",
                 "ped_location_label": "pedestrian_location", "ped_movement_label": "pedestrian_movement"}
    used_in_a = {a_overlap.get(c, c) for c in a_selected_cols}
    uni["used_in_A"] = uni["feature"].isin(used_in_a)
    uni = uni.sort_values("importance_mean", ascending=False)
    additional = uni[~uni["used_in_A"]].head(3)["feature"].tolist()
    uni["selected_additional"] = uni["feature"].isin(additional)
    uni.to_csv(TAB_DIR / "C_additional_attributes.csv", index=False)
    SUMMARY["C_additional"] = additional

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp["feature"], imp["importance_mean"], xerr=imp["importance_std"],
            color=["#e67e22" if g == "demographic" else "#2c7fb8" for g in imp["group"]], capsize=2)
    ax.invert_yaxis()
    ax.set_xlabel("Drop in test PR-AUC when feature permuted (mean ± SD, 5 repeats)")
    ax.set_title(f"Permutation importance, {best} (orange = demographic)")
    savefig(fig, "fig07_permutation_importance.png")
    plt.close(fig)

    # Logistic regression odds ratios (for interpretation of direction).
    lr = fitted["Logistic regression"][0]
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    pd.DataFrame({"term": names, "coef": coefs, "odds_ratio": np.exp(coefs)}) \
        .sort_values("coef", ascending=False).to_csv(TAB_DIR / "C_lr_coefficients.csv", index=False)

    # Secondary 3-class view: per-class recall with a multinomial LR.
    ysev_tr = df["casualty_severity"].to_numpy()[tr_idx]
    ysev_te = df["casualty_severity"].to_numpy()[te_idx]
    m3 = make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced"), num, cat).fit(Xtr, ysev_tr)
    p3 = m3.predict(Xte)
    cm3 = pd.DataFrame(confusion_matrix(ysev_te, p3, labels=[1, 2, 3]),
                       index=["Actual Fatal", "Actual Serious", "Actual Slight"],
                       columns=["Pred Fatal", "Pred Serious", "Pred Slight"])
    cm3.to_csv(TAB_DIR / "C_three_class_confusion.csv")
    SUMMARY["three_class_recall"] = dict(zip(["Fatal", "Serious", "Slight"],
                                             recall_score(ysev_te, p3, labels=[1, 2, 3], average=None).round(3).tolist()))
    SUMMARY["three_class_balanced_accuracy"] = float(balanced_accuracy_score(ysev_te, p3))


def sensitivity(df):
    """Severity recording method and the DfT adjusted estimates (descriptive only)."""
    t = df.groupby("casualty_injury_based").agg(n=("ksi", "size"), ksi_rate=("ksi", "mean"),
                                                adjusted_serious_mean=("casualty_adjusted_severity_serious", "mean"))
    t.to_csv(TAB_DIR / "sensitivity_reporting_method.csv")
    SUMMARY["adjusted_vs_recorded"] = {
        "recorded_ksi_rate": float(df["ksi"].mean()),
        "adjusted_ksi_rate": float(((df["casualty_severity"] == 1) +
                                    df["casualty_adjusted_severity_serious"].where(df["casualty_severity"] != 1, 0)).mean()),
    }


def main():
    df = pd.read_csv(CLEAN_CSV, dtype={"collision_index": str, "collision_ref_no": str},
                     low_memory=False)
    for c in ["sex_of_casualty", "casualty_imd_decile", "casualty_distance_banding", "pedestrian_location",
              "pedestrian_movement", "car_passenger", "bus_or_coach_passenger", "casualty_type"]:
        df[c] = df[c].astype("Int64")
    sev = df["severity_label"].value_counts().reindex(["Fatal", "Serious", "Slight"])
    SUMMARY["severity_counts"] = sev.to_dict()
    _, a_cols = investigation_a(df)
    investigation_b(df)
    investigation_c(df, a_cols)
    sensitivity(df)
    (TAB_DIR / "summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    print(json.dumps(SUMMARY, indent=2, default=str))


if __name__ == "__main__":
    main()
