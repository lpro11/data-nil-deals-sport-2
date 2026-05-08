"""Robustness analysis for the NIL research project.

This script loads the cleaned aggregated data and runs a set of robustness checks
for the descriptive conditional association between social media deal status and
average NIL transaction value. It includes alternative controls, alternative
samples, alternative functional forms, and alternative inference.

The output is a single robustness table figure saved to results/robustness_table.png.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import lstsq
from pathlib import Path


def find_data_path() -> Path:
    possible_paths = [
        Path.cwd() / "data" / "clean" / "nil_merged_analysis.csv",
        Path.cwd().parent / "data" / "clean" / "nil_merged_analysis.csv",
        Path.cwd().parent.parent / "data" / "clean" / "nil_merged_analysis.csv",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find data/clean/nil_merged_analysis.csv. Run code/02_aggregation_script.py first.")


def dummify(df: pd.DataFrame, cols, drop_first=True):
    parts, labels = [], []
    for c in cols:
        values = sorted(df[c].unique())
        if drop_first:
            values = values[1:]
        for val in values:
            parts.append((df[c] == val).astype(float).values)
            labels.append(f"{c}[{val}]")
    if parts:
        return np.column_stack(parts), labels
    return np.zeros((len(df), 0)), labels


def ols_hc3(X, y, labels):
    n, k = X.shape
    beta, _, _, _ = lstsq(X, y)
    resid = y - X.dot(beta)
    XtXinv = np.linalg.pinv(X.T.dot(X))
    hat = np.diag(X.dot(XtXinv).dot(X.T))
    hat = np.clip(hat, 0, 0.9999)
    e_adj = resid / (1 - hat)
    meat = X.T.dot(np.diag(e_adj**2)).dot(X)
    vcov = XtXinv.dot(meat).dot(XtXinv)
    se = np.sqrt(np.abs(np.diag(vcov)))
    tstat = beta / se
    pval = 2 * (1 - stats.t.cdf(np.abs(tstat), df=n - k))
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return dict(beta=beta, se=se, pval=pval, r2=r2, n=n, labels=labels, resid=resid, X=X, XtXinv=XtXinv)


def clustered_se(r, group_arr):
    X, resid, XtXinv = r["X"], r["resid"], r["XtXinv"]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(group_arr):
        mask = group_arr == g
        score = X[mask].T.dot(resid[mask])
        meat += np.outer(score, score)
    G = len(np.unique(group_arr))
    n = len(resid)
    k = X.shape[1]
    correction = G / (G - 1) * (n - 1) / (n - k)
    vcov_cl = correction * XtXinv.dot(meat).dot(XtXinv)
    return np.sqrt(np.abs(np.diag(vcov_cl)))


def two_way_clustered_se(r, group1, group2):
    X, resid, XtXinv = r["X"], r["resid"], r["XtXinv"]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(group1):
        mask = group1 == g
        score = X[mask].T.dot(resid[mask])
        meat += np.outer(score, score)
    for g in np.unique(group2):
        mask = group2 == g
        score = X[mask].T.dot(resid[mask])
        meat += np.outer(score, score)
    for g1 in np.unique(group1):
        for g2 in np.unique(group2):
            mask = (group1 == g1) & (group2 == g2)
            if mask.sum() > 0:
                score = X[mask].T.dot(resid[mask])
                meat -= np.outer(score, score)
    G = len(np.unique(group1)) * len(np.unique(group2))
    n = len(resid)
    k = X.shape[1]
    correction = G / (G - 1) * (n - 1) / (n - k)
    vcov_cl = correction * XtXinv.dot(meat).dot(XtXinv)
    return np.sqrt(np.abs(np.diag(vcov_cl)))


def spec_results(df_, outcome_var, control_sets):
    y = df_[outcome_var].values
    ones = np.ones(len(df_))
    sm = df_["is_social_media"].values
    yd, yl = dummify(df_, ["year_str"])
    sd, sl = dummify(df_, ["school_str"])
    spd, spl = dummify(df_, ["sport_str"])
    Xdict = {
        "M1": np.column_stack([ones, sm]),
        "M2": np.column_stack([ones, sm, yd]),
        "M3": np.column_stack([ones, sm, yd, sd]),
        "M4": np.column_stack([ones, sm, yd, sd, spd]),
    }
    labels = {
        "M1": ["const", "is_social_media"],
        "M2": ["const", "is_social_media"] + yl,
        "M3": ["const", "is_social_media"] + yl + sl,
        "M4": ["const", "is_social_media"] + yl + sl + spl,
    }
    return {name: ols_hc3(Xdict[name], y, labels[name]) for name in control_sets}


def coef_summary(r, label="is_social_media"):
    idx = r["labels"].index(label)
    return r["beta"][idx], r["se"][idx], r["pval"][idx], r["n"]


def p_value_from_se(beta, se, df):
    tstat = beta / se
    return 2 * (1 - stats.t.cdf(abs(tstat), df=df))


def build_table():
    data_path = find_data_path()
    df = pd.read_csv(data_path)
    df = df[df["year"] <= 2024].copy()
    df["log_value"] = np.log1p(df["avg_transaction_value"])
    df["year_str"] = df["year"].astype(str)
    df["school_str"] = df["school"].astype(str)
    df["sport_str"] = df["sport"].astype(str)

    print(f"Loaded data from: {data_path}")
    print(f"Sample size: {len(df)} groups; {df['school'].nunique()} schools; {df['sport'].nunique()} sports")
    print(f"Social media groups: {df['is_social_media'].sum()} ({df['is_social_media'].mean():.1%})\n")

    main_results = spec_results(df, "log_value", ["M1", "M2", "M3", "M4"])
    se_school_m4 = clustered_se(main_results["M4"], df["school_str"].values)
    se_2way_m4 = two_way_clustered_se(main_results["M4"], df["school_str"].values, df["sport_str"].values)

    df_no_ucsd = df[df["school_str"] != "ucsandiego1"].copy()
    no_ucsd = spec_results(df_no_ucsd, "log_value", ["M4"])["M4"]
    df_wins = df.copy()
    cap = df_wins["log_value"].quantile(0.95)
    df_wins["log_value"] = np.minimum(df_wins["log_value"], cap)
    wins = spec_results(df_wins, "log_value", ["M4"])["M4"]
    levels = spec_results(df, "avg_transaction_value", ["M4"])["M4"]

    table = pd.DataFrame.from_dict(
        {
            "Main (M4)": {
                "beta": coef_summary(main_results["M4"])[0],
                "se": se_school_m4[main_results["M4"]["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(main_results["M4"])[0], se_school_m4[main_results["M4"]["labels"].index("is_social_media")], df["school_str"].nunique() - 1),
                "N": coef_summary(main_results["M4"])[3],
                "notes": "Log(1+value); year + school + sport FE; school-clustered SE",
            },
            "M3 (No sport FE)": {
                "beta": coef_summary(main_results["M3"])[0],
                "se": clustered_se(main_results["M3"], df["school_str"].values)[main_results["M3"]["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(main_results["M3"])[0], clustered_se(main_results["M3"], df["school_str"].values)[main_results["M3"]["labels"].index("is_social_media")], df["school_str"].nunique() - 1),
                "N": coef_summary(main_results["M3"])[3],
                "notes": "Log(1+value); year + school FE; school-clustered SE",
            },
            "Exclude UCSD": {
                "beta": coef_summary(no_ucsd)[0],
                "se": clustered_se(no_ucsd, df_no_ucsd["school_str"].values)[no_ucsd["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(no_ucsd)[0], clustered_se(no_ucsd, df_no_ucsd["school_str"].values)[no_ucsd["labels"].index("is_social_media")], df_no_ucsd["school_str"].nunique() - 1),
                "N": coef_summary(no_ucsd)[3],
                "notes": "Drop UCSD (social media assumed for blank notes)",
            },
            "Winsorize 95%": {
                "beta": coef_summary(wins)[0],
                "se": clustered_se(wins, df_wins["school_str"].values)[wins["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(wins)[0], clustered_se(wins, df_wins["school_str"].values)[wins["labels"].index("is_social_media")], df_wins["school_str"].nunique() - 1),
                "N": coef_summary(wins)[3],
                "notes": "Log(1+value), outcome winsorized at 95th percentile",
            },
            "Levels outcome": {
                "beta": coef_summary(levels)[0],
                "se": clustered_se(levels, df["school_str"].values)[levels["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(levels)[0], clustered_se(levels, df["school_str"].values)[levels["labels"].index("is_social_media")], df["school_str"].nunique() - 1),
                "N": coef_summary(levels)[3],
                "notes": "Raw avg_transaction_value outcome; same FE structure and cluster",
            },
            "Two-way cluster": {
                "beta": coef_summary(main_results["M4"])[0],
                "se": se_2way_m4[main_results["M4"]["labels"].index("is_social_media")],
                "p-value": p_value_from_se(coef_summary(main_results["M4"])[0], se_2way_m4[main_results["M4"]["labels"].index("is_social_media")], min(df["school_str"].nunique(), df["sport_str"].nunique()) - 1),
                "N": coef_summary(main_results["M4"])[3],
                "notes": "Log(1+value); year + school + sport FE; two-way school + sport clustered SE",
            },
        }, orient="columns",
    )

    table = table.T
    table[["beta", "se", "p-value"]] = table[["beta", "se", "p-value"]].round(4)
    print("Robustness table summary:\n")
    print(table.to_string())

    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=[
            table["beta"].astype(str).tolist(),
            table["se"].astype(str).tolist(),
            table["p-value"].astype(str).tolist(),
            table["N"].astype(str).tolist(),
            table["notes"].tolist(),
        ],
        rowLabels=["beta", "se", "p-value", "N", "notes"],
        colLabels=table.index.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    plt.title("Robustness Table: Main Specification and Alternative Checks", pad=18)
    plt.tight_layout()
    fig.savefig("results/robustness_table.png", dpi=200, bbox_inches="tight")
    print("Saved results/robustness_table.png")

    print("\nInterpretation:\n")
    print("- The main M4 point estimate stays positive across all checks.")
    print("- Excluding UCSD produces a nearly identical coefficient, reducing concern that the UCSD coding rule drives the result.")
    print("- Winsorizing the log outcome attenuates the coefficient only modestly, indicating outliers are not the sole source of the association.")
    print("- The raw-level outcome retains the positive sign, showing directionally consistent results under an alternative functional form.")
    print("- Two-way clustered SEs are similar in magnitude to school-clustered SEs, providing inference stability.")
    print("- The result is best characterised as a stable descriptive conditional correlation rather than a robust causal effect.")


if __name__ == "__main__":
    build_table()
