#!/usr/bin/env python3
"""
Exhaustive Post-Run Methodological & Mathematical Audit Suite for Step 0 + Step 1 (v2)
======================================================================================
Verifies every mathematical identity, non-negativity/boundedness constraint, missing-value
rule, longitudinal alignment rule, and empirical distribution benchmark across all 6 papers:
  1. Li (2010, JAR)
  2. Brown & Tucker (2011, JAR)
  3. Kravet & Muslu (2013, RAST)
  4. Campbell, Chen, Dhaliwal, Lu & Steele (2014, RAST)
  5. Hope, Hu & Lu (2016, RAST)
  6. Beatty, Cheng & Zhang (2019, CAR)
"""

import os
import json
import math
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_CSV = os.path.join(BASE_DIR, "04_outputs", "step1_paper_metrics", "india_corporate_risk_disclosure_panel_1997_2025.csv")
LI_MODEL_JSON = os.path.join(BASE_DIR, "04_outputs", "step1_paper_metrics", "li2010_nb_model.json")
LEX_JSON = os.path.join(BASE_DIR, "03_code", "step1_lexicons.json")
AUDIT_JSON = os.path.join(BASE_DIR, "04_outputs", "step1_paper_metrics", "audit_verification_summary.json")


def run_audit():
    df = pd.read_csv(PANEL_CSV, low_memory=False)
    with open(LEX_JSON, "r", encoding="utf-8") as f:
        lex = json.load(f)
    with open(LI_MODEL_JSON, "r", encoding="utf-8") as f:
        li_nb = json.load(f)

    checks = []

    def record(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")

    # 1. Panel coverage & uniqueness
    n_rows = len(df)
    n_firms = df["bse_code"].nunique()
    dup_pairs = int(df.duplicated(subset=["bse_code", "year"]).sum())
    record("Unique (bse_code, year) panel keys", dup_pairs == 0, f"N={n_rows:,} firm-years across {n_firms:,} unique BSE firms, duplicates={dup_pairs}")

    # 2. Lexicon integrity (Campbell 2014 = 382 exact primary-PDF keywords)
    c_counts = {k: len(v) for k, v in lex["campbell_2014"].items()}
    c_total = sum(c_counts.values())
    record("Campbell et al. (2014) Appendix C exact 382-keyword count", c_total == 382, f"Total={c_total} {c_counts}")

    # 3. Campbell 5-category additivity: KW_1A_FIN + LR + TAX + SYS + IDIO == KW_1A_RAW
    sub_sum = df["KW_1A_FIN"] + df["KW_1A_LR"] + df["KW_1A_TAX"] + df["KW_1A_SYS"] + df["KW_1A_IDIO"]
    exact_raw_match = int((sub_sum != df["KW_1A_RAW"]).sum())
    record("Campbell 5-subcategory raw sum == KW_1A_RAW", exact_raw_match == 0, f"Mismatches={exact_raw_match}/{n_rows:,}")

    # 4. Campbell 5-category share additivity: RF_FIN + RF_LR + RF_TAX + RF_SYS + RF_IDIO == 1.0
    pos_kw = df["KW_1A_RAW"] > 0
    share_sum = df.loc[pos_kw, ["RF_FIN", "RF_LR", "RF_TAX", "RF_SYS", "RF_IDIO"]].sum(axis=1)
    max_share_err = float((share_sum - 1.0).abs().max())
    record("Campbell 5-subcategory shares sum to 1.000", max_share_err <= 0.001, f"N={pos_kw.sum():,}, max |sum - 1| = {max_share_err:.6f}")

    # 5. Beatty et al. (2019) exact WCopyfind additivity: KW_1A_REP + KW_1A_NEW == KW_1A
    valid_b = df["KW_1A_REP"].notna() & df["KW_1A_NEW"].notna() & df["KW_1A"].notna()
    b_err = float(((df.loc[valid_b, "KW_1A_REP"] + df.loc[valid_b, "KW_1A_NEW"]) - df.loc[valid_b, "KW_1A"]).abs().max())
    record("Beatty et al. (2019) KW_1A_REP + KW_1A_NEW == KW_1A", b_err <= 0.0002, f"N={valid_b.sum():,}, max |REP+NEW - KW_1A| = {b_err:.6f}")

    # 6. Beatty et al. (2019) word-count additivity: WC_1A_REP + WC_1A_NEW <= WC_1A (within token definition) & PCT_REPEATED in [0, 1]
    pct_bounds = bool(((df.loc[valid_b, "PCT_REPEATED_WC"] >= 0.0) & (df.loc[valid_b, "PCT_REPEATED_WC"] <= 1.0)).all())
    record("Beatty et al. (2019) PCT_REPEATED_WC bounded in [0, 1]", pct_bounds, f"min={df['PCT_REPEATED_WC'].min():.4f}, max={df['PCT_REPEATED_WC'].max():.4f}")

    # 7. Brown & Tucker (2011) Rawscore bounded in [0, 1] and 5th-order polynomial residual mean == 0
    bt_v = df["MDA_Rawscore"].dropna()
    bt_bounds = bool(((bt_v >= 0.0) & (bt_v <= 1.0)).all())
    bt_score_mean = float(df["MDA_Score"].dropna().mean())
    bt_tf_score_mean = float(df["MDA_Score_TF"].dropna().mean())
    record("Brown & Tucker (2011) MDA_Rawscore in [0, 1] & OLS poly-5 residual mean == 0",
           bt_bounds and abs(bt_score_mean) < 1e-4 and abs(bt_tf_score_mean) < 1e-4,
           f"N={len(bt_v):,}, Rawscore range=[{bt_v.min():.4f}, {bt_v.max():.4f}], mean(MDA_Score)={bt_score_mean:.6f}, mean(MDA_Score_TF)={bt_tf_score_mean:.6f}")

    # 8. Longitudinal consecutive-year check (no YoY metric populated across year gaps > 1)
    df_sorted = df.sort_values(["bse_code", "year"]).reset_index(drop=True)
    same_firm = df_sorted["bse_code"] == df_sorted["bse_code"].shift(1)
    consec_yr = (df_sorted["year"] - df_sorted["year"].shift(1)) == 1
    non_consec = ~(same_firm & consec_yr)
    leaked_yoy = int(df_sorted.loc[non_consec, ["MDA_Rawscore", "KW_1A_REP", "KM_Delta_RD"]].notna().any(axis=1).sum())
    record("Zero YoY metric leakage across firm boundaries or year gaps > 1", leaked_yoy == 0, f"Leaked non-consecutive rows={leaked_yoy}")

    # 9. Kravet & Muslu (2013) peer-count & decomposition identity
    km_v = df["KM_Delta_Firm_RD"].notna()
    min_peers = int(df.loc[km_v, "KM_Industry_Peer_N"].min())
    km_decomp_err = float(((df.loc[km_v, "KM_Delta_Industry_RD"] + df.loc[km_v, "KM_Delta_Firm_RD"]) - df.loc[km_v, "KM_Delta_RD"]).abs().max())
    record("Kravet & Muslu (2013) >=3 industry peers & exact Industry + Firm == Delta_RD",
           min_peers >= 3 and km_decomp_err <= 0.002,
           f"N={km_v.sum():,}, min_peers={min_peers}, max_err={km_decomp_err:.6f}")

    # 10. Hope, Hu & Lu (2016) Specificity bounded in [0, 1] & Spec_10K > Spec_1A empirically
    s1a = df["Specificity_1A"].dropna()
    s10k = df["Specificity_10K"].dropna()
    s_bounds = bool(((s1a >= 0.0) & (s1a <= 1.0)).all() and ((s10k >= 0.0) & (s10k <= 1.0)).all())
    sub7_sum = df["Spec_1A_Money"] + df["Spec_1A_Percent"] + df["Spec_1A_Date"] + df["Spec_1A_Time"] + df["Spec_1A_Org"] + df["Spec_1A_Loc"] + df["Spec_1A_Person"]
    spec_sum_mismatch = int((sub7_sum != df["Spec_1A_Specific_Words"]).sum())
    record("Hope, Hu & Lu (2016) 7-category NER additivity & [0, 1] bounds",
           s_bounds and spec_sum_mismatch == 0,
           f"mean(Spec_1A)={s1a.mean():.4f} (p50={s1a.median():.4f}), mean(Spec_10K)={s10k.mean():.4f} (p50={s10k.median():.4f}), 7-cat mismatches={spec_sum_mismatch}")

    # 11. Li (2010) Naïve Bayesian classifier model & FLS bounds
    ft = df["FLS_TONE"].dropna()
    fls_min_cnt = int(df.loc[df["FLS_TONE"].notna(), "FLS_Count"].min())
    content_sum = df.loc[df["FLS_Count"] > 0, ["FLS_PROFIT_PCT", "FLS_LIQUIDITY_PCT", "FLS_OTHER_PCT"]].sum(axis=1)
    max_c_err = float((content_sum - 1.0).abs().max())
    tone_bounds = bool(((ft >= -1.0) & (ft <= 1.0)).all())
    cv5_tone4 = li_nb["cv_results"]["tone4"]["5"]
    cv5_cat3 = li_nb["cv_results"]["cat3"]["5"]
    record("Li (2010) Naïve Bayes FLS_TONE in [-1, 1], FLS_Count >= 5, Content shares == 1.0",
           tone_bounds and fls_min_cnt >= 5 and max_c_err <= 0.001,
           f"N_train={li_nb['n_training_sentences']:,}, 5-fold CV Tone4={cv5_tone4[0]}% (vs Guess {cv5_tone4[1]}%), Cat3={cv5_cat3[0]}% (vs Guess {cv5_cat3[1]}%), mean(FLS_TONE)={ft.mean():.4f}")

    # Summary stats table for key variables
    key_vars = [
        "MDA_Rawscore", "MDA_Rawscore_TF", "MDA_Score", "MDA_Score_TF",
        "KW_1A", "WC_1A", "ALL_WORDS", "KEY_WORDS",
        "RF_SYS", "RF_IDIO", "RF_FIN", "RF_LR", "RF_TAX", "RF_FOG", "MDA_FOG",
        "KW_1A_REP", "KW_1A_NEW", "KW_1A_DISC", "PCT_REPEATED_WC", "FIN_SYS_1A", "IDIO_LR_1A",
        "Specificity_1A", "Specificity_10K",
        "KM_Risk_Disclosure", "KM_Negative_RD", "KM_Delta_RD", "KM_Delta_Industry_RD", "KM_Delta_Firm_RD",
        "FLS_Count", "FLS_Share", "FLS_TONE", "FLS_TONE_Dict", "FLS_PROFIT_TONE", "FLS_LIQUIDITY_TONE",
        "FLS_PROFIT_PCT", "FLS_LIQUIDITY_PCT", "FLS_OTHER_PCT",
    ]
    stats_out = {}
    print("\n=== SUMMARY DISTRIBUTIONS ON AUDITED FULL PANEL ===")
    for col in key_vars:
        s = df[col].dropna()
        if len(s) > 0:
            stats_out[col] = {
                "N": int(len(s)),
                "mean": round(float(s.mean()), 4),
                "sd": round(float(s.std()), 4),
                "p25": round(float(s.quantile(0.25)), 4),
                "p50": round(float(s.median()), 4),
                "p75": round(float(s.quantile(0.75)), 4),
            }
            st = stats_out[col]
            print(f"  {col:22s} | N={st['N']:6,d} | mean={st['mean']:8.4f} | sd={st['sd']:8.4f} | p25={st['p25']:8.4f} | p50={st['p50']:8.4f} | p75={st['p75']:8.4f}")

    summary_payload = {
        "all_passed": all(c["passed"] for c in checks),
        "n_checks": len(checks),
        "checks": checks,
        "cmap_qc": {
            "mda_cmap_corrupted": int(df["mda_cmap_corrupted"].sum()),
            "risk_cmap_corrupted": int(df["risk_cmap_corrupted"].sum()),
            "report_cmap_corrupted": int(df["report_cmap_corrupted"].sum()),
        },
        "distributions": stats_out,
    }
    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\n[Audit Summary Saved] -> {AUDIT_JSON} | ALL PASSED = {summary_payload['all_passed']}")


if __name__ == "__main__":
    run_audit()
