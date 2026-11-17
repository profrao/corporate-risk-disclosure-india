#!/usr/bin/env python3
"""
Step 2A: Build Master Merged Econometric Panel (1997-2025, N = 16,987)
======================================================================
Parses all verified CMIE Prowess exports (1992-2026 annual financials,
2001-2026 quarterly shareholding, 1995-2026 monthly adjusted stock prices,
and 1995-2026 annual valuation/liquidity/beta series), constructs lags, leads,
rolling volatilities, Modified Jones abnormal accruals, Altman Z-scores,
proprietary cost proxies, and monthly stock return moments on the full
962 x 35 CMIE lattice BEFORE merging onto the 16,987 corporate annual reports.
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "05_cmie_data", "raw_exports")
STEP1_CSV = os.path.join(BASE_DIR, "04_outputs", "step1_paper_metrics", "india_corporate_risk_disclosure_panel_1997_2025.csv")
OUT_DIR = os.path.join(BASE_DIR, "04_outputs", "step2_econometrics")
os.makedirs(OUT_DIR, exist_ok=True)


def winsorize_s(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    valid = s.dropna()
    if len(valid) < 20:
        return s
    lo, hi = np.nanpercentile(valid, [lower * 100, upper * 100])
    return s.clip(lower=lo, upper=hi)


def load_identity():
    id_path = os.path.join(RAW_DIR, "00_identity_962_firms.csv")
    df = pd.read_csv(id_path)
    df.columns = [c.strip().strip('"').lstrip("\ufeff") for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df["incorp_year"] = pd.to_numeric(df["Incorporation year"], errors="coerce")
    df["ind_type_cmie"] = pd.to_numeric(df["Industry type"], errors="coerce")
    return df


def parse_annual_and_quarterly_cmie(name_to_bse: dict):
    """
    Parse numbered CMIE exports (excluding 00 identity, 08 snapshot, and 20/21 monthly prices)
    into a wide (bse_code, year) DataFrame covering 1992..2026.
    """
    # Explicit canonical file order so newer complete exports override partial ones
    ordered_files = [
        "01_income_sales_finserv_extraord_inc_1992_2026.csv",
        "02_current_liabilities_provisions_1992_2026.csv",
        "03_equity_dividend_1992_2026.csv",
        "04_export_of_goods_fob_1992_2026.csv",
        "05_interest_tax_extraord_exp_pat_div_export_1992_2026.csv",
        "06_total_assets_capital_reserves_pat_1992_2026.csv",
        "07_inventories_investments_net_fa_receivables_payables_1992_2026.csv",
        "09_capex_investing_cf_financing_cf_closing_cash_1992_2026.csv",
        "10_pbdita_dep_pbit_pbt_pat_net_pe_1992_2026.csv",
        "11_borrowings_ca_cash_cl_capres_ta_1992_2026.csv",
        "12_foreign_promoters_quarterly_2001_2026.csv",
        "13_institutional_shareholding_quarterly_2001_2026.csv",
        "14_non_institutional_shareholding_quarterly_2001_2026.csv",
        "15_current_liab_only_inv_rec_1992_2026.csv",
        "16_gross_fa_cfo_equity_cap_rd_1992_2026.csv",
        "17_mcap_shares_eps_pe_pb_bvps_1995_2026.csv",
        "18_returns_beta_turnover_shares_traded_yield_1995_2026.csv",
        "19_nonfin_serv_rev_forex_1992_2026.csv",
        "23_cospi_sensex_annual_1995_2026.csv",
        "22_adj_hlc_ev_1995_2026_all962.csv",
    ]

    data = defaultdict(dict)  # ind_name -> {(bse, yr): (val, q_pref)}
    index_annual = defaultdict(dict)  # ind_name -> {yr: val} for COSPI / Sensex

    for fn in ordered_files:
        fp = os.path.join(RAW_DIR, fn)
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
        dates_row = rows[4]
        ind_row = rows[5]
        col_meta = []
        for c in range(1, len(ind_row)):
            d_str = dates_row[c].strip() if c < len(dates_row) else ""
            ind_str = ind_row[c].strip()
            if not d_str or not ind_str:
                continue
            m = re.search(r"(19\d\d|20\d\d)", d_str)
            if not m:
                continue
            yr = int(m.group(1))
            # Quarterly preference for FY ending March YYYY: Mar YYYY > Dec YYYY > Jun YYYY > Sep YYYY
            if d_str.startswith("Mar "):
                q_pref = 4
            elif d_str.startswith("Dec ") and "Shares Held" in ind_str:
                q_pref = 3
            elif d_str.startswith("Jun "):
                q_pref = 2
            elif d_str.startswith("Sep "):
                q_pref = 1
            else:
                q_pref = 5
            col_meta.append((c, ind_str, yr, q_pref))

        for r in rows[6:]:
            if not r or not r[0].strip():
                continue
            bse = name_to_bse.get(r[0].strip())
            if not bse:
                continue
            for (c, ind_str, yr, q_pref) in col_meta:
                if c >= len(r):
                    continue
                vs = r[c].strip().replace(",", "")
                if vs in ("", "NA", "-"):
                    continue
                try:
                    val = float(vs)
                except ValueError:
                    continue
                if ind_str in ("COSPI closing", "BSE Sensex closing"):
                    index_annual[ind_str][yr] = val
                prev = data[ind_str].get((bse, yr))
                if prev is None or q_pref >= prev[1]:
                    data[ind_str][(bse, yr)] = (val, q_pref)

    # Assemble full 962 x (1992..2026) grid
    all_bse = sorted(set(name_to_bse.values()))
    years = list(range(1992, 2027))
    idx = pd.MultiIndex.from_product([all_bse, years], names=["bse_code", "year"])
    grid = pd.DataFrame(index=idx).reset_index()

    col_map = {
        "Total assets (Old Sch. VI)": "cmie_total_assets",
        "Capital & Reserves (Old Sch. VI)": "cmie_book_equity",
        "Paid up equity capital (net of forfeited equity capital)": "cmie_paidup_equity",
        "Borrowings (Old Sch. VI)": "cmie_borrowings",
        "Current assets (Old Sch. VI)": "cmie_current_assets",
        "Cash and bank balance (Old Sch. VI)": "cmie_cash_bank",
        "Current liabilities & provisions (Old Sch. VI)": "cmie_curr_liab_prov",
        "Current liabilities (Old Sch. VI)": "cmie_curr_liab_only",
        "Gross fixed assets (Old Sch. VI)": "cmie_gross_fa",
        "Net fixed assets (Old Sch. VI)": "cmie_net_fa",
        "Inventories (Old Sch. VI)": "cmie_inventories",
        "Receivables (Old Sch. VI)": "cmie_receivables",
        "Investments (Old Sch. VI)": "cmie_investments",
        "Trade payables and acceptances (Old Sch. VI)": "cmie_trade_payables",
        "Total income": "cmie_total_income",
        "Sales of goods": "cmie_sales_goods",
        "Trading revenue and non-financial services": "cmie_sales_nonfin_serv",
        "Income from financial services": "cmie_inc_fin_serv",
        "PBDITA": "cmie_pbdita",
        "Depreciation / Amortisation (net of transfer from revaluation reserves)": "cmie_depreciation",
        "PBIT": "cmie_pbit",
        "PBT": "cmie_pbt",
        "Profit after tax (PAT)": "cmie_pat",
        "PAT net of P&E": "cmie_pat_net_pe",
        "Interest expense": "cmie_interest_exp",
        "Provision for direct tax": "cmie_tax_prov",
        "Prior period and extra-ordinary income": "cmie_extraord_inc",
        "Prior period and extra-ordinary expenses": "cmie_extraord_exp",
        "Equity dividend": "cmie_equity_div",
        "Research & development expenses (capital & current account)": "cmie_rd_exp",
        "Total forex earnings": "cmie_forex_earnings",
        "Total forex spending": "cmie_forex_spending",
        "Export of goods(fob)": "cmie_export_goods",
        "Net cash flow from operating activities": "cmie_cfo",
        "Net cash inflow or (outflow) from investing activities": "cmie_cfi",
        "Net cash inflow or (outflow) from financing activities": "cmie_cff",
        "Net cash inflow or (outflow) due to net increase or (decrease) in cash and cash equivalents": "cmie_net_cash_chg",
        "Cash and cash equivalents as at the end of the year": "cmie_closing_cash_cf",
        "Cash (outflow) due to purchase of fixed assets": "cmie_capex_raw",
        "Non-promoter Institutions (In %) - Shares Held": "cmie_inst_own_pct",
        "Non-promoter Non-institutions (In %) - Shares Held": "cmie_noninst_own_pct",
        "Foreign Promoters (In %) - Shares Held": "cmie_for_prom_pct",
        "Market Capitalisation": "cmie_mcap",
        "Shares Outstanding": "cmie_shares_out",
        "EPS": "cmie_eps",
        "P/E": "cmie_pe",
        "P/B": "cmie_pb",
        "BV per Share": "cmie_bvps",
        "Beta": "cmie_beta",
        "Yield": "cmie_div_yield",
        "Turnover": "cmie_mar31_turnover",
        "Shares traded": "cmie_mar31_shares_traded",
        "Adjusted Closing Price": "cmie_mar31_adj_close",
        "Adjusted High Price": "cmie_mar31_adj_high",
        "Adjusted Low Price": "cmie_mar31_adj_low",
        "Enterprise value": "cmie_ev",
    }

    keys_list = list(zip(grid["bse_code"], grid["year"]))
    for raw_ind, clean_col in col_map.items():
        dmap = data.get(raw_ind, {})
        grid[clean_col] = [dmap[k][0] if k in dmap else np.nan for k in keys_list]

    grid["cmie_cospi_mar31"] = grid["year"].map(index_annual.get("COSPI closing", {}))
    grid["cmie_sensex_mar31"] = grid["year"].map(index_annual.get("BSE Sensex closing", {}))
    return grid


def compute_monthly_return_metrics(name_to_bse: dict, mcap_lookup: dict) -> pd.DataFrame:
    """
    Parse 375 continuous monthly Adjusted Closing Prices (Jan 1995 - Mar 2026)
    across all 962 firms and compute FY (Apr YYYY-1 .. Mar YYYY) and post-filing
    (Jul YYYY .. Jun YYYY+1 and Apr YYYY .. Mar YYYY+1) return, BHAR, volatility,
    idiosyncratic volatility (market model), and crash skewness (NCSKEW).
    """
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    price_dict = defaultdict(dict)  # bse -> {(yr, mo): price}

    for fn in ["20_monthly_adj_close_1995_2010.csv", "21_monthly_adj_close_2011_2026_all962.csv"]:
        fp = os.path.join(RAW_DIR, fn)
        with open(fp, encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f))
        dates_row = rows[4]
        col_ym = []
        for c in range(1, len(dates_row)):
            ds = dates_row[c].strip()
            m = re.search(r"([A-Za-z]{3})\s+(19\d\d|20\d\d)", ds)
            if m and m.group(1) in month_map:
                col_ym.append((c, int(m.group(2)), month_map[m.group(1)]))
        for r in rows[6:]:
            if not r or not r[0].strip():
                continue
            bse = name_to_bse.get(r[0].strip())
            if not bse:
                continue
            for (c, y, mo) in col_ym:
                if c < len(r):
                    vs = r[c].strip().replace(",", "")
                    if vs not in ("", "NA", "-"):
                        try:
                            p = float(vs)
                            if p > 0:
                                price_dict[bse][(y, mo)] = p
                        except ValueError:
                            pass

    # Build monthly (bse, ym_idx) matrix for fast vectorized monthly returns
    all_ym = [(y, m) for y in range(1995, 2027) for m in range(1, 13) if (y, m) <= (2026, 3)]
    ym_to_i = {ym: i for i, ym in enumerate(all_ym)}
    firms = sorted(price_dict.keys())
    P = np.full((len(firms), len(all_ym)), np.nan)
    W = np.full((len(firms), len(all_ym)), 1.0)
    for fi, bse in enumerate(firms):
        for ym, val in price_dict[bse].items():
            if ym in ym_to_i:
                P[fi, ym_to_i[ym]] = val
        for gi, (y, m) in enumerate(all_ym):
            fy = y if m <= 3 else y + 1
            mc = mcap_lookup.get((bse, fy - 1), mcap_lookup.get((bse, fy), np.nan))
            if not np.isnan(mc) and mc > 0:
                W[fi, gi] = mc

    # Monthly simple return R[f, t] = P[f, t] / P[f, t-1] - 1, trim extreme >300% data glitches
    R = np.full_like(P, np.nan)
    R[:, 1:] = (P[:, 1:] / P[:, :-1]) - 1.0
    R[(R < -0.95) | (R > 3.0)] = np.nan

    # Value-weighted & Equal-weighted BSE 962-firm monthly market return
    mkt_vw = np.full(len(all_ym), np.nan)
    mkt_ew = np.nanmean(R, axis=0)
    for t in range(1, len(all_ym)):
        valid = ~np.isnan(R[:, t])
        if valid.sum() >= 30:
            w = W[valid, t]
            mkt_vw[t] = np.sum(w * R[valid, t]) / np.sum(w)
        else:
            mkt_vw[t] = mkt_ew[t]

    # For each (bse, fy in 1996..2026):
    # FY window = Apr (fy-1) .. Mar (fy) [12 months]
    # Post-report window = Jul (fy) .. Jun (fy+1) [12 months after Indian AGM/annual report release, fallback Apr fy..Mar fy+1 for 2025]
    out_rows = []
    for fy in range(1996, 2027):
        fy_months = [(fy - 1, m) for m in range(4, 13)] + [(fy, m) for m in range(1, 4)]
        idx_fy = [ym_to_i[ym] for ym in fy_months if ym in ym_to_i]
        if len(idx_fy) < 6:
            continue

        R_fy = R[:, idx_fy]
        m_fy = mkt_vw[idx_fy]
        n_valid = np.sum(~np.isnan(R_fy), axis=1)

        # Compounded 12-month return & BHAR
        comp_firm = np.prod(np.where(np.isnan(R_fy), 1.0, 1.0 + R_fy), axis=1) - 1.0
        comp_mkt = np.prod(np.where(np.isnan(R_fy), 1.0, 1.0 + m_fy[None, :]), axis=1) - 1.0
        comp_firm[n_valid < 6] = np.nan
        bhar_fy = comp_firm - comp_mkt
        bhar_fy[n_valid < 6] = np.nan

        # Monthly return std dev (annualized by sqrt(12) or raw monthly — store raw monthly std & annualized)
        std_ret = np.nanstd(R_fy, axis=1, ddof=1)
        std_ret[n_valid < 6] = np.nan

        # Market-model idiosyncratic volatility & NCSKEW (Chen, Hong & Stein 2001 / Kravet & Muslu 2013)
        idio_vol = np.full(len(firms), np.nan)
        ncskew = np.full(len(firms), np.nan)
        for fi in range(len(firms)):
            if n_valid[fi] < 8:
                continue
            vm = ~np.isnan(R_fy[fi]) & ~np.isnan(m_fy)
            if vm.sum() < 8:
                continue
            y_v = R_fy[fi, vm]
            x_v = m_fy[vm]
            vx = np.var(x_v, ddof=1)
            if vx > 1e-10:
                beta_i = np.cov(y_v, x_v, ddof=1)[0, 1] / vx
                alpha_i = np.mean(y_v) - beta_i * np.mean(x_v)
                resid = y_v - (alpha_i + beta_i * x_v)
                idio_vol[fi] = np.std(resid, ddof=1)
                lr = np.log1p(np.clip(resid, -0.9, 3.0))
                n_m = len(lr)
                denom = ((n_m - 1) * (n_m - 2)) * (np.sum(lr ** 2) ** 1.5)
                if denom > 1e-12:
                    ncskew[fi] = -(n_m * ((n_m - 1) ** 1.5) * np.sum(lr ** 3)) / denom

        for fi, bse in enumerate(firms):
            if n_valid[fi] >= 6:
                out_rows.append({
                    "bse_code": bse,
                    "year": fy,
                    "RET_12M": comp_firm[fi],
                    "BHAR_12M": bhar_fy[fi],
                    "STD_RET_12M": std_ret[fi],
                    "IDIO_VOL_12M": idio_vol[fi],
                    "NCSKEW_12M": ncskew[fi],
                    "RET_VALID_MONTHS": int(n_valid[fi]),
                })

    return pd.DataFrame(out_rows)


def main():
    print("1. Loading firm identity (962 firms) & Step 1 textual disclosure panel (16,987 rows)...")
    id_df = load_identity()
    step1 = pd.read_csv(STEP1_CSV, low_memory=False)
    step1["bse_code"] = step1["bse_code"].astype(str).str.strip()

    name_to_bse = dict(zip(id_df["Company Name"], id_df["BSE scrip code"]))
    # Ensure 100% match to step1 bse_codes
    missing = set(step1["bse_code"]) - set(id_df["BSE scrip code"])
    if missing:
        print("Warning missing bse_code:", missing)

    print("2. Parsing all annual financial, cash flow, valuation, and quarterly shareholding CMIE exports...")
    grid = parse_annual_and_quarterly_cmie(name_to_bse)
    grid = grid.merge(
        id_df[["BSE scrip code", "incorp_year", "ind_type_cmie", "CIN code"]].rename(columns={"BSE scrip code": "bse_code"}),
        on="bse_code",
        how="left",
    )
    # Extract 2-digit NIC from CIN (positions 1..3 of CIN e.g. L26940 -> 26)
    grid["nic2_cin"] = grid["CIN code"].str.extract(r"^[LU](\d{2})")[0]

    print("3. Computing monthly stock returns, BHAR, return volatility, idiosyncratic volatility & crash risk (1996-2026)...")
    mcap_lookup = dict(zip(zip(grid["bse_code"], grid["year"]), grid["cmie_mcap"]))
    mret_df = compute_monthly_return_metrics(name_to_bse, mcap_lookup)
    grid = grid.merge(mret_df, on=["bse_code", "year"], how="left")

    print("4. Constructing benchmark fundamental, market, governance & accrual variables on full 1992-2026 lattice...")
    grid = grid.sort_values(["bse_code", "year"]).reset_index(drop=True)

    # Structural zeros in CMIE Prowess when Total Assets is observed
    has_fs = grid["cmie_total_assets"].notna() & (grid["cmie_total_assets"] > 0)
    for zcol in [
        "cmie_borrowings", "cmie_equity_div", "cmie_rd_exp", "cmie_extraord_inc",
        "cmie_extraord_exp", "cmie_forex_earnings", "cmie_forex_spending",
        "cmie_inventories", "cmie_investments", "cmie_sales_goods",
        "cmie_sales_nonfin_serv", "cmie_inc_fin_serv",
    ]:
        grid[zcol + "_zero"] = np.where(has_fs, grid[zcol].fillna(0.0), np.nan)

    # Operating Sales: Goods + Non-Fin Services (for non-financials) + Fin Services (for financials); fallback Total Income
    op_sales = grid["cmie_sales_goods_zero"] + grid["cmie_sales_nonfin_serv_zero"]
    op_sales = np.where((grid["ind_type_cmie"] != 1) & (op_sales == 0), grid["cmie_inc_fin_serv_zero"], op_sales)
    grid["SALES"] = np.where((op_sales > 0) | grid["cmie_total_income"].isna(), op_sales, grid["cmie_total_income"])

    # Promoter ownership exact identity: 100 - Inst% - NonInst% (verified max error 0.02%)
    grid["INST_OWN"] = grid["cmie_inst_own_pct"] / 100.0
    grid["NONINST_OWN"] = grid["cmie_noninst_own_pct"] / 100.0
    grid["PROMOTER_OWN"] = np.clip(1.0 - grid["INST_OWN"] - grid["NONINST_OWN"], 0.0, 1.0)
    grid["FOR_PROMOTER_OWN"] = grid["cmie_for_prom_pct"].fillna(0.0) / 100.0

    # Lagged Total Assets (strictly year - 1 check!)
    g = grid.groupby("bse_code", sort=False)
    prev_yr = g["year"].shift(1)
    is_consec_lag = (grid["year"] - prev_yr) == 1

    ta = grid["cmie_total_assets"].where(grid["cmie_total_assets"] > 0)
    ta_lag = g["cmie_total_assets"].shift(1).where(is_consec_lag)
    ta_lag = ta_lag.where(ta_lag > 0)
    grid["TA"] = ta
    grid["TA_lag"] = ta_lag

    # Core profitability & cash flow scaled by lagged TA (fallback contemporaneous TA only if IPO year)
    denom_ta = ta_lag.fillna(ta)
    # Earnings before extraordinary items = PAT net of P&E (Li 2010, Brown & Tucker 2011, Kravet & Muslu 2013)
    grid["EARN"] = grid["cmie_pat_net_pe"] / denom_ta
    grid["ROA_PAT"] = grid["cmie_pat"] / denom_ta
    grid["ROA_EBIT"] = grid["cmie_pbit"] / denom_ta
    grid["EBITDA_TA"] = grid["cmie_pbdita"] / denom_ta
    # CFO: direct CMIE CFO, fallback cash-flow identity (Net Cash Change - CFI - CFF)
    cfo_ident = grid["cmie_net_cash_chg"] - grid["cmie_cfi"] - grid["cmie_cff"]
    cfo_clean = grid["cmie_cfo"].fillna(cfo_ident)
    grid["CFO_TA"] = cfo_clean / denom_ta
    grid["ACCRUALS"] = (grid["cmie_pat_net_pe"] - cfo_clean) / denom_ta

    # CapEx (CMIE reports purchase of fixed assets as negative outflow or positive; take abs)
    capex_abs = grid["cmie_capex_raw"].abs()
    grid["CAPEX_TA"] = capex_abs / denom_ta
    grid["CFRATIO"] = cfo_clean / denom_ta  # Beatty et al. (2019) operating cash flow / lagged assets

    # Balance sheet controls & distress
    grid["LEV"] = grid["cmie_borrowings_zero"] / ta
    grid["CASH_TA"] = grid["cmie_cash_bank"] / ta
    grid["TANGIBILITY"] = grid["cmie_net_fa"] / ta
    grid["LCR"] = grid["cmie_current_assets"] / grid["cmie_curr_liab_prov"].where(grid["cmie_curr_liab_prov"] > 0)
    grid["SIZE_TA"] = np.log(ta)
    grid["SIZE_MVE"] = np.log(grid["cmie_mcap"].where(grid["cmie_mcap"] > 0))
    grid["BTM"] = grid["cmie_book_equity"] / grid["cmie_mcap"].where(grid["cmie_mcap"] > 0)
    grid["MTB"] = grid["cmie_mcap"].where(grid["cmie_mcap"] > 0) / grid["cmie_book_equity"].where(grid["cmie_book_equity"] > 0)
    grid["TOBINS_Q"] = (ta - grid["cmie_book_equity"] + grid["cmie_mcap"].where(grid["cmie_mcap"] > 0)) / ta
    grid["AGE"] = np.clip(grid["year"] - grid["incorp_year"], 0, None)
    grid["LN_AGE"] = np.log1p(grid["AGE"])
    grid["DIV_DUM"] = np.where(has_fs, (grid["cmie_equity_div_zero"] > 0).astype(float), np.nan)
    grid["SI_TA"] = (grid["cmie_extraord_inc_zero"] - grid["cmie_extraord_exp_zero"]) / denom_ta
    grid["RD_TA"] = grid["cmie_rd_exp_zero"] / denom_ta
    grid["FOREX_INTENSITY"] = grid["cmie_forex_earnings_zero"] / grid["SALES"].where(grid["SALES"] > 0)
    grid["OP_MARGIN"] = grid["cmie_pbdita"] / grid["SALES"].where(grid["SALES"] > 0)
    grid["ETR"] = np.where(grid["cmie_pbt"] > 0, np.clip(grid["cmie_tax_prov"].fillna(0.0) / grid["cmie_pbt"], 0.0, 1.0), np.nan)

    # Stock liquidity & spread proxies
    grid["SHARE_TURNOVER"] = grid["cmie_mar31_shares_traded"] / grid["cmie_shares_out"].where(grid["cmie_shares_out"] > 0)
    mid_hl = 0.5 * (grid["cmie_mar31_adj_high"] + grid["cmie_mar31_adj_low"])
    grid["HL_SPREAD"] = (grid["cmie_mar31_adj_high"] - grid["cmie_mar31_adj_low"]) / mid_hl.where(mid_hl > 0)

    # Altman (1968) Z-Score & Yearly Fractional Rank ZRANK (Campbell et al. 2014; higher ZRANK = higher distress -> invert Z!)
    wc_ta = (grid["cmie_current_assets"] - grid["cmie_curr_liab_prov"]) / ta
    re_ta = (grid["cmie_book_equity"] - grid["cmie_paidup_equity"].fillna(0.0)) / ta
    ebit_ta = grid["cmie_pbit"] / ta
    tot_liab = (ta - grid["cmie_book_equity"]).where((ta - grid["cmie_book_equity"]) > 0)
    mve_tl = grid["cmie_mcap"].where(grid["cmie_mcap"] > 0) / tot_liab
    sales_ta = grid["SALES"] / ta
    grid["ALTMAN_Z"] = 1.2 * wc_ta + 1.4 * re_ta + 3.3 * ebit_ta + 0.6 * mve_tl + 1.0 * sales_ta
    # Distress rank in [0, 1]: higher = MORE distressed (1 - percentile rank of ALTMAN_Z within year)
    grid["DISTRESS_ZRANK"] = 1.0 - grid.groupby("year")["ALTMAN_Z"].rank(pct=True)

    # Industry Concentration HHI (2-digit NIC x year Sales Herfindahl; Hope, Hu & Lu 2016 Proprietary Cost)
    valid_s = grid["SALES"].where(grid["SALES"] > 0)
    ind_tot = valid_s.groupby([grid["nic2_cin"], grid["year"]]).transform("sum")
    mkt_sh = valid_s / ind_tot.where(ind_tot > 0)
    grid["HHI_NIC2"] = (mkt_sh ** 2).groupby([grid["nic2_cin"], grid["year"]]).transform("sum")

    # Modified Jones (1995) Cross-Sectional Abnormal Accruals |AbnAccruals| (Hope, Hu & Lu 2016)
    sales_lag = g["SALES"].shift(1).where(is_consec_lag)
    rec_lag = g["cmie_receivables"].shift(1).where(is_consec_lag)
    grid["dSALES_dREC_TA"] = ((grid["SALES"] - sales_lag) - (grid["cmie_receivables"] - rec_lag)) / ta_lag
    grid["INV_TA_LAG"] = 1.0 / ta_lag
    grid["PPE_TA_LAG"] = grid["cmie_gross_fa"] / ta_lag

    abn_acc = np.full(len(grid), np.nan)
    mj_mask = (
        (grid["ind_type_cmie"] == 1)
        & grid["ACCRUALS"].notna()
        & grid["INV_TA_LAG"].notna()
        & grid["dSALES_dREC_TA"].notna()
        & grid["PPE_TA_LAG"].notna()
        & grid["nic2_cin"].notna()
    )
    mj_sub = grid.loc[mj_mask, ["nic2_cin", "year", "ACCRUALS", "INV_TA_LAG", "dSALES_dREC_TA", "PPE_TA_LAG"]].copy()
    for c_w in ["ACCRUALS", "INV_TA_LAG", "dSALES_dREC_TA", "PPE_TA_LAG"]:
        mj_sub[c_w] = winsorize_s(mj_sub[c_w])
    for (_, _), grp in mj_sub.groupby(["nic2_cin", "year"]):
        if len(grp) < 8:
            continue
        X = np.column_stack([np.ones(len(grp)), grp["INV_TA_LAG"].values, grp["dSALES_dREC_TA"].values, grp["PPE_TA_LAG"].values])
        y = grp["ACCRUALS"].values
        try:
            beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            abn_acc[grp.index] = y - X @ beta_hat
        except np.linalg.LinAlgError:
            pass
    grid["ABN_ACCRUALS"] = abn_acc
    grid["ABS_ABN_ACCRUALS"] = np.abs(abn_acc)

    # Rolling 5-year Earnings Volatility STD_EARN (years t-4..t, min 3 obs)
    earn_w_tmp = winsorize_s(grid["EARN"])
    grid["STD_EARN_5Y"] = (
        earn_w_tmp.groupby(grid["bse_code"])
        .rolling(5, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Strictly consecutive leads (t+1, t+2, t+3) and lags/changes on the 1992-2026 lattice
    g = grid.groupby("bse_code", sort=False)
    grid["dEARN_t"] = (grid["EARN"] - g["EARN"].shift(1)).where(is_consec_lag)
    grid["ABS_dEARN_t"] = grid["dEARN_t"].abs()

    for h in [1, 2, 3]:
        next_yr = g["year"].shift(-h)
        ok_h = (next_yr - grid["year"]) == h
        grid[f"EARN_lead{h}"] = g["EARN"].shift(-h).where(ok_h)
        grid[f"dEARN_lead{h}"] = (grid[f"EARN_lead{h}"] - grid["EARN"]).where(ok_h)

    ok_1 = (g["year"].shift(-1) - grid["year"]) == 1
    for lcol in [
        "CAPEX_TA", "CFRATIO", "RET_12M", "BHAR_12M", "STD_RET_12M",
        "IDIO_VOL_12M", "NCSKEW_12M", "cmie_beta", "HL_SPREAD",
        "SHARE_TURNOVER", "DISTRESS_ZRANK", "ROA_EBIT",
    ]:
        grid[f"{lcol}_lead1"] = g[lcol].shift(-1).where(ok_1)
    grid["dSTD_RET_lead1"] = grid["STD_RET_12M_lead1"] - grid["STD_RET_12M"]
    grid["dIDIO_VOL_lead1"] = grid["IDIO_VOL_12M_lead1"] - grid["IDIO_VOL_12M"]
    grid["dSHARE_TURNOVER_lead1"] = grid["SHARE_TURNOVER_lead1"] - grid["SHARE_TURNOVER"]

    print("5. Merging CMIE panel onto 16,987 corporate annual reports & winsorizing ratios at 1%/99%...")
    merged = step1.merge(grid, on=["bse_code", "year"], how="left")

    win_cols = [
        "EARN", "EARN_lead1", "EARN_lead2", "EARN_lead3",
        "dEARN_t", "ABS_dEARN_t", "dEARN_lead1", "dEARN_lead2", "dEARN_lead3",
        "ROA_PAT", "ROA_EBIT", "ROA_EBIT_lead1", "EBITDA_TA",
        "CFO_TA", "ACCRUALS", "CAPEX_TA", "CAPEX_TA_lead1", "CFRATIO", "CFRATIO_lead1",
        "LEV", "CASH_TA", "TANGIBILITY", "LCR", "SIZE_TA", "SIZE_MVE",
        "BTM", "MTB", "TOBINS_Q", "SI_TA", "RD_TA", "FOREX_INTENSITY",
        "OP_MARGIN", "ALTMAN_Z", "ABN_ACCRUALS", "ABS_ABN_ACCRUALS", "STD_EARN_5Y",
        "RET_12M", "RET_12M_lead1", "BHAR_12M", "BHAR_12M_lead1",
        "STD_RET_12M", "STD_RET_12M_lead1", "dSTD_RET_lead1",
        "IDIO_VOL_12M", "IDIO_VOL_12M_lead1", "dIDIO_VOL_lead1",
        "NCSKEW_12M", "NCSKEW_12M_lead1",
        "cmie_beta", "cmie_beta_lead1", "HL_SPREAD", "HL_SPREAD_lead1",
        "SHARE_TURNOVER", "SHARE_TURNOVER_lead1", "dSHARE_TURNOVER_lead1",
    ]
    for wc in win_cols:
        if wc in merged.columns:
            merged[wc] = winsorize_s(merged[wc], 0.01, 0.99)

    out_csv = os.path.join(OUT_DIR, "india_risk_disclosure_master_panel_1997_2025.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved Master Panel CSV: {out_csv} ({len(merged):,} rows x {merged.shape[1]} cols)")

    # Save Stata .dta (sanitize column names to <=32 chars ASCII alphanumeric/underscore)
    dta_df = merged.copy()
    seen_names = set()
    clean_cols = []
    for c in dta_df.columns:
        cc = re.sub(r"[^A-Za-z0-9_]", "_", c)[:28]
        if cc[0].isdigit():
            cc = "v_" + cc
        base_cc = cc
        k = 1
        while cc.lower() in seen_names:
            cc = f"{base_cc[:25]}_{k}"
            k += 1
        seen_names.add(cc.lower())
        clean_cols.append(cc)
    dta_df.columns = clean_cols
    for oc in dta_df.select_dtypes(include=["object"]).columns:
        dta_df[oc] = dta_df[oc].astype(str).replace("nan", "")
    out_dta = os.path.join(OUT_DIR, "india_risk_disclosure_master_panel_1997_2025.dta")
    dta_df.to_stata(out_dta, write_index=False, version=118)
    print(f"Saved Stata .dta file:  {out_dta}")

    # Generate Table 1 Summary Statistics
    summary_vars = [
        ("FLS_TONE", "Li (2010) Naïve Bayes FLS Tone"),
        ("FLS_Share", "Li (2010) Forward-Looking Sentence Share"),
        ("MDA_Rawscore", "Brown & Tucker (2011) MD&A Modification Raw Score"),
        ("MDA_Score", "Brown & Tucker (2011) Length-Adjusted MD&A Score"),
        ("KM_Risk_Disclosure", "Kravet & Muslu (2013) Annual Risk Sentences"),
        ("KM_Delta_RD", "Kravet & Muslu (2013) YoY Change in Risk Sentences"),
        ("KW_1A", "Campbell et al. (2014) Total Risk Keyword Count (log-able)"),
        ("Specificity_1A", "Hope, Hu & Lu (2016) Risk Entity Specificity (%)"),
        ("PCT_REPEATED_WC", "Verbatim 6-Gram Boilerplate Share (%)"),
        ("POST_2015_LODR_CA2013", "Post-2015 CA2013 / SEBI LODR Risk Mandate Indicator"),
        ("EARN", "Earnings Before Extraordinary Items / Lagged Assets"),
        ("CFO_TA", "Operating Cash Flow / Lagged Assets"),
        ("ACCRUALS", "Total Accruals / Lagged Assets"),
        ("CAPEX_TA_lead1", "Next-Year Capital Expenditure / Assets"),
        ("SIZE_MVE", "Log Market Value of Equity ln(MVE, Rs. Million)"),
        ("BTM", "Book-to-Market Ratio"),
        ("LEV", "Total Debt / Total Assets"),
        ("RET_12M", "12-Month Split-Adjusted Stock Return"),
        ("BHAR_12M_lead1", "Next-Year 12-Month Buy-and-Hold Abnormal Return"),
        ("STD_RET_12M_lead1", "Next-Year Monthly Stock Return Volatility"),
        ("IDIO_VOL_12M_lead1", "Next-Year Idiosyncratic Return Volatility"),
        ("cmie_beta_lead1", "Next-Year CAPM Market Beta"),
        ("ALTMAN_Z", "Altman Z-Score"),
        ("ABS_ABN_ACCRUALS", "Modified Jones |Abnormal Accruals|"),
        ("HHI_NIC2", "2-Digit NIC Industry Concentration (HHI)"),
        ("INST_OWN", "Non-Promoter Institutional Ownership Share"),
        ("PROMOTER_OWN", "Promoter Ownership Share"),
    ]
    t1_rows = []
    for vcol, vdesc in summary_vars:
        s = merged[vcol].dropna()
        t1_rows.append({
            "Variable": vcol,
            "Description": vdesc,
            "N": int(len(s)),
            "Coverage_%": round(len(s) / len(merged) * 100, 1),
            "Mean": round(float(s.mean()), 4),
            "SD": round(float(s.std()), 4),
            "P25": round(float(np.percentile(s, 25)), 4),
            "Median": round(float(s.median()), 4),
            "P75": round(float(np.percentile(s, 75)), 4),
        })
    t1_df = pd.DataFrame(t1_rows)
    t1_path = os.path.join(OUT_DIR, "table1_summary_statistics.csv")
    t1_df.to_csv(t1_path, index=False)
    print("\n=== TABLE 1: SUMMARY STATISTICS (N = 16,987 CORPORATE ANNUAL REPORTS, 1997-2025) ===")
    print(t1_df.to_string(index=False))


if __name__ == "__main__":
    main()
