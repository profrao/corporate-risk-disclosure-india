*===============================================================================
* PURE STATA MASTER EMPIRICAL SCRIPT: replicate_paper.do (100% STATA, NO PYTHON)
*===============================================================================
* Paper Title:
*   "Mandatory Risk Disclosure, Insider Governance, and the Boilerplate Paradox:
*    Evidence from Three Decades of Corporate Narratives in India"
*
* Scope:
*   Computes and exports ALL empirical manuscript tables directly inside Stata
*   (zero Python dependencies, zero external wrappers, canonical Stata `reghdfe`
*   finite-sample clustered VCE & singleton handling):
*     - Table 1B : Sample Construction & CMIE Prowess Merge Waterfall
*     - Table 1D : 29-Year (FY1997-FY2025) Year-by-Year Panel Distribution
*     - Table 1E : Panel Distribution Across Corporate Ownership Categories
*     - Table 1F : Top 15 CMIE Industry Groups in the Panel
*     - Table 3A : Full-Sample 58-Variable Summary Statistics
*     - Table 3B : Structural Evolution Across Three Disclosure Regimes
*     - Table 3C : Pre-Mandate (1997-2014) vs. Post-Mandate (2015-2025) Tests
*     - Table 4  : 14 x 14 Correlation Matrix (Pearson Lower / Spearman Upper)
*     - Table 5A : Within-Firm Structural Break After the 2014/2015 Mandate (7 cols)
*     - Table 5B : Staggered DiD: SEBI LODR Regulation 21 Cutoffs (6 cols)
*     - Table 5C : Dynamic Event Study Around FY2014 (FY2008-FY2023, Base FY2014=0)
*     - Table 5D : Staggered Event Study Around Reg. 21 RMC (t in [-5, +5])
*     - Table 6A : Determinants of FLS_TONE & Multi-Horizon Future Earnings (7 cols)
*     - Table 6B : Polarity Asymmetry, Content Topics & ML vs. Dictionary (4 cols)
*     - Table 7  : VSM MD&A Modification Determinants & Market Response (7 cols)
*     - Table 8  : Risk Sentences, Volatility, Post-AGM Window & Crash Risk (7 cols)
*     - Table 9A : Taxonomic Risk Keyword Determinants Across 6 Categories (8 cols)
*     - Table 9B : Post-Disclosure Beta, Volatility, Spread & BHAR (5 cols)
*     - Table 10 : Proprietary Costs, Promoter Share & Entity Specificity (6 cols)
*     - Table 11 : Investment-Cash Flow Sensitivity & Governance Regime (4 cols)
*     - Table 12 : Ownership Concentration, Business Groups, MNCs & PSUs (6 cols)
*     - Table 13 : Subsample, Ex-COVID, Two-Way Clustering & Post-AGM (5 cols)
*     - Table 14 : Master Hypothesis Scorecard (H1a-H5, 1-Way vs. 2-Way Clustered)
*
* How to Run in Stata (GUI or Batch CLI):
*   cd "."
*   do replicate_paper.do
*
* Pure-Stata Outputs Written to `./stata_only_outputs/`:
*   - STATA_MANUSCRIPT_TABLES_RESULTS.md : Complete Markdown report of all 22 tables
*   - STATA_REGRESSION_TABLES_WORD.rtf   : Word-ready RTF tables (Tables 5A-13)
*   - stata_complete_execution.log       : Complete native Stata console log
*   - tables_csv/<TableName>.csv         : Individual CSV files written by Stata
*===============================================================================

version 16.0
clear all
capture file close _all
set more off
set linesize 135
set varabbrev off

*-------------------------------------------------------------------------------
* 0. VERIFY REQUIRED STATA PACKAGES & RESOLVE DATASET / OUTPUT PATHS
*-------------------------------------------------------------------------------
foreach pkg in ftools reghdfe estout {
    capture which `pkg'
    if _rc != 0 & "`pkg'" == "estout" {
        capture which esttab
    }
    if _rc != 0 {
        di as text "Installing Stata package `pkg' from SSC..."
        ssc install `pkg', replace
    }
}

local dta_name "india_risk_disclosure_replication_dataset_1997_2025.dta"
local data_path ""

if "$DATA_PATH" != "" & fileexists("$DATA_PATH") {
    local data_path "$DATA_PATH"
}
else if fileexists("`dta_name'") {
    local data_path "`dta_name'"
}
else if fileexists("06_journal_replication_package/`dta_name'") {
    local data_path "06_journal_replication_package/`dta_name'"
}
else {
    di as error "ERROR: Could not locate `dta_name'. Please cd to 06_journal_replication_package first."
    exit 601
}

mata: st_local("pkg_dir", pathgetparent(st_local("data_path")))
if "`pkg_dir'" == "" local pkg_dir "."
local out_dir "`pkg_dir'/stata_only_outputs"
local csv_dir "`out_dir'/tables_csv"

capture mkdir "`out_dir'"
capture mkdir "`csv_dir'"

capture log close _stata_log
log using "`out_dir'/stata_complete_execution.log", replace text name(_stata_log)

local rtf_path "`out_dir'/STATA_REGRESSION_TABLES_WORD.rtf"
capture erase "`rtf_path'"

local md_path "`out_dir'/STATA_MANUSCRIPT_TABLES_RESULTS.md"
file open mdh using "`md_path'", write replace
file write mdh "# Complete Manuscript Empirical Results — Pure Stata Estimation (`Tables 1B–14`)" _n _n
file write mdh "- **Software:** Stata (`StataNow 19.5 SE`) — 100% pure native Stata execution (zero Python)." _n
file write mdh "- **Script:** `replicate_paper.do`" _n
file write mdh "- **Dataset:** `india_risk_disclosure_replication_dataset_1997_2025.dta` (`16,987` firm-years, `961` unique BSE firms, `FY1997–FY2025`)." _n
file write mdh "- **Estimator:** Canonical `reghdfe` multi-way fixed effects (Correia, 2015) with automatic singleton pruning and finite-sample clustered standard errors (`(G/(G-1)) * ((N-1)/(N-K))`, finite-cluster `t(G-1)` inference; Cameron, Gelbach, & Miller, 2011 PSD adjustment for two-way `firm × year` clustering)." _n _n
file write mdh "---" _n _n

*-------------------------------------------------------------------------------
* 1. PURE-STATA HELPER PROGRAMS (NATIVE `reghdfe` + DIRECT MARKDOWN/CSV WRITER)
*-------------------------------------------------------------------------------
capture program drop _tag_nfirms
program define _tag_nfirms, eclass
    syntax, nfirms(integer)
    ereturn scalar N_firms = `nfirms'
end

capture program drop run_stata_hdfe
program define run_stata_hdfe
    syntax varlist(min=2 numeric) [if], absorb(varlist) cluster(varlist) ///
        mname(string) csvhandle(string) fename(string) clname(string)

    gettoken dep indep : varlist

    quietly reghdfe `dep' `indep' `if', absorb(`absorb') vce(cluster `cluster')

    tempvar esmp ftag
    gen byte `esmp' = e(sample)
    egen byte `ftag' = tag(firm_id) if `esmp' == 1
    quietly count if `ftag' == 1
    local n_firms = r(N)
    _tag_nfirms, nfirms(`n_firms')

    local N_obs  = e(N)
    local n_sing = e(num_singletons)
    local r2_tot = e(r2)
    local r2_w   = e(r2_within)
    local F_st   = e(F)

    foreach v of local indep {
        local b  = _b[`v']
        local se = _se[`v']
        local t  = `b' / `se'
        local p  = 2 * ttail(e(df_r), abs(`t'))
        local st ""
        if `p' < 0.01      local st "***"
        else if `p' < 0.05 local st "**"
        else if `p' < 0.10 local st "*"

        file write `csvhandle' `""`mname'","`dep'","`v'",`=string(`b',"%14.6f")',`=string(`se',"%14.6f")',`=string(`t',"%10.4f")',`=string(`p',"%10.6f")',"`st'","`fename'","`clname'",`N_obs',`n_firms',`n_sing',`=string(`r2_tot',"%8.4f")',`=string(`r2_w',"%8.4f")',`=string(`F_st',"%10.2f")'"' _n
    }
end

capture program drop write_md_reg_table
program define write_md_reg_table
    syntax, mdhandle(string) title(string) models(string) vars(string) ///
        feline(string) clline(string)

    file write `mdhandle' "## `title'" _n _n

    local K : word count `models'

    * Header & alignment rows
    local hdr "| Independent Variable |"
    local aln "| :--- |"
    local dpr "| **Dependent Variable** |"
    forvalues m_i = 1/`K' {
        local m : word `m_i' of `models'
        quietly estimates restore `m'
        local col_lbl "(`m_i')"
        local dep_nm "`e(depvar)'"
        local hdr "`hdr' `col_lbl' |"
        local aln "`aln' ---: |"
        local dpr "`dpr' **`dep_nm'** |"
    }
    file write `mdhandle' "`hdr'" _n "`aln'" _n "`dpr'" _n

    * Coefficient & (t-stat) rows
    foreach v of local vars {
        local rline "| `v' |"
        forvalues m_i = 1/`K' {
            local m : word `m_i' of `models'
            quietly estimates restore `m'
            capture local b = _b[`v']
            if _rc == 0 {
                local se = _se[`v']
                local t  = `b' / `se'
                local p  = 2 * ttail(e(df_r), abs(`t'))
                local st ""
                if `p' < 0.01      local st "***"
                else if `p' < 0.05 local st "**"
                else if `p' < 0.10 local st "*"
                local cell = strtrim(string(`b', "%9.4f")) + "`st' (" + strtrim(string(`t', "%6.2f")) + ")"
                local rline "`rline' `cell' |"
            }
            else {
                local rline "`rline' — |"
            }
        }
        file write `mdhandle' "`rline'" _n
    }

    * Summary statistics footer rows
    file write `mdhandle' "| **Fixed Effects** | `feline' |" _n
    file write `mdhandle' "| **SE Clustering** | `clline' |" _n

    local n_ln   "| **Observations (N)** |"
    local f_ln   "| **Unique BSE Firms** |"
    local s_ln   "| **Singletons Dropped** |"
    local r2_ln  "| **Total R-squared** |"
    local r2w_ln "| **Within R-squared** |"
    local fst_ln "| **F-statistic** |"

    forvalues m_i = 1/`K' {
        local m : word `m_i' of `models'
        quietly estimates restore `m'
        local sn   = strtrim(string(e(N), "%12.0fc"))
        local sf   = strtrim(string(e(N_firms), "%12.0fc"))
        local ss   = strtrim(string(e(num_singletons), "%12.0fc"))
        local sr2  = strtrim(string(e(r2), "%6.4f"))
        local sr2w = strtrim(string(e(r2_within), "%6.4f"))
        local sfst = strtrim(string(e(F), "%9.2f"))

        local n_ln   "`n_ln' `sn' |"
        local f_ln   "`f_ln' `sf' |"
        local s_ln   "`s_ln' `ss' |"
        local r2_ln  "`r2_ln' `sr2' |"
        local r2w_ln "`r2w_ln' `sr2w' |"
        local fst_ln "`fst_ln' `sfst' |"
    }
    file write `mdhandle' "`n_ln'" _n "`f_ln'" _n "`s_ln'" _n "`r2_ln'" _n "`r2w_ln'" _n "`fst_ln'" _n _n
    file write `mdhandle' "---" _n _n
end

*-------------------------------------------------------------------------------
* 2. LOAD PANEL DATASET & CONSTRUCT VARIABLES IN PURE STATA
*-------------------------------------------------------------------------------
use "`data_path'", clear

replace bse_code = strtrim(bse_code)
encode bse_code, gen(firm_id)

gen double ind_num = cond(!missing(nic2_cin), nic2_cin, NIC_2digit)
egen long IND_FE = group(ind_num), missing

gen double risk_found_num = risk_found
gen double mda_found_num  = mda_found
gen double fin_note_found = (cond(missing(fin_risk_note_words), 0, fin_risk_note_words) > 0)

gen double LN_KM_RD       = ln(1 + max(KM_Risk_Disclosure, 0))     if !missing(KM_Risk_Disclosure)
gen double LN_KM_NEG_RD   = ln(1 + max(KM_Negative_RD, 0))         if !missing(KM_Negative_RD)
gen double LN_KM_OTHER_RD = ln(1 + max(KM_Other_RD, 0))            if !missing(KM_Other_RD)
gen double LN_KM_NON_RD   = ln(1 + max(KM_Non_Risk_Disclosure, 0)) if !missing(KM_Non_Risk_Disclosure)
gen double LN_KW_1A       = ln(1 + max(KW_1A_RAW, 0))              if !missing(KW_1A_RAW)

foreach c_tag in FIN LR TAX SYS IDIO INDIA_EXT {
    quietly summ KW_1A_`c_tag'
    local mult = cond(r(max) < 5.0, 1000.0, 1.0)
    gen double LN_KW_`c_tag' = ln(1 + max(KW_1A_`c_tag' * `mult', 0)) if !missing(KW_1A_`c_tag')
}
gen double LN_KW_MDA   = ln(1 + max(KW_MDA * 1000.0, 0))   if !missing(KW_MDA)
gen double LN_KW_OTHER = ln(1 + max(KW_OTHER * 1000.0, 0)) if !missing(KW_OTHER)

gen double ABS_BHAR_12M      = abs(BHAR_12M)
gen double KM_Delta_RD_100   = KM_Delta_RD / 100.0
gen double KM_Delta_Firm_100 = KM_Delta_Firm_RD / 100.0
gen double KM_Delta_Ind_100  = KM_Delta_Industry_RD / 100.0

gen double BG_AFFILIATE = ustrregexm(Ownership_group, "(?i)(group|gorup)")
gen double PSU_GOVT     = ustrregexm(Ownership_group, "(?i)(govt\.|government)")
gen double FOREIGN_MNC  = ustrregexm(Ownership_group, "(?i)foreign")

gen str40 OWN_CATEGORY = "Standalone Indian Private"
replace OWN_CATEGORY = "Indian Business Group Affiliate"  if BG_AFFILIATE == 1
replace OWN_CATEGORY = "Foreign MNC Subsidiary"           if FOREIGN_MNC  == 1
replace OWN_CATEGORY = "Government / Public Sector (PSU)" if PSU_GOVT     == 1

gen double _neg_mcap = -cmie_mcap if !missing(cmie_mcap)
bysort year: egen double MCAP_RANK_YEAR = rank(_neg_mcap), track
drop _neg_mcap

gen double SEBI_REG21_RMC = .
replace SEBI_REG21_RMC = cond( ///
    (inrange(year, 2015, 2018) & MCAP_RANK_YEAR <= 100) | ///
    (inrange(year, 2019, 2020) & MCAP_RANK_YEAR <= 500) | ///
    (year >= 2021 & MCAP_RANK_YEAR <= 1000), 1.0, 0.0) if !missing(cmie_mcap)

gen double FLS_TONE_x_POST  = FLS_TONE * POST_2015_LODR_CA2013
gen double ABS_dEARN_x_POST = ABS_dEARN_t * POST_2015_LODR_CA2013
gen double MDA_Score_x_POST = MDA_Score * POST_2015_LODR_CA2013
gen double PROM_x_POST      = PROMOTER_OWN * POST_2015_LODR_CA2013
gen double INST_x_POST      = INST_OWN * POST_2015_LODR_CA2013
gen double BG_x_POST        = BG_AFFILIATE * POST_2015_LODR_CA2013

quietly summ Specificity_10K
gen double SPEC_Z                = (Specificity_10K - r(mean)) / r(sd)
gen double CFRATIO_x_POST        = CFRATIO * POST_2015_LODR_CA2013
gen double CFRATIO_x_SPEC        = CFRATIO * SPEC_Z
gen double SPEC_x_POST           = SPEC_Z * POST_2015_LODR_CA2013
gen double CFRATIO_x_SPEC_x_POST = CFRATIO * SPEC_Z * POST_2015_LODR_CA2013
gen double CFRATIO_x_KW_NEW      = CFRATIO * KW_1A_NEW
gen double CFRATIO_x_KW_REP      = CFRATIO * KW_1A_REP

sort firm_id year
quietly count
local TOTAL_N = r(N)
egen byte _firm_tag_all = tag(firm_id)
quietly count if _firm_tag_all == 1
local TOTAL_FIRMS = r(N)
drop _firm_tag_all

*-------------------------------------------------------------------------------
* 3. SAMPLE COVERAGE & DESCRIPTIVE TABLES (TABLES 1B, 1D, 1E, 1F, 3A, 3B, 3C, 4)
*-------------------------------------------------------------------------------

* ===================== TABLE 1B =====================
file open fh1b using "`csv_dir'/Table1B_SampleCoverage.csv", write replace
file write fh1b "Stage_or_Domain,Firm_Years,Pct_of_Valid_Panel,Unique_BSE_Firms" _n
file write mdh "## Table 1B: Sample Construction & CMIE Prowess Merge Reconciliation" _n _n
file write mdh "| Stage / Data Domain | Firm-Years (`N`) | % of Panel | Unique BSE Firms |" _n
file write mdh "| :--- | ---: | ---: | ---: |" _n

forvalues s_idx = 1/11 {
    if `s_idx' == 1  {
        local slbl "Valid English Annual Report Panel (FY1997-FY2025)"
        local cond "1 == 1"
    }
    if `s_idx' == 2  {
        local slbl "Block A: MD&A Segmented (mda_found == 1)"
        local cond "mda_found > 0 & !missing(mda_found)"
    }
    if `s_idx' == 3  {
        local slbl "Block B: Substantive Risk Section (risk_found == 1)"
        local cond "risk_found > 0 & !missing(risk_found)"
    }
    if `s_idx' == 4  {
        local slbl "Block C: Directors' Report Segmented (directors_found == 1)"
        local cond "directors_found > 0 & !missing(directors_found)"
    }
    if `s_idx' == 5  {
        local slbl "Quarantined Ind-AS 107 Financial Risk Footnotes"
        local cond "fin_note_found == 1"
    }
    if `s_idx' == 6  {
        local slbl "Quarantined AGM Notice / Proxy / E-Voting Spans"
        local cond `"strtrim(notice_spans) != """'
    }
    if `s_idx' == 7  {
        local slbl "CMIE Annual Balance Sheet (Total Assets TA)"
        local cond "!missing(TA)"
    }
    if `s_idx' == 8  {
        local slbl "CMIE Annual Cash Flow Statement (Operating CFO_TA)"
        local cond "!missing(CFO_TA)"
    }
    if `s_idx' == 9  {
        local slbl "CMIE Continuous Monthly Stock Returns (RET_12M)"
        local cond "!missing(RET_12M)"
    }
    if `s_idx' == 10 {
        local slbl "CMIE Annual Market Valuation (SIZE_MVE)"
        local cond "!missing(SIZE_MVE)"
    }
    if `s_idx' == 11 {
        local slbl "CMIE Quarterly Shareholding Pattern (PROMOTER_OWN / INST_OWN)"
        local cond "!missing(INST_OWN)"
    }
    quietly count if `cond'
    local fy  = r(N)
    local pct = `fy' / `TOTAL_N' * 100.0
    tempvar ft
    egen byte `ft' = tag(firm_id) if `cond'
    quietly count if `ft' == 1
    local uf = r(N)
    drop `ft'
    file write fh1b `""`slbl'",`fy',`=string(`pct',"%6.2f")',`uf'"' _n
    file write mdh "| `slbl' | `=strtrim(string(`fy',"%12.0fc"))' | `=strtrim(string(`pct',"%6.2f"))'% | `=strtrim(string(`uf',"%9.0fc"))' |" _n
}
file close fh1b
file write mdh _n "---" _n _n

* ===================== TABLE 1D =====================
file open fh1d using "`csv_dir'/Table1D_Yearly.csv", write replace
file write fh1d "Year,N_Reports,MDA_%,RiskSec_%,KM_Sentences,KW_1A_Raw,Boilerplate_%,Specificity_1A_%,FLS_Tone,Fin_Cov_%,MktCap_Cov_%,InstOwn_Cov_%" _n
file write mdh "## Table 1D: 29-Year (`FY1997–FY2025`) Year-by-Year Panel & Disclosure Trajectory" _n _n
file write mdh "| Fiscal Year | Reports (`N`) | MD&A % | Risk Sec % | Risk Sentences | Risk KW | Boilerplate % | Specificity_1A % | FLS Tone | Assets Cov % | MktCap Cov % | InstOwn Cov % |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

forvalues yr = 1997/2025 {
    quietly count if year == `yr'
    local yn = r(N)
    quietly summ mda_found_num if year == `yr'
    local y_mda = strtrim(string(r(mean) * 100.0, "%5.1f"))
    quietly summ risk_found_num if year == `yr'
    local y_rsk = strtrim(string(r(mean) * 100.0, "%5.1f"))
    quietly summ KM_Risk_Disclosure if year == `yr'
    local y_km  = strtrim(string(r(mean), "%6.1f"))
    quietly summ KW_1A_RAW if year == `yr'
    local y_kw  = strtrim(string(r(mean), "%6.1f"))
    quietly summ PCT_REPEATED_WC if year == `yr'
    local y_bp  = cond(r(N) > 0, strtrim(string(r(mean) * 100.0, "%5.1f")), "—")
    local y_bpc = cond(r(N) > 0, strtrim(string(r(mean) * 100.0, "%5.1f")), "")
    quietly summ Specificity_1A if year == `yr'
    local y_sp  = cond(r(N) > 0, strtrim(string(r(mean) * 100.0, "%5.2f")), "—")
    local y_spc = cond(r(N) > 0, strtrim(string(r(mean) * 100.0, "%5.2f")), "")
    quietly summ FLS_TONE if year == `yr'
    local y_ft  = cond(r(N) > 0, strtrim(string(r(mean), "%6.3f")), "—")
    local y_ftc = cond(r(N) > 0, strtrim(string(r(mean), "%6.3f")), "")
    quietly count if year == `yr' & !missing(TA)
    local y_fin = strtrim(string(r(N) / `yn' * 100.0, "%5.1f"))
    quietly count if year == `yr' & !missing(SIZE_MVE)
    local y_mc  = strtrim(string(r(N) / `yn' * 100.0, "%5.1f"))
    quietly count if year == `yr' & !missing(INST_OWN)
    local y_io  = strtrim(string(r(N) / `yn' * 100.0, "%5.1f"))

    file write fh1d "`yr',`yn',`y_mda',`y_rsk',`y_km',`y_kw',`y_bpc',`y_spc',`y_ftc',`y_fin',`y_mc',`y_io'" _n
    file write mdh "| `yr' | `yn' | `y_mda' | `y_rsk' | `y_km' | `y_kw' | `y_bp' | `y_sp' | `y_ft' | `y_fin' | `y_mc' | `y_io' |" _n
}
file close fh1d
file write mdh _n "---" _n _n

* ===================== TABLE 1E =====================
file open fh1e using "`csv_dir'/Table1E_Ownership.csv", write replace
file write fh1e "Category,Firm_Years,Share_%,Unique_Firms,Mean_KM_RD,Mean_Boilerplate_%,Mean_Specificity_1A_%,Mean_Promoter_%" _n
file write mdh "## Table 1E: Sample Distribution & Disclosure Profile Across Corporate Ownership Categories" _n _n
file write mdh "| Ownership Category | Firm-Years | Panel Share % | Unique Firms | Mean Risk Sentences | Mean Boilerplate % | Mean Specificity_1A % | Mean Promoter % |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

foreach oc in "Indian Business Group Affiliate" "Foreign MNC Subsidiary" "Government / Public Sector (PSU)" "Standalone Indian Private" {
    quietly count if OWN_CATEGORY == "`oc'"
    local on  = r(N)
    local osh = strtrim(string(`on' / `TOTAL_N' * 100.0, "%6.2f"))
    tempvar oft
    egen byte `oft' = tag(firm_id) if OWN_CATEGORY == "`oc'"
    quietly count if `oft' == 1
    local ouf = r(N)
    drop `oft'
    quietly summ KM_Risk_Disclosure if OWN_CATEGORY == "`oc'"
    local okm = strtrim(string(r(mean), "%6.1f"))
    quietly summ PCT_REPEATED_WC if OWN_CATEGORY == "`oc'"
    local obp = strtrim(string(r(mean) * 100.0, "%6.2f"))
    quietly summ Specificity_1A if OWN_CATEGORY == "`oc'"
    local osp = strtrim(string(r(mean) * 100.0, "%6.2f"))
    quietly summ PROMOTER_OWN if OWN_CATEGORY == "`oc'"
    local opr = strtrim(string(r(mean) * 100.0, "%6.2f"))
    file write fh1e `""`oc'",`on',`osh',`ouf',`okm',`obp',`osp',`opr'"' _n
    file write mdh "| `oc' | `=strtrim(string(`on',"%12.0fc"))' | `osh'% | `ouf' | `okm' | `obp'% | `osp'% | `opr'% |" _n
}
file close fh1e
file write mdh _n "---" _n _n

* ===================== TABLE 1F =====================
preserve
egen byte _ig_ft = tag(Industry_group firm_id)
collapse (count) Firm_Years=year (sum) Unique_Firms=_ig_ft, by(Industry_group)
gsort -Firm_Years Industry_group
gen double Share_Pct = Firm_Years / `TOTAL_N' * 100.0
keep in 1/15
file open fh1f using "`csv_dir'/Table1F_TopIndustries.csv", write replace
file write fh1f "Industry_Group,Firm_Years,Unique_Firms,Share_%" _n
file write mdh "## Table 1F: Top 15 CMIE Industry Groups Represented in the Panel" _n _n
file write mdh "| Rank | CMIE Industry Group | Firm-Years | Unique Firms | Share of Panel % |" _n
file write mdh "| :--- | :--- | ---: | ---: | ---: |" _n
forvalues r = 1/`=_N' {
    local ig = Industry_group[`r']
    local fy = Firm_Years[`r']
    local uf = Unique_Firms[`r']
    local sh = strtrim(string(Share_Pct[`r'], "%6.2f"))
    file write fh1f `""`ig'",`fy',`uf',`sh'"' _n
    file write mdh "| `r' | `ig' | `=strtrim(string(`fy',"%9.0fc"))' | `uf' | `sh'% |" _n
}
file close fh1f
restore
file write mdh _n "---" _n _n

* ===================== TABLE 3A =====================
local desc_vars ///
    FLS_TONE FLS_TONE_Dict FLS_Share FLS_Pos_Share FLS_Neg_Share FLS_Unc_Share ///
    FLS_PROFIT_TONE FLS_LIQUIDITY_TONE MDA_Rawscore MDA_Rawscore_TF MDA_Score ///
    Risk_Rawscore KM_Risk_Disclosure KM_Negative_RD KM_Delta_RD ///
    KM_Delta_Industry_RD KM_Delta_Firm_RD KW_1A_RAW RF_IDIO RF_SYS RF_LR ///
    RF_FIN RF_TAX KW_1A_INDIA_EXT RF_FOG MDA_FOG Specificity_1A Specificity_10K ///
    PCT_REPEATED_WC KW_1A_REP KW_1A_NEW KW_1A_DISC EARN CFO_TA ACCRUALS ///
    CAPEX_TA_lead1 SIZE_MVE SIZE_TA BTM TOBINS_Q LEV CASH_TA LCR ALTMAN_Z ///
    ABS_ABN_ACCRUALS HHI_NIC2 RD_TA FOREX_INTENSITY RET_12M BHAR_12M_lead1 ///
    STD_RET_12M_lead1 IDIO_VOL_12M_lead1 NCSKEW_12M_lead1 cmie_beta_lead1 ///
    SHARE_TURNOVER HL_SPREAD INST_OWN PROMOTER_OWN

tabstat `desc_vars', statistics(count mean sd p5 p25 p50 p75 p95) columns(statistics)

file open fh3a using "`csv_dir'/Table3A_Descriptives.csv", write replace
file write fh3a "Var,N,Cov_%,Mean,SD,P5,P25,Med,P75,P95" _n
file write mdh "## Table 3A: Complete 58-Variable Summary Statistics (Stata `summarize, detail`)" _n _n
file write mdh "| Variable | `N` | Cov % | Mean | SD | P5 | P25 | Median | P75 | P95 |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n
foreach v of local desc_vars {
    quietly summ `v', detail
    local sn  = r(N)
    local cov = strtrim(string(`sn' / `TOTAL_N' * 100.0, "%5.1f"))
    local sm  = strtrim(string(r(mean), "%10.4f"))
    local ssd = strtrim(string(r(sd), "%10.4f"))
    local sp5 = strtrim(string(r(p5), "%10.4f"))
    local sp25 = strtrim(string(r(p25), "%10.4f"))
    local sp50 = strtrim(string(r(p50), "%10.4f"))
    local sp75 = strtrim(string(r(p75), "%10.4f"))
    local sp95 = strtrim(string(r(p95), "%10.4f"))
    file write fh3a "`v',`sn',`cov',`sm',`ssd',`sp5',`sp25',`sp50',`sp75',`sp95'" _n
    file write mdh "| `v' | `=strtrim(string(`sn',"%12.0fc"))' | `cov'% | `sm' | `ssd' | `sp5' | `sp25' | `sp50' | `sp75' | `sp95' |" _n
}
file close fh3a
file write mdh _n "---" _n _n

* ===================== TABLE 3B =====================
file open fh3b using "`csv_dir'/Table3B_ThreeRegimes.csv", write replace
file write fh3b "Regime,Firm_Years,MDA_%,RiskSec_%,KM_Sentences,KW_1A_Raw,Boilerplate_%,Specificity_1A_%,FLS_Tone,Fog_Index" _n
file write mdh "## Table 3B: Structural Evolution Across India's Three Statutory Disclosure Regimes (`1997–2025`)" _n _n
file write mdh "| Regulatory Regime | Firm-Years | MD&A % | Risk Sec % | Risk Sentences | Risk KW | Boilerplate % | Specificity_1A % | FLS Tone | Risk Fog |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

forvalues r_i = 1/3 {
    if `r_i' == 1 {
        local rlbl "Regime I: Pre-Clause 49 Voluntary (1997-2005)"
        local rcond "inrange(year, 1997, 2005)"
    }
    if `r_i' == 2 {
        local rlbl "Regime II: Clause 49 Mandatory MD&A (2006-2014)"
        local rcond "inrange(year, 2006, 2014)"
    }
    if `r_i' == 3 {
        local rlbl "Regime III: CA 2013 & SEBI LODR Mandate (2015-2025)"
        local rcond "inrange(year, 2015, 2025)"
    }
    quietly count if `rcond'
    local rn = r(N)
    quietly summ mda_found_num if `rcond'
    local r_mda = strtrim(string(r(mean) * 100.0, "%5.1f"))
    quietly summ risk_found_num if `rcond'
    local r_rsk = strtrim(string(r(mean) * 100.0, "%5.1f"))
    quietly summ KM_Risk_Disclosure if `rcond'
    local r_km  = strtrim(string(r(mean), "%6.1f"))
    quietly summ KW_1A_RAW if `rcond'
    local r_kw  = strtrim(string(r(mean), "%6.1f"))
    quietly summ PCT_REPEATED_WC if `rcond'
    local r_bp  = strtrim(string(r(mean) * 100.0, "%6.2f"))
    quietly summ Specificity_1A if `rcond'
    local r_sp  = strtrim(string(r(mean) * 100.0, "%6.2f"))
    quietly summ FLS_TONE if `rcond'
    local r_ft  = strtrim(string(r(mean), "%6.3f"))
    quietly summ RF_FOG if `rcond'
    local r_fg  = strtrim(string(r(mean), "%6.2f"))

    file write fh3b `""`rlbl'",`rn',`r_mda',`r_rsk',`r_km',`r_kw',`r_bp',`r_sp',`r_ft',`r_fg'"' _n
    file write mdh "| `rlbl' | `=strtrim(string(`rn',"%12.0fc"))' | `r_mda'% | `r_rsk'% | `r_km' | `r_kw' | `r_bp'% | `r_sp'% | `r_ft' | `r_fg' |" _n
}
file close fh3b
file write mdh _n "---" _n _n

* ===================== TABLE 3C =====================
local uni_vars ///
    risk_found_num mda_found_num KM_Risk_Disclosure KM_Negative_RD ///
    KW_1A_RAW PCT_REPEATED_WC KW_1A_NEW KW_1A_REP ///
    Specificity_1A Specificity_10K MDA_Rawscore FLS_TONE RF_FOG ///
    EARN LEV BTM CAPEX_TA_lead1 STD_RET_12M_lead1 INST_OWN PROMOTER_OWN

file open fh3c using "`csv_dir'/Table3C_Univariate.csv", write replace
file write fh3c "Var,Pre_N,Pre_Mean,Pre_Med,Post_N,Post_Mean,Post_Med,Diff_Mean,t_stat,Stars,Diff_Med,MW_p" _n
file write mdh "## Table 3C: Pre-Mandate (`1997–2014`) vs. Post-Mandate (`2015–2025`) Parametric & Non-Parametric Tests" _n _n
file write mdh "| Variable | Pre `N` | Pre Mean | Pre Med | Post `N` | Post Mean | Post Med | Diff Mean | Welch `t` | Diff Med | Ranksum `p` |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

foreach uv of local uni_vars {
    quietly summ `uv' if POST_2015_LODR_CA2013 == 0, detail
    local n0 = r(N)
    local m0 = r(mean)
    local d0 = r(p50)
    quietly summ `uv' if POST_2015_LODR_CA2013 == 1, detail
    local n1 = r(N)
    local m1 = r(mean)
    local d1 = r(p50)
    quietly ttest `uv', by(POST_2015_LODR_CA2013) unequal
    local tst = -r(t)
    local pv  = r(p)
    local st ""
    if `pv' < 0.01      local st "***"
    else if `pv' < 0.05 local st "**"
    else if `pv' < 0.10 local st "*"
    quietly ranksum `uv', by(POST_2015_LODR_CA2013)
    local mwp = 2 * normal(-abs(r(z)))
    local dm  = `m1' - `m0'
    local dd  = `d1' - `d0'

    local sm0  = strtrim(string(`m0', "%10.4f"))
    local sd0  = strtrim(string(`d0', "%10.4f"))
    local sm1  = strtrim(string(`m1', "%10.4f"))
    local sd1  = strtrim(string(`d1', "%10.4f"))
    local sdm  = strtrim(string(`dm', "%10.4f"))
    local stst = strtrim(string(`tst', "%8.2f"))
    local sdd  = strtrim(string(`dd', "%10.4f"))
    local smwp = strtrim(string(`mwp', "%6.4f"))

    file write fh3c "`uv',`n0',`sm0',`sd0',`n1',`sm1',`sd1',`sdm',`stst',`st',`sdd',`smwp'" _n
    file write mdh "| `uv' | `=strtrim(string(`n0',"%9.0fc"))' | `sm0' | `sd0' | `=strtrim(string(`n1',"%9.0fc"))' | `sm1' | `sd1' | `sdm'`st' | `stst' | `sdd' | `smwp' |" _n
}
file close fh3c
file write mdh _n "---" _n _n

* ===================== TABLE 4 =====================
local corr_vars "FLS_TONE MDA_Score LN_KM_RD LN_KW_1A Specificity_1A PCT_REPEATED_WC EARN CFO_TA SIZE_MVE BTM LEV STD_RET_12M INST_OWN PROMOTER_OWN"
file open fh4 using "`csv_dir'/Table4_Correlations.csv", write replace
file write fh4 "Variable,FLS_TONE,MDA_Score,LN_KM_RD,LN_KW_1A,Specificity_1A,PCT_REPEATED_WC,EARN,CFO_TA,SIZE_MVE,BTM,LEV,STD_RET_12M,INST_OWN,PROMOTER_OWN" _n
file write mdh "## Table 4: Complete `14 × 14` Correlation Matrix (Pearson Lower Triangle / Spearman Upper Triangle)" _n _n
file write mdh "| Variable | FLS_TONE | MDA_Score | LN_KM_RD | LN_KW_1A | Spec_1A | Boiler | EARN | CFO_TA | SIZE_MVE | BTM | LEV | STD_RET | INST_OWN | PROM_OWN |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

local n_cv : word count `corr_vars'
forvalues i = 1/`n_cv' {
    local vi : word `i' of `corr_vars'
    local row_csv "`vi'"
    local row_md  "| `vi' |"
    forvalues j = 1/`n_cv' {
        local vj : word `j' of `corr_vars'
        if `i' == `j' {
            local cell "1.000"
        }
        else if `i' > `j' {
            quietly corr `vi' `vj'
            local cell = strtrim(string(r(rho), "%6.3f"))
        }
        else {
            quietly spearman `vi' `vj'
            local cell = strtrim(string(r(rho), "%6.3f"))
        }
        local row_csv "`row_csv',`cell'"
        local row_md  "`row_md' `cell' |"
    }
    file write fh4 "`row_csv'" _n
    file write mdh "`row_md'" _n
}
file close fh4
file write mdh _n "---" _n _n

*-------------------------------------------------------------------------------
* 4. ECONOMETRIC REGRESSIONS & EVENT STUDIES IN EXACT MANUSCRIPT ORDER
*-------------------------------------------------------------------------------
local csv_hdr "Model,Dep_Var,Regressor,Coef,SE,t_stat,p_value,Stars,FE,Cluster,N,Firms,Singletons_Dropped,R2_Total,R2_Within,F_stat"

* ===================== TABLE 5A =====================
local c_h1 "POST_2015_LODR_CA2013 SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
file open fh using "`csv_dir'/Table5A_Mandate_WithinFirm.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t5a_1: run_stata_hdfe risk_found_num  `c_h1', absorb(firm_id) cluster(firm_id) mname("(1) Risk_Found")     csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_2: run_stata_hdfe LN_KM_RD        `c_h1', absorb(firm_id) cluster(firm_id) mname("(2) ln(RiskSent)")   csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_3: run_stata_hdfe LN_KW_1A        `c_h1', absorb(firm_id) cluster(firm_id) mname("(3) ln(KW_1A)")      csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_4: run_stata_hdfe PCT_REPEATED_WC `c_h1', absorb(firm_id) cluster(firm_id) mname("(4) Boilerplate%")   csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_5: run_stata_hdfe MDA_Rawscore    `c_h1', absorb(firm_id) cluster(firm_id) mname("(5) MDA_Rawscore")   csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_6: run_stata_hdfe Specificity_1A  `c_h1', absorb(firm_id) cluster(firm_id) mname("(6) Specificity_1A") csvhandle(fh) fename("Firm") clname("Firm")
eststo t5a_7: run_stata_hdfe RF_FOG          `c_h1', absorb(firm_id) cluster(firm_id) mname("(7) Risk_Fog")       csvhandle(fh) fename("Firm") clname("Firm")
file close fh
esttab t5a_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t5a_* using "`rtf_path'", append rtf title("Table 5A: Within-Firm Structural Break Following the 2014/2015 Indian Statutory Risk Mandate") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 5A: Within-Firm Structural Break Following the 2014/2015 Indian Statutory Risk Mandate") ///
    models("t5a_1 t5a_2 t5a_3 t5a_4 t5a_5 t5a_6 t5a_7") vars("`c_h1'") ///
    feline("Firm | Firm | Firm | Firm | Firm | Firm | Firm") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 5C =====================
preserve
local es_ctrls "SIZE_MVE BTM LEV EARN STD_RET_12M"
egen double _n0507 = count(risk_found_num) if inrange(year, 2005, 2007), by(firm_id)
bysort firm_id: egen double _valid_pre = max(_n0507)
egen double _s0507 = total(cond(inrange(year, 2005, 2007), risk_found_num, 0)), by(firm_id)
gen byte _valid_exp = (!missing(_valid_pre) & _valid_pre > 0)
gen double _UNEXP   = (_s0507 == 0) if _valid_exp == 1

local yr_vars ""
local ux_vars ""
forvalues y = 2008/2023 {
    if `y' != 2014 {
        gen double _yr_`y' = (year == `y')
        local yr_vars "`yr_vars' _yr_`y'"
    }
}
forvalues y = 2010/2023 {
    if `y' != 2014 {
        gen double _ux_`y' = _UNEXP * (year == `y') if _valid_exp == 1
        local ux_vars "`ux_vars' _ux_`y'"
    }
}
gen double _rf100 = risk_found_num * 100.0
gen double _bp100 = PCT_REPEATED_WC * 100.0
gen double _sp100 = Specificity_1A * 100.0

quietly reghdfe _rf100   `yr_vars' `es_ctrls' if inrange(year, 2008, 2023), absorb(firm_id) vce(cluster firm_id)
eststo es_rf
quietly reghdfe _rf100   `ux_vars' `es_ctrls' if inrange(year, 2010, 2023) & _valid_exp == 1, absorb(firm_id year) vce(cluster firm_id)
eststo es_exp
quietly reghdfe LN_KM_RD `yr_vars' `es_ctrls' if inrange(year, 2008, 2023), absorb(firm_id) vce(cluster firm_id)
eststo es_km
quietly reghdfe _bp100   `yr_vars' `es_ctrls' if inrange(year, 2008, 2023), absorb(firm_id) vce(cluster firm_id)
eststo es_bp
quietly reghdfe _sp100   `yr_vars' `es_ctrls' if inrange(year, 2008, 2023), absorb(firm_id) vce(cluster firm_id)
eststo es_sp

file open fh5c using "`csv_dir'/Table5C_Mandate_EventStudy.csv", write replace
file write fh5c "Year,Within_RiskFound_pp,Within_RiskFound_se,Unexposed_DiD_RiskFound_pp,Unexposed_DiD_RiskFound_se,Within_lnRiskSent_b,Within_lnRiskSent_se,Within_Boilerplate_pp,Within_Boilerplate_se,Within_Specificity1A_pp,Within_Specificity1A_se" _n
file write mdh "## Table 5C: Dynamic Year-by-Year Event-Study Estimates Around the `2014/2015` Mandate (`FY2008–FY2023`, Base `FY2014 = 0`)" _n _n
file write mdh "| Fiscal Year | Risk_Found (pp) | SE | Unexposed DiD (pp) | SE | ln(RiskSent) | SE | Boilerplate (pp) | SE | Specificity_1A (pp) | SE |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

forvalues y = 2008/2023 {
    if `y' == 2014 {
        file write fh5c "2014,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000" _n
        file write mdh "| 2014 (Base) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |" _n
    }
    else {
        quietly estimates restore es_rf
        local brf = strtrim(string(_b[_yr_`y'], "%9.4f"))
        local srf = strtrim(string(_se[_yr_`y'], "%9.4f"))
        local bux_c ""
        local sux_c ""
        local bux_m "—"
        local sux_m "—"
        if `y' >= 2010 {
            quietly estimates restore es_exp
            local bux_c = strtrim(string(_b[_ux_`y'], "%9.4f"))
            local sux_c = strtrim(string(_se[_ux_`y'], "%9.4f"))
            local bux_m "`bux_c'"
            local sux_m "`sux_c'"
        }
        quietly estimates restore es_km
        local bkm = strtrim(string(_b[_yr_`y'], "%9.4f"))
        local skm = strtrim(string(_se[_yr_`y'], "%9.4f"))
        quietly estimates restore es_bp
        local bbp = strtrim(string(_b[_yr_`y'], "%9.4f"))
        local sbp = strtrim(string(_se[_yr_`y'], "%9.4f"))
        quietly estimates restore es_sp
        local bsp = strtrim(string(_b[_yr_`y'], "%9.4f"))
        local ssp = strtrim(string(_se[_yr_`y'], "%9.4f"))

        file write fh5c "`y',`brf',`srf',`bux_c',`sux_c',`bkm',`skm',`bbp',`sbp',`bsp',`ssp'" _n
        file write mdh "| `y' | `brf' | `srf' | `bux_m' | `sux_m' | `bkm' | `skm' | `bbp' | `sbp' | `bsp' | `ssp' |" _n
    }
}
quietly estimates restore es_rf
local n5c_rf = strtrim(string(e(N), "%9.0fc"))
quietly estimates restore es_exp
local n5c_ex = strtrim(string(e(N), "%9.0fc"))
quietly estimates restore es_km
local n5c_km = strtrim(string(e(N), "%9.0fc"))
quietly estimates restore es_bp
local n5c_bp = strtrim(string(e(N), "%9.0fc"))
quietly estimates restore es_sp
local n5c_sp = strtrim(string(e(N), "%9.0fc"))
file write mdh _n "> **Sample sizes (`reghdfe`):** `Risk_Found` `N = `n5c_rf'`; `Unexposed DiD` `N = `n5c_ex'`; `ln(RiskSent)` `N = `n5c_km'`; `Boilerplate` `N = `n5c_bp'`; `Specificity_1A` `N = `n5c_sp'`." _n _n
file write mdh "---" _n _n
file close fh5c
restore

* ===================== TABLE 7 =====================
local bt_x    "ABS_dEARN_t STD_RET_12M SIZE_MVE BTM LEV INST_OWN PROMOTER_OWN"
local bt_tail "STD_RET_12M SIZE_MVE BTM LEV INST_OWN PROMOTER_OWN"
local bt_mkt  "MDA_Score MDA_Score_x_POST ABS_dEARN_t SIZE_MVE BTM LEV STD_RET_12M"

file open fh using "`csv_dir'/Table7_BrownTucker2011.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t7_1: run_stata_hdfe MDA_Rawscore   `bt_x',                              absorb(IND_FE year)  cluster(firm_id) mname("(1) MDA_Rawscore")    csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t7_2: run_stata_hdfe MDA_Score      `bt_x',                              absorb(firm_id year) cluster(firm_id) mname("(2) MDA_Score_Poly5") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t7_3: run_stata_hdfe MDA_Score_TF   `bt_x',                              absorb(firm_id year) cluster(firm_id) mname("(3) MDA_Score_TF")    csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t7_4: run_stata_hdfe Risk_Score     `bt_x',                              absorb(firm_id year) cluster(firm_id) mname("(4) Risk_Score")      csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t7_5: run_stata_hdfe MDA_Score      ABS_dEARN_t ABS_dEARN_x_POST `bt_tail', absorb(IND_FE year) cluster(firm_id) mname("(5) Shock_x_Post")  csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t7_6: run_stata_hdfe ABS_BHAR_12M   `bt_mkt',                            absorb(firm_id year) cluster(firm_id) mname("(6) |BHAR|_Resp")     csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t7_7: run_stata_hdfe SHARE_TURNOVER `bt_mkt',                            absorb(firm_id year) cluster(firm_id) mname("(7) Turnover_Resp")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
file close fh
esttab t7_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t7_* using "`rtf_path'", append rtf title("Table 7: Vector Space Model MD&A Modification Determinants & Market Reaction") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 7: Vector Space Model Narrative Modification Determinants & Equity Market Reaction") ///
    models("t7_1 t7_2 t7_3 t7_4 t7_5 t7_6 t7_7") ///
    vars("ABS_dEARN_t ABS_dEARN_x_POST MDA_Score MDA_Score_x_POST STD_RET_12M SIZE_MVE BTM LEV INST_OWN PROMOTER_OWN") ///
    feline("Industry + Year | Firm + Year | Firm + Year | Firm + Year | Industry + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 5B =====================
local c_stag "SEBI_REG21_RMC SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
file open fh using "`csv_dir'/Table5B_Staggered_SEBI_Reg21_DiD.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t5b_1: run_stata_hdfe risk_found_num  `c_stag', absorb(firm_id year) cluster(firm_id) mname("(1) Risk_Found")      csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t5b_2: run_stata_hdfe LN_KM_RD        `c_stag', absorb(firm_id year) cluster(firm_id) mname("(2) ln(RiskSent)")    csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t5b_3: run_stata_hdfe LN_KW_1A        `c_stag', absorb(firm_id year) cluster(firm_id) mname("(3) ln(KW_1A)")       csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t5b_4: run_stata_hdfe PCT_REPEATED_WC `c_stag', absorb(firm_id year) cluster(firm_id) mname("(4) Boilerplate%")    csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t5b_5: run_stata_hdfe Specificity_1A  `c_stag', absorb(firm_id year) cluster(firm_id) mname("(5) Specificity_1A")  csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t5b_6: run_stata_hdfe Specificity_10K `c_stag', absorb(firm_id year) cluster(firm_id) mname("(6) Specificity_10K") csvhandle(fh) fename("Firm + Year") clname("Firm")
file close fh
esttab t5b_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t5b_* using "`rtf_path'", append rtf title("Table 5B: Causal Staggered DiD Exploiting SEBI LODR Regulation 21 Cutoffs") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 5B: Causal Staggered Difference-in-Differences Exploiting `SEBI LODR Regulation 21` Cutoffs") ///
    models("t5b_1 t5b_2 t5b_3 t5b_4 t5b_5 t5b_6") vars("`c_stag'") ///
    feline("Firm + Year | Firm + Year | Firm + Year | Firm + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 5D =====================
preserve
egen long _first_rmc = min(cond(SEBI_REG21_RMC == 1, year, .)), by(firm_id)
keep if year >= 2009
gen long _rel_t = year - _first_rmc

gen double ev_le6 = (!missing(_rel_t) & _rel_t <= -6)
gen double ev_ge6 = (!missing(_rel_t) & _rel_t >= 6)
local ev_vars "ev_le6 ev_ge6"
forvalues k = -5/5 {
    if `k' == -1 continue
    local sfx = cond(`k' < 0, "m" + string(abs(`k')), "p" + string(`k'))
    gen double ev_`sfx' = (!missing(_rel_t) & _rel_t == `k')
    local ev_vars "`ev_vars' ev_`sfx'"
}
gen double _sp100 = Specificity_1A * 100.0
gen double _bp100 = PCT_REPEATED_WC * 100.0
local st_ctrls "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"

quietly reghdfe _sp100   `ev_vars' `st_ctrls', absorb(firm_id year) vce(cluster firm_id)
quietly test ev_m5 ev_m4 ev_m3 ev_m2
local p_pre_sp = strtrim(string(r(p), "%6.4f"))
local n_st_sp  = strtrim(string(e(N), "%9.0fc"))
eststo st_sp

quietly reghdfe _bp100   `ev_vars' `st_ctrls', absorb(firm_id year) vce(cluster firm_id)
quietly test ev_m5 ev_m4 ev_m3 ev_m2
local p_pre_bp = strtrim(string(r(p), "%6.4f"))
local n_st_bp  = strtrim(string(e(N), "%9.0fc"))
eststo st_bp

quietly reghdfe LN_KM_RD `ev_vars' `st_ctrls', absorb(firm_id year) vce(cluster firm_id)
quietly test ev_m5 ev_m4 ev_m3 ev_m2
local p_pre_km = strtrim(string(r(p), "%6.4f"))
local n_st_km  = strtrim(string(e(N), "%9.0fc"))
eststo st_km

file open fh5d using "`csv_dir'/Table5D_Staggered_SEBI_EventStudy.csv", write replace
file write fh5d "Event_Time_t,Specificity1A_pp,Specificity1A_se,Boilerplate_pp,Boilerplate_se,lnRiskSent_b,lnRiskSent_se" _n
file write mdh "## Table 5D: Staggered Event-Study Leads & Lags (`t` in `[-5, +5]`, Base `t = -1`) Around `SEBI Reg. 21` RMC" _n _n
file write mdh "| Event Time `t` | Specificity_1A (pp) | SE | Boilerplate (pp) | SE | ln(RiskSent) | SE |" _n
file write mdh "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |" _n

forvalues k = -5/5 {
    if `k' == -1 {
        file write fh5d "-1,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000" _n
        file write mdh "| -1 (Base) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |" _n
    }
    else {
        local sfx = cond(`k' < 0, "m" + string(abs(`k')), "p" + string(`k'))
        quietly estimates restore st_sp
        local bsp = strtrim(string(_b[ev_`sfx'], "%9.4f"))
        local ssp = strtrim(string(_se[ev_`sfx'], "%9.4f"))
        quietly estimates restore st_bp
        local bbp = strtrim(string(_b[ev_`sfx'], "%9.4f"))
        local sbp = strtrim(string(_se[ev_`sfx'], "%9.4f"))
        quietly estimates restore st_km
        local bkm = strtrim(string(_b[ev_`sfx'], "%9.4f"))
        local skm = strtrim(string(_se[ev_`sfx'], "%9.4f"))

        file write fh5d "`k',`bsp',`ssp',`bbp',`sbp',`bkm',`skm'" _n
        file write mdh "| `k' | `bsp' | `ssp' | `bbp' | `sbp' | `bkm' | `skm' |" _n
    }
}
file write mdh _n "> **Joint Pre-Trend F-Test p-values (`t = -5 ... -2`):** `Specificity_1A` `p = `p_pre_sp'` (`N = `n_st_sp'`); `Boilerplate` `p = `p_pre_bp'` (`N = `n_st_bp'`); `ln(RiskSent)` `p = `p_pre_km'` (`N = `n_st_km'`)." _n _n
file write mdh "---" _n _n
file close fh5d
restore

* ===================== TABLE 10 =====================
local hhl_base "HHI_NIC2 RD_TA OP_MARGIN ABS_ABN_ACCRUALS SIZE_MVE BTM LEV EARN INST_OWN PROMOTER_OWN"
local hhl_ctrl "`hhl_base' Specificity_10K RF_FOG ALL_WORDS"

file open fh using "`csv_dir'/Table10_HopeHuLu2016.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t10_1: run_stata_hdfe Specificity_1A  `hhl_base' if ind_type_cmie == 1, absorb(year)         cluster(firm_id) mname("(1) Spec1A_YearFE")   csvhandle(fh) fename("Year")            clname("Firm")
eststo t10_2: run_stata_hdfe Specificity_1A  `hhl_base' if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(2) Spec1A_FirmBase") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t10_3: run_stata_hdfe Specificity_1A  `hhl_ctrl' if ind_type_cmie == 1, absorb(IND_FE year)  cluster(firm_id) mname("(3) Spec1A_IndCtrl")  csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t10_4: run_stata_hdfe Specificity_1A  `hhl_ctrl' if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(4) Spec1A_FirmCtrl") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t10_5: run_stata_hdfe Specificity_10K `hhl_base' if ind_type_cmie == 1, absorb(IND_FE year)  cluster(firm_id) mname("(5) Spec10K_IndFE")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t10_6: run_stata_hdfe Specificity_10K `hhl_base' if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(6) Spec10K_FirmFE")  csvhandle(fh) fename("Firm + Year")     clname("Firm")
file close fh
esttab t10_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t10_* using "`rtf_path'", append rtf title("Table 10: Determinants of Risk-Section and Filing-Wide Entity Specificity") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 10: Determinants of Risk-Section (`Specificity_1A`) and Filing-Wide (`Specificity_10K`) Entity Specificity") ///
    models("t10_1 t10_2 t10_3 t10_4 t10_5 t10_6") vars("`hhl_ctrl'") ///
    feline("Year | Firm + Year | Industry + Year | Firm + Year | Industry + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 12 =====================
local gov_c "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
file open fh using "`csv_dir'/Table12_Governance_H5.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t12_1: run_stata_hdfe PCT_REPEATED_WC PROMOTER_OWN INST_OWN BG_AFFILIATE PSU_GOVT `gov_c',             absorb(IND_FE year)  cluster(firm_id) mname("(1) Boiler_Base")    csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t12_2: run_stata_hdfe PCT_REPEATED_WC PROMOTER_OWN INST_OWN BG_AFFILIATE PSU_GOVT FOREIGN_MNC `gov_c', absorb(IND_FE year)  cluster(firm_id) mname("(2) Boiler_Full")    csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t12_3: run_stata_hdfe PCT_REPEATED_WC PROMOTER_OWN PROM_x_POST INST_OWN INST_x_POST BG_x_POST `gov_c', absorb(firm_id year) cluster(firm_id) mname("(3) Boiler_FirmFE")  csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t12_4: run_stata_hdfe Specificity_1A  PROMOTER_OWN INST_OWN BG_AFFILIATE PSU_GOVT FOREIGN_MNC `gov_c', absorb(IND_FE year)  cluster(firm_id) mname("(4) Spec1A_IndFE")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t12_5: run_stata_hdfe Specificity_10K PROMOTER_OWN PROM_x_POST INST_OWN INST_x_POST BG_x_POST `gov_c', absorb(firm_id year) cluster(firm_id) mname("(5) Spec10K_FirmFE") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t12_6: run_stata_hdfe FLS_TONE        PROMOTER_OWN INST_OWN BG_AFFILIATE PSU_GOVT FOREIGN_MNC `gov_c', absorb(IND_FE year)  cluster(firm_id) mname("(6) FLS_Tone_Gov")   csvhandle(fh) fename("Industry + Year") clname("Firm")
file close fh
esttab t12_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t12_* using "`rtf_path'", append rtf title("Table 12: Ownership Concentration & Organizational Form Moderation") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 12: Ownership Concentration & Organizational Form Moderation (`H5`)") ///
    models("t12_1 t12_2 t12_3 t12_4 t12_5 t12_6") ///
    vars("PROMOTER_OWN PROM_x_POST INST_OWN INST_x_POST BG_AFFILIATE BG_x_POST PSU_GOVT FOREIGN_MNC `gov_c'") ///
    feline("Industry + Year | Industry + Year | Firm + Year | Industry + Year | Firm + Year | Industry + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 6A =====================
local li_det "EARN RET_12M ACCRUALS SIZE_MVE BTM STD_RET_12M STD_EARN_5Y LN_AGE SI_TA"
local li_c   "EARN ACCRUALS SIZE_MVE BTM RET_12M STD_RET_12M STD_EARN_5Y SI_TA"

file open fh using "`csv_dir'/Table6A_Li2010_Main.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t6a_1: run_stata_hdfe FLS_TONE    `li_det',                        absorb(IND_FE year)  cluster(firm_id) mname("(1) FLS_Tone_Det")  csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t6a_2: run_stata_hdfe EARN_lead1  FLS_TONE `li_c',                 absorb(IND_FE year)  cluster(firm_id) mname("(2) EARN_t+1_Ind")  csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t6a_3: run_stata_hdfe EARN_lead1  FLS_TONE `li_c',                 absorb(firm_id year) cluster(firm_id) mname("(3) EARN_t+1_Firm") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6a_4: run_stata_hdfe EARN_lead2  FLS_TONE `li_c',                 absorb(firm_id year) cluster(firm_id) mname("(4) EARN_t+2_Firm") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6a_5: run_stata_hdfe EARN_lead3  FLS_TONE `li_c',                 absorb(firm_id year) cluster(firm_id) mname("(5) EARN_t+3_Firm") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6a_6: run_stata_hdfe dEARN_lead1 FLS_TONE `li_c',                 absorb(firm_id year) cluster(firm_id) mname("(6) dEARN_t+1")     csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6a_7: run_stata_hdfe EARN_lead1  FLS_TONE FLS_TONE_x_POST `li_c', absorb(firm_id year) cluster(firm_id) mname("(7) Tone_x_Post")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
file close fh
esttab t6a_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t6a_* using "`rtf_path'", append rtf title("Table 6A: Determinants of FLS_TONE & Multi-Horizon Future Earnings Persistence") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 6A: Determinants of `FLS_TONE` & Multi-Horizon Future Earnings Persistence") ///
    models("t6a_1 t6a_2 t6a_3 t6a_4 t6a_5 t6a_6 t6a_7") ///
    vars("FLS_TONE FLS_TONE_x_POST EARN RET_12M ACCRUALS SIZE_MVE BTM STD_RET_12M STD_EARN_5Y LN_AGE SI_TA") ///
    feline("Industry + Year | Industry + Year | Firm + Year | Firm + Year | Firm + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 6B =====================
file open fh using "`csv_dir'/Table6B_Li2010_Decomp.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t6b_1: run_stata_hdfe EARN_lead1 FLS_Pos_Share FLS_Neg_Share FLS_Unc_Share `li_c',         absorb(IND_FE year)  cluster(firm_id) mname("(1) Polarity_Ind")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t6b_2: run_stata_hdfe EARN_lead1 FLS_Pos_Share FLS_Neg_Share FLS_Unc_Share `li_c',         absorb(firm_id year) cluster(firm_id) mname("(2) Polarity_Firm")  csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6b_3: run_stata_hdfe EARN_lead1 FLS_PROFIT_TONE FLS_LIQUIDITY_TONE FLS_OTHER_TONE `li_c', absorb(firm_id year) cluster(firm_id) mname("(3) Content_Topics") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t6b_4: run_stata_hdfe EARN_lead1 FLS_TONE FLS_TONE_Dict `li_c',                            absorb(firm_id year) cluster(firm_id) mname("(4) NB_vs_Dict")     csvhandle(fh) fename("Firm + Year")     clname("Firm")
file close fh
esttab t6b_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t6b_* using "`rtf_path'", append rtf title("Table 6B: Asymmetric Polarity, Content Topics & ML vs. Dictionary Horse-Race") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 6B: Asymmetric Polarity, Content-Topic Decomposition & ML vs. Dictionary Horse-Race") ///
    models("t6b_1 t6b_2 t6b_3 t6b_4") ///
    vars("FLS_Pos_Share FLS_Neg_Share FLS_Unc_Share FLS_PROFIT_TONE FLS_LIQUIDITY_TONE FLS_OTHER_TONE FLS_TONE FLS_TONE_Dict `li_c'") ///
    feline("Industry + Year | Firm + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm")

* ===================== TABLE 13 =====================
file open fh using "`csv_dir'/Table13_Robustness.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t13_1: run_stata_hdfe PCT_REPEATED_WC `c_h1' if ind_type_cmie == 1,        absorb(firm_id)      cluster(firm_id)      mname("(1) NonFin_Boiler")   csvhandle(fh) fename("Firm")            clname("Firm")
eststo t13_2: run_stata_hdfe PCT_REPEATED_WC `c_h1' if !inlist(year, 2020, 2021), absorb(firm_id)      cluster(firm_id)      mname("(2) ExCOVID_Boiler")  csvhandle(fh) fename("Firm")            clname("Firm")
eststo t13_3: run_stata_hdfe EARN_lead1      FLS_TONE `li_c',                     absorb(firm_id year) cluster(firm_id year) mname("(3) 2WayClust_Li")    csvhandle(fh) fename("Firm + Year")     clname("Two-Way (Firm & Year)")
local bcz_c "TOBINS_Q SIZE_TA LEV CASH_TA TANGIBILITY DIV_DUM"
eststo t13_4: run_stata_hdfe CAPEX_TA_lead1  CFRATIO CFRATIO_x_POST `bcz_c' if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id year) mname("(4) 2WayClust_ICFS") csvhandle(fh) fename("Firm + Year") clname("Two-Way (Firm & Year)")
eststo t13_5: run_stata_hdfe POST_AGM_BHAR   FLS_TONE Specificity_1A cmie_beta SIZE_MVE BTM LEV EARN, absorb(IND_FE year) cluster(firm_id) mname("(5) PostAGM_BHAR") csvhandle(fh) fename("Industry + Year") clname("Firm")
file close fh
esttab t13_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t13_* using "`rtf_path'", append rtf title("Table 13: Subsample, Ex-COVID, Two-Way Clustering & Post-AGM Robustness") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 13: Subsample, COVID-Exclusion, Two-Way Firm × Year Clustering & Post-AGM Robustness") ///
    models("t13_1 t13_2 t13_3 t13_4 t13_5") ///
    vars("POST_2015_LODR_CA2013 FLS_TONE CFRATIO CFRATIO_x_POST Specificity_1A cmie_beta SIZE_MVE SIZE_TA BTM TOBINS_Q LEV EARN ACCRUALS RET_12M STD_RET_12M STD_EARN_5Y LN_AGE SI_TA CASH_TA TANGIBILITY DIV_DUM") ///
    feline("Firm | Firm | Firm + Year | Firm + Year | Industry + Year") ///
    clline("Firm | Firm | Two-Way (Firm & Year) | Two-Way (Firm & Year) | Firm")

* ===================== TABLE 8 =====================
local km_c    "LN_KM_NON_RD SIZE_MVE BTM LEV EARN ABS_dEARN_t RET_12M INST_OWN"
local km_tail "SIZE_MVE BTM LEV EARN ABS_dEARN_t RET_12M INST_OWN"

file open fh using "`csv_dir'/Table8_KravetMuslu2013.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t8_1: run_stata_hdfe STD_RET_12M_lead1    LN_KM_RD STD_RET_12M `km_c',                               absorb(firm_id year) cluster(firm_id) mname("(1) FutVol_TotalRD")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t8_2: run_stata_hdfe STD_RET_12M_lead1    LN_KM_NEG_RD LN_KM_OTHER_RD STD_RET_12M `km_c',            absorb(firm_id year) cluster(firm_id) mname("(2) Neg_vs_OtherRD")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t8_3: run_stata_hdfe STD_RET_12M_lead1    KM_Delta_Firm_100 KM_Delta_Ind_100 STD_RET_12M `km_tail',  absorb(firm_id year) cluster(firm_id) mname("(3) FutVol_FirmVsInd") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t8_4: run_stata_hdfe dSTD_RET_lead1       KM_Delta_Firm_100 KM_Delta_Ind_100 `km_tail',              absorb(IND_FE year)  cluster(firm_id) mname("(4) dVol_FirmVsInd")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t8_5: run_stata_hdfe POST_AGM_STD_RET     KM_Delta_Firm_100 KM_Delta_Ind_100 STD_RET_12M `km_tail',  absorb(firm_id year) cluster(firm_id) mname("(5) PostAGM_OctSep")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t8_6: run_stata_hdfe SHARE_TURNOVER_lead1 KM_Delta_Firm_100 KM_Delta_Ind_100 SHARE_TURNOVER `km_tail', absorb(firm_id year) cluster(firm_id) mname("(6) Fut_Turnover")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t8_7: run_stata_hdfe NCSKEW_12M_lead1     LN_KM_NEG_RD LN_KM_OTHER_RD NCSKEW_12M STD_RET_12M `km_c', absorb(firm_id year) cluster(firm_id) mname("(7) Crash_NCSKEW")     csvhandle(fh) fename("Firm + Year")     clname("Firm")
file close fh
esttab t8_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t8_* using "`rtf_path'", append rtf title("Table 8: Risk Sentence Levels, Modifiers, Peer Innovations, Volatility & Crash Risk") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 8: Risk Sentence Levels, Negative Modifiers, Firm vs. Peer Innovations, Volatility & Crash Skewness") ///
    models("t8_1 t8_2 t8_3 t8_4 t8_5 t8_6 t8_7") ///
    vars("LN_KM_RD LN_KM_NEG_RD LN_KM_OTHER_RD KM_Delta_Firm_100 KM_Delta_Ind_100 STD_RET_12M SHARE_TURNOVER NCSKEW_12M `km_c'") ///
    feline("Firm + Year | Firm + Year | Firm + Year | Industry + Year | Firm + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 9A =====================
local camp_base "LEV DISTRESS_ZRANK LCR cmie_beta IDIO_VOL_12M SIZE_MVE BTM EARN RET_12M"
local camp_det  "`camp_base' LN_KW_MDA LN_KW_OTHER"

file open fh using "`csv_dir'/Table9A_Campbell_Determinants.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t9a_1: run_stata_hdfe LN_KW_1A        `camp_base', absorb(IND_FE year) cluster(firm_id) mname("(1) All_KW_Base")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_2: run_stata_hdfe LN_KW_1A        `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(2) All_KW_Full")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_3: run_stata_hdfe LN_KW_FIN       `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(3) Financial_KW")  csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_4: run_stata_hdfe LN_KW_LR        `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(4) LegalReg_KW")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_5: run_stata_hdfe LN_KW_TAX       `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(5) Tax_KW")        csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_6: run_stata_hdfe LN_KW_SYS       `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(6) Systematic_KW") csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_7: run_stata_hdfe LN_KW_IDIO      `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(7) Idiosync_KW")   csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9a_8: run_stata_hdfe LN_KW_INDIA_EXT `camp_det',  absorb(IND_FE year) cluster(firm_id) mname("(8) India_Inst_KW") csvhandle(fh) fename("Industry + Year") clname("Firm")
file close fh
esttab t9a_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t9a_* using "`rtf_path'", append rtf title("Table 9A: Pre-Disclosure Fundamental & Risk Determinants Across Risk Subcategories") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 9A: Pre-Disclosure Fundamental & Risk Determinants Across Total `ln(1+KW_1A)` and All 6 Subcategories") ///
    models("t9a_1 t9a_2 t9a_3 t9a_4 t9a_5 t9a_6 t9a_7 t9a_8") vars("`camp_det'") ///
    feline("Industry + Year | Industry + Year | Industry + Year | Industry + Year | Industry + Year | Industry + Year | Industry + Year | Industry + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 9B =====================
local cc2 "LN_KW_MDA LN_KW_OTHER SIZE_MVE BTM LEV EARN RET_12M"
file open fh using "`csv_dir'/Table9B_Campbell_Consequences.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t9b_1: run_stata_hdfe cmie_beta_lead1    LN_KW_SYS LN_KW_IDIO cmie_beta `cc2',                   absorb(IND_FE year)  cluster(firm_id) mname("(1) Beta_t+1")      csvhandle(fh) fename("Industry + Year") clname("Firm")
eststo t9b_2: run_stata_hdfe IDIO_VOL_12M_lead1 LN_KW_IDIO LN_KW_SYS Specificity_1A IDIO_VOL_12M `cc2', absorb(firm_id year) cluster(firm_id) mname("(2) IdioVol_t+1")   csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t9b_3: run_stata_hdfe IDIO_VOL_12M_lead1 KW_1A_NEW KW_1A_REP IDIO_VOL_12M `cc2',                 absorb(firm_id year) cluster(firm_id) mname("(3) NewVsRep_Vol")  csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t9b_4: run_stata_hdfe HL_SPREAD_lead1    LN_KW_1A Specificity_1A HL_SPREAD `cc2',                absorb(firm_id year) cluster(firm_id) mname("(4) HL_Spread_t+1") csvhandle(fh) fename("Firm + Year")     clname("Firm")
eststo t9b_5: run_stata_hdfe BHAR_12M_lead1     LN_KW_1A Specificity_1A cmie_beta `cc2',                absorb(IND_FE year)  cluster(firm_id) mname("(5) BHAR_t+1")      csvhandle(fh) fename("Industry + Year") clname("Firm")
file close fh
esttab t9b_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t9b_* using "`rtf_path'", append rtf title("Table 9B: Post-Disclosure Market Consequences: Beta, Volatility, Spread & BHAR") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 9B: Post-Disclosure Market Consequences: Systematic Beta, Idiosyncratic Volatility, Spread & BHAR") ///
    models("t9b_1 t9b_2 t9b_3 t9b_4 t9b_5") ///
    vars("LN_KW_SYS LN_KW_IDIO LN_KW_1A KW_1A_NEW KW_1A_REP Specificity_1A cmie_beta IDIO_VOL_12M HL_SPREAD `cc2'") ///
    feline("Industry + Year | Firm + Year | Firm + Year | Firm + Year | Industry + Year") ///
    clline("Firm | Firm | Firm | Firm | Firm")

* ===================== TABLE 11 =====================
file open fh using "`csv_dir'/Table11_Beatty2019_ICFS.csv", write replace
file write fh "`csv_hdr'" _n
eststo clear
eststo t11_1: run_stata_hdfe CAPEX_TA_lead1 CFRATIO `bcz_c'                                                                        if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(1) Baseline_ICFS") csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t11_2: run_stata_hdfe CAPEX_TA_lead1 CFRATIO CFRATIO_x_POST `bcz_c'                                                         if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(2) Mandate_Shift") csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t11_3: run_stata_hdfe CAPEX_TA_lead1 CFRATIO CFRATIO_x_SPEC CFRATIO_x_POST CFRATIO_x_SPEC_x_POST SPEC_Z SPEC_x_POST `bcz_c' if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(3) Triple_Spec")   csvhandle(fh) fename("Firm + Year") clname("Firm")
eststo t11_4: run_stata_hdfe CAPEX_TA_lead1 CFRATIO CFRATIO_x_KW_NEW CFRATIO_x_KW_REP KW_1A_NEW KW_1A_REP `bcz_c'                  if ind_type_cmie == 1, absorb(firm_id year) cluster(firm_id) mname("(4) NewVsRep_KW")   csvhandle(fh) fename("Firm + Year") clname("Firm")
file close fh
esttab t11_*, b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms num_singletons r2 r2_within F, fmt(%9.0fc %9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
esttab t11_* using "`rtf_path'", append rtf title("Table 11: Next-Year Capital Expenditure, Cash Flow & Post-2015 Governance Regime") b(%9.4f) t(%6.2f) star(* 0.10 ** 0.05 *** 0.01) drop(_cons) stats(N N_firms r2 r2_within F, fmt(%9.0fc %9.0fc %6.4f %6.4f %8.2f)) compress nogaps
write_md_reg_table, mdhandle(mdh) title("Table 11: Next-Year Capital Expenditure (`CAPEX_TA_{t+1}`), Cash Flow (`CFRATIO`) & Post-2015 Governance Regime") ///
    models("t11_1 t11_2 t11_3 t11_4") ///
    vars("CFRATIO CFRATIO_x_POST CFRATIO_x_SPEC CFRATIO_x_SPEC_x_POST SPEC_Z SPEC_x_POST CFRATIO_x_KW_NEW CFRATIO_x_KW_REP KW_1A_NEW KW_1A_REP `bcz_c'") ///
    feline("Firm + Year | Firm + Year | Firm + Year | Firm + Year") ///
    clline("Firm | Firm | Firm | Firm")

* ===================== TABLE 14 =====================
file open fh14 using "`csv_dir'/Table14_MasterScorecard.csv", write replace
file write fh14 "Specification,Dep_Var,Target_Var,Raw_Coef,Std_Effect_SD,se1_std,se2_std,t_FirmCluster,t_TwoWayCluster,p_Firm,p_TwoWay,N" _n
file write mdh "## Table 14: Master Empirical Hypothesis Scorecard (`H1a–H5` — Pure Stata 1-Way vs. 2-Way Clustered)" _n _n
file write mdh "| Specification | Outcome | Predictor | Raw Coef | Std Effect (SD) | `t` (Firm) | `t` (Firm×Year) | `p` (Firm) | `p` (2-Way) | `N` |" _n
file write mdh "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |" _n

forvalues s_i = 1/15 {
    local subcond ""
    if `s_i' == 1  {
        local slbl "H1a (Table 5A.2): 2015 Mandate -> ln(Risk Sentences)"
        local dep "LN_KM_RD"
        local tgt "POST_2015_LODR_CA2013"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id"
        local isb = 1
    }
    if `s_i' == 2  {
        local slbl "H1b (Table 5A.4): 2015 Mandate -> 6-Gram Boilerplate"
        local dep "PCT_REPEATED_WC"
        local tgt "POST_2015_LODR_CA2013"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id"
        local isb = 1
    }
    if `s_i' == 3  {
        local slbl "H1b (Table 5A.7): 2015 Mandate -> Gunning Fog Index"
        local dep "RF_FOG"
        local tgt "POST_2015_LODR_CA2013"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id"
        local isb = 1
    }
    if `s_i' == 4  {
        local slbl "H1b (Table 5A.6): 2015 Mandate -> Entity Specificity_1A"
        local dep "Specificity_1A"
        local tgt "POST_2015_LODR_CA2013"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id"
        local isb = 1
    }
    if `s_i' == 5  {
        local slbl "Reg. 21 DiD (Table 5B.5): Board RMC -> Specificity_1A"
        local dep "Specificity_1A"
        local tgt "SEBI_REG21_RMC"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id year"
        local isb = 1
    }
    if `s_i' == 6  {
        local slbl "Reg. 21 DiD (Table 5B.2): Board RMC -> ln(Risk Sentences)"
        local dep "LN_KM_RD"
        local tgt "SEBI_REG21_RMC"
        local ctr "SIZE_MVE BTM LEV EARN STD_RET_12M LN_AGE"
        local fes "firm_id year"
        local isb = 1
    }
    if `s_i' == 7  {
        local slbl "H2a (Table 6A.3): NB FLS Tone -> Future EARN_{t+1}"
        local dep "EARN_lead1"
        local tgt "FLS_TONE"
        local ctr "EARN ACCRUALS SIZE_MVE BTM RET_12M STD_RET_12M STD_EARN_5Y SI_TA"
        local fes "firm_id year"
        local isb = 0
    }
    if `s_i' == 8  {
        local slbl "H2b (Table 7.2): Abs(dEARN) Shock -> MD&A Modification"
        local dep "MDA_Score"
        local tgt "ABS_dEARN_t"
        local ctr "STD_RET_12M SIZE_MVE BTM LEV INST_OWN PROMOTER_OWN"
        local fes "firm_id year"
        local isb = 0
    }
    if `s_i' == 9  {
        local slbl "H2b (Table 7.5): Abs(dEARN) x Post-2015 -> MD&A Score"
        local dep "MDA_Score"
        local tgt "ABS_dEARN_x_POST"
        local ctr "ABS_dEARN_t STD_RET_12M SIZE_MVE BTM LEV INST_OWN PROMOTER_OWN"
        local fes "IND_FE year"
        local isb = 0
    }
    if `s_i' == 10 {
        local slbl "H3 (Table 8.4): Peer Industry Delta_RD -> dVol_{t+1}"
        local dep "dSTD_RET_lead1"
        local tgt "KM_Delta_Ind_100"
        local ctr "KM_Delta_Firm_100 SIZE_MVE BTM LEV EARN ABS_dEARN_t RET_12M INST_OWN"
        local fes "IND_FE year"
        local isb = 0
    }
    if `s_i' == 11 {
        local slbl "H3 (Table 9A.3): Distress Rank -> Financial Risk KW"
        local dep "LN_KW_FIN"
        local tgt "DISTRESS_ZRANK"
        local ctr "LEV LCR cmie_beta IDIO_VOL_12M SIZE_MVE BTM EARN RET_12M LN_KW_MDA LN_KW_OTHER"
        local fes "IND_FE year"
        local isb = 0
    }
    if `s_i' == 12 {
        local slbl "H3 (Table 9B.3): Repeated Risk KW -> IdioVol_{t+1}"
        local dep "IDIO_VOL_12M_lead1"
        local tgt "KW_1A_REP"
        local ctr "KW_1A_NEW IDIO_VOL_12M LN_KW_MDA LN_KW_OTHER SIZE_MVE BTM LEV EARN RET_12M"
        local fes "firm_id year"
        local isb = 0
    }
    if `s_i' == 13 {
        local slbl "H4a (Table 10.5): R&D Proprietary Cost -> Spec_10K"
        local dep "Specificity_10K"
        local tgt "RD_TA"
        local ctr "HHI_NIC2 OP_MARGIN ABS_ABN_ACCRUALS SIZE_MVE BTM LEV EARN INST_OWN PROMOTER_OWN"
        local fes "IND_FE year"
        local isb = 0
        local subcond "if ind_type_cmie == 1"
    }
    if `s_i' == 14 {
        local slbl "H5 (Table 10.2): Promoter Share -> Specificity_1A"
        local dep "Specificity_1A"
        local tgt "PROMOTER_OWN"
        local ctr "HHI_NIC2 OP_MARGIN ABS_ABN_ACCRUALS SIZE_MVE BTM LEV EARN INST_OWN RD_TA"
        local fes "firm_id year"
        local isb = 0
        local subcond "if ind_type_cmie == 1"
    }
    if `s_i' == 15 {
        local slbl "H4b (Table 11.2): Post-2015 ICFS Wedge Reduction"
        local dep "CAPEX_TA_lead1"
        local tgt "CFRATIO_x_POST"
        local ctr "CFRATIO TOBINS_Q SIZE_TA LEV CASH_TA TANGIBILITY DIV_DUM"
        local fes "firm_id year"
        local isb = 0
        local subcond "if ind_type_cmie == 1"
    }

    quietly reghdfe `dep' `tgt' `ctr' `subcond', absorb(`fes') vce(cluster firm_id)
    local braw = _b[`tgt']
    local se1  = _se[`tgt']
    local t1   = `braw' / `se1'
    local p1   = 2 * ttail(e(df_r), abs(`t1'))
    local n14  = e(N)

    tempvar es14
    gen byte `es14' = e(sample)
    quietly summ `dep' if `es14' == 1
    local sdy = r(sd)
    if `isb' == 1 {
        local sdx = 1.0
    }
    else {
        quietly summ `tgt' if `es14' == 1
        local sdx = r(sd)
    }
    drop `es14'

    quietly reghdfe `dep' `tgt' `ctr' `subcond', absorb(`fes') vce(cluster firm_id year)
    local se2 = _se[`tgt']
    local t2  = `braw' / `se2'
    local p2  = 2 * ttail(e(df_r), abs(`t2'))

    local scf  = `sdx' / `sdy'
    local bstd = `braw' * `scf'
    local s1s  = `se1' * abs(`scf')
    local s2s  = `se2' * abs(`scf')

    local sbraw = strtrim(string(`braw', "%9.4f"))
    local sbstd = strtrim(string(`bstd', "%9.4f"))
    local ss1s  = strtrim(string(`s1s', "%9.4f"))
    local ss2s  = strtrim(string(`s2s', "%9.4f"))
    local st1   = strtrim(string(`t1', "%7.2f"))
    local st2   = strtrim(string(`t2', "%7.2f"))
    local sp1   = strtrim(string(`p1', "%7.4f"))
    local sp2   = strtrim(string(`p2', "%7.4f"))

    file write fh14 `""`slbl'","`dep'","`tgt'",`sbraw',`sbstd',`ss1s',`ss2s',`st1',`st2',`sp1',`sp2',`n14'"' _n
    file write mdh "| `slbl' | `dep' | `tgt' | `sbraw' | `sbstd' | `st1' | `st2' | `sp1' | `sp2' | `=strtrim(string(`n14',"%9.0fc"))' |" _n
}
file close fh14
file close mdh

xtset firm_id year

di as result "=============================================================================="
di as result " PURE STATA EXECUTION COMPLETE (ZERO PYTHON)!"
di as result "   Markdown Report : `md_path'"
di as result "   Word RTF Tables : `rtf_path'"
di as result "   CSV Table Suite : `csv_dir'"
di as result "=============================================================================="

log close _stata_log
