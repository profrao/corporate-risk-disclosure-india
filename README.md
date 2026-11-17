# Mandatory Risk Disclosure, Insider Governance, and the Boilerplate Paradox: Replication Code & Textual Data (India, 1997–2025)

This repository provides the complete natural-language processing (NLP) pipeline, econometric replication scripts (Python and Stata), verified domain lexicons, labeled forward-looking training sentences, and firm-year textual disclosure dataset for **16,987 English corporate annual reports across 961 Bombay Stock Exchange (BSE)-listed Indian firms (`FY1997–FY2025`)**.

The empirical pipeline implements and extends six foundational textual accounting and finance benchmarks:
1. **Li (2010)** — *Journal of Accounting Research* 48(5), 1049–1102 (Forward-Looking Statements & Multinomial Naïve Bayes Tone)
2. **Brown and Tucker (2011)** — *Journal of Accounting Research* 49(2), 309–346 (Vector-Space MD&A Modification)
3. **Kravet and Muslu (2013)** — *Review of Accounting Studies* 18(4), 1088–1122 (Filing-Wide Risk Sentence Innovations)
4. **Campbell, Chen, Dhaliwal, Lu, and Steele (2014)** — *Review of Accounting Studies* 19(1), 396–455 (382-Term 5-Category Risk Taxonomy & Readability)
5. **Hope, Hu, and Lu (2016)** — *Review of Accounting Studies* 21(4), 1005–1045 (7-Category Named-Entity Specificity)
6. **Beatty, Cheng, and Zhang (2019)** — *Contemporary Accounting Research* 36(2), 805–838 (WCopyfind 6-Gram Verbatim Boilerplate)

---

## Repository Structure

```text
├── README.md                                                    # Repository overview & replication instructions
├── requirements.txt                                             # Python package dependencies
├── LICENSE                                                      # MIT License (code & scripts)
├── code/
│   ├── step0_segment_reports.py                                 # Step 0 statutory section parser & Ind-AS 107 / AGM quarantine
│   ├── step1_lexicons.json                                      # Campbell (382 terms), Kravet-Muslu, Li FLS, & LM lexicons
│   ├── train_li2010_naive_bayes.py                              # Multinomial Naïve Bayes FLS classifier trainer & N-fold CV
│   ├── step1_compute_paper_metrics.py                           # Step 1 143-column textual disclosure panel builder
│   ├── audit_verify_step1.py                                    # 11-point algebraic & methodological verification suite
│   ├── step2_build_merged_panel.py                              # Step 2 CMIE Prowess 35-year lattice builder & panel merger
│   ├── replicate_paper.do                                       # Standalone Stata 16+ master econometric replication script
│   └── render_stata_figures.py                                  # Vector/350-DPI figure renderer from Stata CSV outputs
├── data/
│   ├── india_corporate_risk_disclosure_panel_1997_2025.csv      # 16,987 × 143 textual variables extracted from BSE filings
│   ├── li2010_training_fls_labeled.jsonl                        # 2,400 labeled Indian MD&A forward-looking training sentences
│   ├── li2010_nb_model.json                                     # Trained Multinomial Naïve Bayes log-probability parameters
│   ├── cmie_bse_codes_961.txt                                   # List of 961 BSE scrip codes in the panel
│   └── audit_verification_summary.json                          # Step 1 verification audit output
├── docs/
│   └── VARIABLE_CODEBOOK.md                                     # Complete textual & econometric variable definitions
└── figures/
    └── figure1.png ... figure6.png                              # Publication Figures 1–6 (350 DPI)
```

---

## Data Availability & Replication Workflow

1. **Public Corporate Filing & Textual Data (Included in `data/`)**:
   Corporate annual reports are publicly available statutory filings from BSE India (`https://www.bseindia.com`). All **143 document-level and section-level textual measures** (`16,987` firm-years), the **2,400 labeled Indian MD&A forward-looking sentences** (`li2010_training_fls_labeled.jsonl`), and the trained **Multinomial Naïve Bayes parameters** (`li2010_nb_model.json`) are provided directly in `data/`.

2. **Accounting, Stock-Return, and Ownership Data (`CMIE ProwessIQ`)**:
   Financial statement fundamentals, adjusted monthly stock prices (`1995–2026`), market capitalizations, and quarterly promoter/institutional shareholding data are obtained by subscription from the **Centre for Monitoring Indian Economy (CMIE) ProwessIQ** (`https://prowessiq.cmie.com`), whose license terms restrict public redistribution of raw financial series:
   - Query the 961 BSE scrip codes in `data/cmie_bse_codes_961.txt` over `1992–2026` in CMIE ProwessIQ and place exports in `05_cmie_data/raw_exports/`.
   - Run `python code/step2_build_merged_panel.py` to construct the merged firm-year dataset (`.csv` and `.dta`).
   - Run `do code/replicate_paper.do` in **Stata 16+** (requires `ftools`, `reghdfe`, and `estout` from SSC) to reproduce all summary statistics, univariate regime tests, event studies, and all 72 high-dimensional fixed-effects regressions.
