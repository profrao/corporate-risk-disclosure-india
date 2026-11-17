# Variable Codebook: Audited Indian Corporate Risk Disclosure & Textual Analytics Panel (1997–2025, v2)

- **Dataset File**: `india_corporate_risk_disclosure_panel_1997_2025.csv`
- **Unit of Observation**: Firm-Year (`bse_code` × `year`)
- **Total Valid English Reports**: `16,987` firm-years across `961` BSE-listed companies (`1997–2025`), segmented from `17,651` raw annual report Markdown files (`388` non-English/broken-font reports and `276` empty files excluded in Step 0; partial section-level PDF `/ToUnicode` CMap font corruptions flagged via `*_cmap_corrupted` in Step 1).
- **Auxiliary Files**:
  - `../step0_sections/section_manifest.csv` — Audited Step 0 line boundaries (`mda_start_line`, `mda_end_line`, `risk_spans`, `notice_spans`, `directors_start_line`).
  - `../step1_lexicons.json` — Primary-PDF-verified lexicons (`382` Campbell et al. 2014 keywords + `30` case-sensitive acronyms; `18` Kravet & Muslu 2013 Table A1 rules + Section 4.6 negative modifiers; `17` Li 2010 Appendix B FLS roots + `10` legal-boilerplate exclusions + past-tense retrospection regex).
  - `li2010_nb_model.json` & `LI2010_VALIDATION_REPORT.md` — Porter-stemmed, negation-preserving Multinomial Naïve Bayesian classifier trained on `2,400` stratified Indian MD&A FLS sentences labeled on an NVIDIA H200 GPU (`qwen2.5:72b`) and cross-validated across $N = 3, 5, 10, 25$ folds.
  - `audit_verification_summary.json` — Automated mathematical & empirical audit verification log.

---

## 1. Identifiers, Metadata, Disjoint Partitions & Font-Encoding QC Flags

| Variable Name | Description | Source |
| :--- | :--- | :--- |
| `bse_code` | BSE Scrip Code (6-digit identifier matching folder name in `annual_reports/`) | BSE / CMIE |
| `year` / `year_int` | Report fiscal year (`1997–2025`) | Report filename |
| `Company Name` | Official company name in CMIE Prowess | `cmie_identity_indicators.csv` |
| `Prowess company code` | CMIE Prowess unique numeric company identifier | `cmie_identity_indicators.csv` |
| `CIN code` / `ISIN code` | Corporate Identity Number (MCA) and ISIN | `cmie_identity_indicators.csv` |
| `NSE symbol` / `BSE scrip id` | Stock exchange trading ticker symbols | `cmie_identity_indicators.csv` |
| `NIC code` / `NIC_2digit` / `NIC name` | National Industrial Classification code (full & 2-digit industry code) | `cmie_identity_indicators.csv` |
| `Industry group`, `Ownership group`, `Size decile` | CMIE Prowess industry, ownership, and size classifications | `cmie_identity_indicators.csv` |
| `mda_found`, `mda_start_line`, `mda_end_line` | Indicator and 1-based line span for **Block A (`MDA_Full`)** (including graphic-banner MD&As recovered via canonical Cautionary Statement anchors) | Step 0 v2 |
| `risk_found`, `risk_spans`, `risk_headings` | Indicator, exact `start-end;...` line coordinates, and matched headings for **Block B (`Risk_Section`)** (excluding Ind-AS 107 financial notes and Risk Management Committee attendance tables) | Step 0 v2 |
| `directors_found`, `directors_start_line`, `directors_end_line` | Indicator and line span for **Block C (`Directors_Report`)** | Step 0 v2 |
| `notice_spans` | Exact line spans of **AGM Notice / Proxy / E-voting Instructions** (excluded from narrative risk & whole-report sentence counts) | Step 0 v2 |
| `fin_risk_note_words` | Word count of Ind-AS 107 *Financial Risk Management* footnote inside Financial Statements (tracked separately from narrative risk) | Step 0 v2 |
| `mda_narrative_words`, `mda_ex_risk_words`, `risk_narrative_words`, `other_disjoint_words` | Clean narrative word counts across the **mutually disjoint** report partition (`Risk_Section`, `MDA \ Risk`, and `Other \ (Risk ∪ MDA ∪ Notice)`) | Step 1 v2 |
| `mda_cmap_corrupted`, `risk_cmap_corrupted`, `report_cmap_corrupted` | Binary QC flags (`1/0`) indicating sections where $>25\%$ of words suffered PDF `/ToUnicode` CMap font shifts (e.g., Caesar-shifted consonant soup) or $<18\%$ English stopwords; corrupted section metrics are set to `NaN` rather than false zeros | Step 1 v2 |

---

## 2. Brown & Tucker (2011, *JAR*) — Porter-Stemmed VSM MD&A & Risk Modification Scores

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `MDA_Rawscore` | **Primary specification**: $1 - \cos(\theta_{i,t})$ between consecutive years' (`year_t - year_{t-1} == 1`) **Porter (1980)-stemmed** MD&A vectors with **no stopword removal** and pure corpus-wide inverse document frequency $\text{IDF}_k = \ln(M / m_k)$ (so words appearing in all $M$ MD&As receive $\ln(1)=0$). | Brown & Tucker (2011), Sec. 3.2, pp. 315–317 & Footnotes 9–10 |
| `MDA_Score` | Residual from regressing `MDA_Rawscore` on a 5th-order Taylor polynomial of current-year MD&A narrative word count ($\text{Length}_{i,t}^1 \dots \text{Length}_{i,t}^5$). Mean equals `0.0000` by construction. | Brown & Tucker (2011), Appendix B, Table B1 (p. 344) |
| `MDA_Rawscore_CohortIDF` | Same as `MDA_Rawscore`, computing $\text{IDF}_{k,t} = \ln(M_t / m_{k,t})$ within fiscal-year cohort $t$ (when $M_t \ge 20$). | Brown & Tucker (2011), p. 317 |
| `MDA_Rawscore_TF` / `MDA_Score_TF` | **Footnote 10 alternative specification**: Porter-stemmed cosine dissimilarity **without IDF** after removing English stopwords (`nltk`), plus its 5th-order polynomial length-adjusted residual `MDA_Score_TF`. | Brown & Tucker (2011), Footnote 10 (p. 317) |
| `Risk_Rawscore` / `Risk_Score` | Porter-stemmed TF-IDF cosine dissimilarity and 5th-order polynomial length-adjusted score applied to consecutive years' `Risk_Section`. | Brown & Tucker (2011) applied to Risk section |

---

## 3. Campbell, Chen, Dhaliwal, Lu & Steele (2014, *RAST*) — 382-Keyword 5-Category Risk Taxonomy & Fog

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `ALL_WORDS` / `KEY_WORDS` | $\ln(1 + \text{risk\_narrative\_words})$ and $\ln(1 + \text{KW\_1A\_RAW})$ in `Risk_Section`. | Campbell et al. (2014), Sec. 3.3 & Appendix A |
| `KW_1A_RAW`, `KW_1A_FIN`, `KW_1A_LR`, `KW_1A_TAX`, `KW_1A_SYS`, `KW_1A_IDIO` | Exact non-overlapping longest-phrase-first counts across all **382 Appendix C keywords** (`66` Financial, `53` Legal & Regulatory, `26` Tax, `111` Other-Systematic, `126` Other-Idiosyncratic), enforcing **case-sensitive matching** on all uppercase/dotted acronyms (`IT`, `I.T.`, `IRS`, `EU`, `VAT`, `SPE`, `VIE`, `MBS`, `MSR`, `REIT`, `IFRS`, `GDP`, `GNP`, `OPEB`, `RMB`, `SARS`, `FIN 48`) per Appendix A (p. 39). Satisfies `FIN + LR + TAX + SYS + IDIO == KW_1A_RAW` identically. | Campbell et al. (2014), Appendices A & C (pp. 39–46) |
| `RF_FIN`, `RF_LR`, `RF_TAX`, `RF_SYS`, `RF_IDIO` | Subcategory keyword proportions (`KW_1A_<CAT> / KW_1A_RAW`) when `KW_1A_RAW > 0`. Satisfies `RF_FIN + RF_LR + RF_TAX + RF_SYS + RF_IDIO == 1.0` identically. | Campbell et al. (2014), Eq. (1) & Table 2 |
| `KW_1A_INDIA_EXT` | Separate count of India-specific statutory/macro risk terms (`IBC`, `NCLT`, `NPA`, `GST`, `MAT`, `SEBI`, `RBI`, `monsoon`, `rupee depreciation`, `share pledging`) — kept strictly separate from the `382` US Campbell keywords. | India institutional extension |
| `MDA_DISC_WORDS`, `MDA_DISC_KW`, `ALL_DISC` | $\ln(1 + \text{words in MDA\_ExRisk})$, $\ln(1 + \text{risk keywords in MDA\_ExRisk})$, and $\ln(1 + \text{narrative words in report})$. | Campbell et al. (2014), Eqs. (2)–(4) |
| `RF_FOG`, `MDA_FOG`, `FOG_OTHER`, `ALL_FOG` | Acronym-safe Gunning Fog readability index $0.4 \times (\text{Words/Sentences} + 100 \times \text{Complex Words/Words})$ across `Risk_Section`, `MDA_ExRisk`, disjoint `Other`, and whole report. | Campbell et al. (2014) / Beatty et al. (2019) |

---

## 4. Beatty, Cheng & Zhang (2019, *CAR*) — `WCopyfind` 6-Word Phrase Decomposition & 2-Factor Scores

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `KW_1A`, `KW_MDA`, `KW_OTHER`, `KW_MDA_Full` | Campbell risk keyword counts (scaled by `1,000`) across mutually disjoint `Risk_Section`, `MDA \ Risk`, and `Other`, plus unsubtracted `MDA_Full`. | Beatty et al. (2019), Sec. 4 & Appendix |
| `WC_1A`, `WC_MDA`, `WC_OTHER` | Narrative word counts (scaled by `1,000`) across mutually disjoint `Risk_Section`, `MDA \ Risk`, and `Other`. | Beatty et al. (2019), Sec. 4 |
| `KW_1A_REP`, `KW_1A_NEW`, `KW_1A_DISC` | **`WCopyfind v4.1.5` 6-word exact phrase decomposition** (scaled by `1,000`): contiguous runs of $\ge 6$ matching words between years $t-1$ and $t$ are joined with `' ; '` (preventing cross-gap false phrases); `KW_1A_NEW = KW_1A_t - KW_1A_REP_t` and `KW_1A_DISC = KW_1A_{t-1} - KW_1A_REP_{t-1}` (p. 145), guaranteeing `KW_1A_REP + KW_1A_NEW == KW_1A` identically. | Beatty et al. (2019), Sec. 4 (p. 145) & Figs. 2–3 |
| `WC_1A_REP`, `WC_1A_NEW`, `WC_1A_DISC`, `PCT_REPEATED_WC`, `PCT_REPEATED_KW` | Total word-count equivalents (scaled by `1,000`) and proportions of repeated words/keywords in `Risk_Section`. | Beatty et al. (2019), Sec. 4 & Table 1 |
| `Delta_KW_1A`, `Delta_KW_MDA`, `Delta_KW_OTHER`, `Delta_WC_*`, `Delta_FOG_*` | Consecutive-year changes in section risk keywords, word counts, and Gunning Fog indices. | Beatty et al. (2019), Models (1)–(3) |
| `FIN_SYS_1A`, `IDIO_LR_1A` | Varimax-rotated 2-factor regression scores extracted from the 5 Campbell risk subcategories (`FIN`, `SYS`, `IDIO`, `LR`, `TAX`). | Beatty et al. (2019), Sec. 5 & Appendix |
| `POST_2009_Crisis`, `POST_2015_LODR_CA2013` | Regime indicators for post-GFC (`year >= 2009`) and post-Companies Act 2013 / SEBI LODR (`year >= 2015`). | Beatty et al. (2019) / India regimes |

---

## 5. Hope, Hu & Lu (2016, *RAST*) — 7-Category Statistical NER `Specificity`

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `Specificity_1A` | Total non-stopword tokens inside non-overlapping 7-category Named Entities (`Person`, `Location`, `Organization`, `Money`, `Percent`, `Date`, `Time`) extracted via `spaCy en_core_web_sm` statistical NER + Indian Rupee/Crore/Lakh/FY rules (excluding generic self-references like *"The Company"* or *"Board of Directors"*), **divided by total words in `Risk_Section` after removing `nltk` English stopwords**. | Hope, Hu & Lu (2016), Sec. 3.1, Footnote 13 & Appendix C (p. 39) |
| `Specificity_10K` | Same 7-category statistical NER `Specificity` ratio computed over the rest of the annual report (`full_ex_1a`, excluding `Risk_Section` and AGM Notice). | Hope, Hu & Lu (2016), Eq. (1) & Appendix C |
| `Spec_1A_Specific_Words`, `Spec_1A_NonStop_Words` | Exact numerator (specific entity words) and denominator (total non-stopword tokens) of `Specificity_1A`. | Hope, Hu & Lu (2016), Appendix C |
| `Spec_1A_Money`, `Spec_1A_Percent`, `Spec_1A_Date`, `Spec_1A_Time`, `Spec_1A_Org`, `Spec_1A_Loc`, `Spec_1A_Person` | Non-overlapping entity word counts across each of the 7 Stanford/spaCy NER classes (`sum == Spec_1A_Specific_Words`). | Hope, Hu & Lu (2016), Table 2 & Table 6 |

---

## 6. Kravet & Muslu (2013, *RAST*) & Li (2006) — Sentence-Level Risk Disclosures & Peer Decomposition

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `KM_Risk_Disclosure` | Total narrative sentences in the annual report (excluding AGM Notice / Proxy boilerplate and tables) matching any of Kravet & Muslu's **18 exact Table A1 keyword rules** (`can/cannot`, `could`, `may`, `might`, `risk*`, `uncertain*`, `likely to`, `subject to`, `potential*`, `vary*/varies`, `depend*`, `expos*`, `fluctuat*`, `possibl*`, `susceptible`, `affect` [exact], `influenc*`, `hedg*`). | Kravet & Muslu (2013), Sec. 2.5 & Table A1 (p. 34) |
| `KM_Negative_RD`, `KM_Other_RD` | Partition of `KM_Risk_Disclosure` into sentences also containing explicit negative-outcome modifiers (`adverse*`, `negativ*`, `loss*`, `lose`, `losing`, `lost`, `declin*`, `decreas*`, `drop*`, `fall*`, `fell`, `fail*`, `deteriorat*`, `impair*`, `damage*`, `harm*`, `hurt*`, `penalt*`, `litigat*`, `default*`, `bankrupt*`, `shortfall*`, `unable`, `inability`, `delay*`, `disrupt*`, `threat*`, `weak*`, `downturn*`, `recession*`, `deficit*`) vs. neutral/other risk sentences. | Kravet & Muslu (2013), Sec. 4.6 |
| `KM_Non_Risk_Disclosure`, `KM_Total_Sentences` | Non-risk narrative sentences (`KM_Total_Sentences - KM_Risk_Disclosure`) and total narrative sentences. | Kravet & Muslu (2013), Eq. (1) |
| `KM_Delta_RD`, `KM_Delta_Negative_RD`, `KM_Delta_Other_RD`, `KM_Delta_Non_RD` | Consecutive-year (`year_t - year_{t-1} == 1`) annual sentence changes. | Kravet & Muslu (2013), Eq. (1) |
| `KM_Industry_Peer_N`, `KM_Delta_Industry_RD`, `KM_Delta_Firm_RD` | Peer firm count in 2-digit `NIC` × `year`, industry-year median `KM_Delta_RD` (**requiring `KM_Industry_Peer_N >= 3`**, else `NaN`), and firm-specific component `KM_Delta_RD - KM_Delta_Industry_RD`. | Kravet & Muslu (2013), Sec. 2.6.4 & Table 5 |
| `Li2006_Risk_Words`, `Li2006_Log_Risk_Words`, `Li2006_Delta_Log_Risk` | Count, $\ln(1+\text{count})$, and annual log change of Li (2006) 6 risk/uncertainty words (`risk`, `risks`, `risky`, `uncertain`, `uncertainty`, `uncertainties`). | Li (2006); Kravet & Muslu (2013) |

---

## 7. Li (2010, *JAR*) — Forward-Looking Statements (`FLS`) & Naïve Bayesian Tone/Content ML Classifier

| Variable Name | Formula / Definition | Paper Reference |
| :--- | :--- | :--- |
| `FLS_Count` / `FLS_Share` | Count and share (`FLS_Count / mda_sentences`) of Forward-Looking Sentences in `MDA_Full` matching Li's **17 exact Appendix B triggers** (`will`, `should`, `can`, `could`, `may`, `might`, `expect*`, `anticipate*`, `believe*`, `plan*`, `hope*`, `intend*`, `seek*`, `project*`, `forecast*`, `objective`, `goal` — **no `"shall"`**), excluding ALL-CAPS sentences, all **10 legal-boilerplate words** (`undersigned`, `herein`, `hereinafter`, `hereof`, `hereon`, `hereto`, `theretofore`, `therein`, `thereof`, `thereon`), safe-harbor disclaimers, and **past-tense retrospections** (`was/were/had/had been` + `expected/anticipated/forecasted/projected/believed`). | Li (2010), Appendix B (pp. 1097–1098) & Footnote 7 |
| `FLS_TONE` / `FLS_TONE_All` | **Naïve Bayesian ML Tone** (`MultinomialNB` on Porter-stemmed tokens with negations `cannot/no/none/nor/not` retained; 5-fold CV accuracy `74.83%` vs `37.06%` Informed Guessing on 4-cat Tone; `77.67%` vs `38.70%` on 3-cat Tone), averaging $+1$ (`POS`), $0$ (`NEU`), and $-1$ (`NEG` or `UNC`) across FLS sentences (`FLS_TONE` requires `FLS_Count >= 5` per p. 1064; `FLS_TONE_All` covers `FLS_Count >= 1`). | Li (2010), Secs. 3–5 & Table 2 |
| `FLS_TONE_Dict` | Dictionary-based FLS tone benchmark (`FLS_Count >= 5`) replicating Li's Section 5.8 comparison showing ML Naïve Bayes vs. dictionary word counts. | Li (2010), Sec. 5.8 |
| `FLS_PROFIT_TONE`, `FLS_LIQUIDITY_TONE`, `FLS_OTHER_TONE` | Average Naïve Bayesian Tone within FLS sentences classified by the Content Naïve Bayesian model (`77.83%` 5-fold CV accuracy vs `46.09%` Informed Guessing) into **Profit/Operations (Cats 1–4)**, **Liquidity/CapEx/Financing (Cats 5–7)**, and **Other (Cats 8–12)**. | Li (2010), Appendix C & Table 8 |
| `FLS_PROFIT_PCT`, `FLS_LIQUIDITY_PCT`, `FLS_OTHER_PCT` | Proportions of FLS sentences in `MDA_Full` classified into Profit/Ops (`1–4`), Liquidity/Cap (`5–7`), and Other (`8–12`); sum equals `1.0000`. | Li (2010), Table 1 & Appendix C |
| `FLS_Pos_Share`, `FLS_Neu_Share`, `FLS_Neg_Share`, `FLS_Unc_Share` | Proportions of FLS sentences classified by the 4-class Naïve Bayesian Tone model as `POS`, `NEU`, `NEG`, and `UNC`; sum equals `1.0000`. | Li (2010), Table 1 |
