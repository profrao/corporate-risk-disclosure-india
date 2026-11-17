#!/usr/bin/env python3
"""
Step 1 (v2 - Audited): Corporate Risk Disclosure & Textual Analytics Panel (1997-2025)
======================================================================================
Implements 100% paper-faithful variable constructions for all 6 foundational corporate
risk disclosure & textual analytics papers on the segmented Indian Annual Reports corpus:

  1. Li (2010, JAR):
     - Exact Appendix B Forward-Looking Statements (FLS) extraction in MD&A:
       17 FLS trigger roots (no "shall"), acronym-safe sentence splitting (Footnote 7),
       excluding ALL-CAPS sentences, all 10 legal-boilerplate words (`undersigned`, `herein`,
       `hereinafter`, `hereof`, `hereon`, `hereto`, `theretofore`, `therein`, `thereof`,
       `thereon`), safe-harbor disclaimers, and past-tense retrospections (`was/were/had/had been`
       + `expected/anticipated/forecasted/projected/believed`).
     - Genuine Porter-stemmed, negation-preserving (`cannot/no/none/nor/not` retained)
       Naïve Bayesian Machine Learning Classifier (`li2010_nb_model.json`) predicting 4-category
       Tone (`POS`=+1, `NEU`=0, `NEG`/`UNC`=-1) and 12 Content Categories (`PROFIT` [1-4],
       `LIQUIDITY` [5-7], `OTHER` [8-12]), plus dictionary tone benchmark (`FLS_TONE_Dict`).

  2. Brown & Tucker (2011, JAR):
     - Porter (1980) stemming on ALL MD&A words (`Footnote 9`), NO stopword removal (`Footnote 10`:
       pure IDF downweighting `IDF_k = ln(M / m_k)`), TF-IDF Vector Space Model cosine
       dissimilarity (`MDA_Rawscore`, plus fiscal-year cohort `MDA_Rawscore_CohortIDF`).
     - 5th-order Taylor polynomial adjustment on current-year MD&A word count (`Length_t^1..5`,
       `Appendix B, Table B1`) -> `MDA_Score` (and `Risk_Rawscore`, `Risk_Score`).

  3. Kravet & Muslu (2013, RAST):
     - Whole-report acronym-safe sentence parsing excluding AGM Notice / Proxy legal boilerplate.
     - Exact 18 Table A1 keyword rules (`affect` exact without `*`; `*` suffixes only where
       specified in Table A1) -> `KM_Risk_Disclosure`, `KM_Non_Risk_Disclosure`, plus Section 4.6
       negative-outcome risk sentences (`KM_Negative_RD`, `KM_Other_RD`).
     - Annual changes (`KM_Delta_RD`, `KM_Delta_Non_RD`, `KM_Delta_Negative_RD`) decomposed into
       `KM_Delta_Industry_RD` (industry-year median requiring >=3 peer firms in 2-digit NIC x year)
       and `KM_Delta_Firm_RD`.

  4. Campbell, Chen, Dhaliwal, Lu & Steele (2014, RAST):
     - All 382 PDF-verified Appendix C keywords compiled into a single longest-phrase-first,
       non-overlapping matcher with case-sensitive acronym matching (`IT`, `IRS`, `EU`, `VAT`,
       `SPE`, `VIE`, `MBS`, `MSR`, `REIT`, `IFRS`, `GDP`, `GNP`, `OPEB`, `RMB`, `SARS`, `FIN 48`
       per Appendix A, PDF p. 39) and punctuation-safe word boundaries `(?<![A-Za-z0-9])...(?![A-Za-z0-9])`.
     - Disjoint section controls (`MDA_ExRisk` and `OTHER` excluding 1A, MD&A, and AGM Notice)
       so `KW_1A`, `KW_MDA`, and `KW_OTHER` have zero overlap.

  5. Hope, Hu & Lu (2016, RAST):
     - Statistical Named Entity Recognition (`spaCy en_core_web_sm` CRF/CNN NER + Indian Rupee/Crore/Lakh
       currency & fiscal-date entity rules) restricted to the exact 7 Stanford NER categories
       (`Person`, `Location`, `Organization`, `Money`, `Percent`, `Date`, `Time`; Footnote 13).
     - Exact Appendix C numerator & denominator: number of specific entity words divided by
       total words AFTER removing `nltk.corpus.stopwords.words('english')`, computed on both
       `Risk_Section` (`Specificity_1A`) and the rest of the annual report (`Specificity_10K`).

  6. Beatty, Cheng & Zhang (2019, CAR):
     - `WCopyfind v4.1.5` 6-word exact phrase matching between consecutive years' Risk disclosures,
       joining contiguous matched phrases with `" ; "` (never gluing across deleted/new gaps),
       decomposing both risk keywords (`KW_1A_REP`, `KW_1A_NEW`, `KW_1A_DISC`, scaled by 1,000)
       and total words (`WC_1A_REP`, `WC_1A_NEW`, `WC_1A_DISC`, `PCT_REPEATED_WC`), plus 2-factor
       PCA/FA scores (`FIN_SYS_1A`, `IDIO_LR_1A`).
"""

import os

# Prevent BLAS/Accelerate thread oversubscription across 17 multiprocessing spaCy/thinc workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import re
import json
import math
import time
import itertools
import argparse
from functools import lru_cache
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AR_DIR = os.path.join(BASE_DIR, "annual_reports")
STEP0_DIR = os.path.join(BASE_DIR, "04_outputs", "step0_sections")
MANIFEST_CSV = os.path.join(STEP0_DIR, "section_manifest.csv")
LEXICON_JSON = os.path.join(BASE_DIR, "03_code", "step1_lexicons.json")
OUT_DIR = os.path.join(BASE_DIR, "04_outputs", "step1_paper_metrics")
LI_NB_MODEL_JSON = os.path.join(OUT_DIR, "li2010_nb_model.json")

# ---------------------------------------------------------------------------
# 1. Load Verified Lexicons & Build Exact Non-Overlapping Matchers
# ---------------------------------------------------------------------------
with open(LEXICON_JSON, "r", encoding="utf-8") as f:
    LEX = json.load(f)

NLTK_STOPWORDS = set(stopwords.words("english"))
LI_NEGATIONS = set(LEX["li_2010"]["negation_keep_words"])
LI_STOPWORDS = NLTK_STOPWORDS - LI_NEGATIONS
_PORTER = PorterStemmer()


@lru_cache(maxsize=250000)
def stem_word(w):
    """Cached Martin Porter (1980) stemmer (exact for Brown & Tucker 2011 and Li 2010)."""
    return _PORTER.stem(w)


def _expand_campbell_kw(kw):
    """Expand parenthesized alternatives like 'Capital (lease|leases)' into explicit phrases."""
    tokens = kw.strip().split()
    options_per_tok = []
    for tok in tokens:
        if tok.startswith("(") and tok.endswith(")") and "|" in tok:
            opts = [x.strip() for x in tok[1:-1].split("|") if x.strip()]
            options_per_tok.append(opts)
        else:
            options_per_tok.append([tok])
    return [" ".join(comb) for comb in itertools.product(*options_per_tok)]


def build_campbell_master_matcher():
    """
    Build a single unified, longest-phrase-first, non-overlapping regex matcher across
    all 382 Campbell et al. (2014) Appendix C keywords, honoring case-sensitivity on
    acronyms (Appendix A, PDF p. 39) and punctuation-safe word boundaries.
    """
    cs_acronyms = set(LEX["campbell_2014_case_sensitive"])
    exact_cat_map = {}   # case-sensitive surface -> category
    lower_cat_map = {}   # lowercased surface -> category
    regex_entries = []   # (word_count, char_len, regex_str)

    # Order categories so Financial, Tax, Legal take precedence before Other-Idio / Other-Sys
    # per Campbell et al. (2014) Footnote 3 ("We first classify key words into financial, tax, and legal...")
    cat_priority = ("RF_FIN", "RF_TAX", "RF_LR", "RF_IDIO", "RF_SYS")
    seen_exact = set()
    seen_lower = set()

    for cat in cat_priority:
        for raw_kw in LEX["campbell_2014"][cat]:
            is_cs = raw_kw in cs_acronyms
            for surf in _expand_campbell_kw(raw_kw):
                norm_surf = " ".join(surf.split())
                if not norm_surf:
                    continue
                esc_parts = [re.escape(p) for p in norm_surf.split()]
                joined_pat = r"\s+".join(esc_parts)
                if is_cs:
                    if norm_surf in seen_exact:
                        continue
                    seen_exact.add(norm_surf)
                    exact_cat_map[norm_surf] = cat
                    pat_str = joined_pat
                else:
                    low_s = norm_surf.lower()
                    if low_s in seen_lower:
                        continue
                    seen_lower.add(low_s)
                    lower_cat_map[low_s] = cat
                    pat_str = f"(?i:{joined_pat})"
                regex_entries.append((len(norm_surf.split()), len(norm_surf), pat_str))

    # Sort strictly longest-first so 'Value added tax' matches before 'Tax' and 'Capital lease' before 'lease'
    regex_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
    master_alt = "|".join(x[2] for x in regex_entries)
    master_re = re.compile(rf"(?<![A-Za-z0-9])(?:{master_alt})(?![A-Za-z0-9])")
    return master_re, exact_cat_map, lower_cat_map


CAMPBELL_MASTER_RE, CAMPBELL_EXACT_MAP, CAMPBELL_LOWER_MAP = build_campbell_master_matcher()

# India-specific institutional risk lexicon extension (tracked separately so US Campbell counts stay 100% pure)
INDIA_RISK_EXT_RE = re.compile(
    r"\b(?:non[\s\-]*performing\s+assets?|npa|insolvency\s+and\s+bankruptcy\s+code|ibc|nclt|nclat|"
    r"corporate\s+debt\s+restructuring|external\s+commercial\s+borrowings?|ecb|pledg(?:e|ing)\s+of\s+shares|"
    r"working\s+capital\s+cycle|consortium\s+of\s+banks|cash\s+credit\s+limit|rbi\s+repo\s+rate|moratorium|"
    r"goods\s+and\s+services?\s+tax|gst|input\s+tax\s+credit|minimum\s+alternate\s+tax|mat\s+credit|"
    r"income\s+tax\s+appellate\s+tribunal|itat|customs?\s+duty|excise\s+duty|service\s+tax|cess|"
    r"show[\s\-]*cause\s+notice|sebi\s+regulations?|stock\s+exchange\s+penalt\w+|high\s+court|supreme\s+court|"
    r"arbitration\s+award|pollution\s+control\s+board|cpcb|spcb|national\s+green\s+tribunal|ngt|"
    r"competition\s+commission\s+of\s+india|cci|contingent\s+liabilit\w+|"
    r"monsoon|erratic\s+rainfall|rupee\s+depreciation|inr[\s\/]*usd|demoneti[sz]ation|"
    r"geopolitical\s+tensions?|global\s+slowdown|sovereign\s+rating|fiscal\s+deficit|current\s+account\s+deficit)\b",
    re.I,
)

# Kravet & Muslu (2013) Table A1 (18 risk keywords) & Section 4.6 (negative risk modifiers)
KM_RISK_SENT_RE = re.compile("|".join(LEX["kravet_muslu_2013"]), re.I)
KM_NEG_MOD_RE = re.compile("|".join(LEX["kravet_muslu_2013_neg"]), re.I)
LI2006_RISK_RE = re.compile(r"\b(?:risk|risks|risky|uncertain|uncertainty|uncertainties)\b", re.I)

# Li (2010) Appendix B exact FLS filters
LI_CFG = LEX["li_2010"]
FLS_TRIGGER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in LI_CFG["fls_triggers"]) + r")\b",
    re.I,
)
FLS_LEGAL_EXCL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in LI_CFG["legal_boilerplate_exclusions"]) + r")\b|"
    r"cautionary\s+statement|meaning\s+of\s+applicable\s+securities\s+laws|"
    r"actual\s+results\s+(?:may|could|might)\s+differ\s+materially|"
    r"statements?\s+in\s+(?:this|the)\s+management\s+discussion\s+(?:and|&)\s+analysis\s+describing|"
    r"undertakes?\s+no\s+obligation\s+to\s+publicly\s+update",
    re.I,
)
FLS_RETRO_EXCL_RE = re.compile(LI_CFG["retrospection_pattern"], re.I)

# Dictionary Tone Benchmark for Li (2010) Section 5.8 comparison
POS_TONE_RE = re.compile(
    r"\b(?:grow|grows|growing|growth|increase|increases|increasing|improve|improves|improving|improvement|"
    r"strong|stronger|strength|strengthen|robust|favorable|favourable|optimistic|confident|positive|"
    r"profit|profitable|profitability|gain|gains|expand|expands|expanding|expansion|opportunity|opportunities|"
    r"healthy|upside|outperform|leadership|momentum|recovery|efficient|efficiency|synergy|synergies|"
    r"buoyant|accelerate|accelerating|benefit|benefits|beneficial|achieve|achieving|"
    r"enhance|enhancing|enhancement|successful|successfully|lucrative|promising|advancement|superior)\b",
    re.I,
)
NEG_TONE_RE = re.compile(
    r"\b(?:decline|declines|declining|decrease|decreases|decreasing|loss|losses|adverse|adversely|"
    r"deteriorate|deteriorating|deterioration|weak|weaken|weakening|weakness|slowdown|slowing|stagnant|"
    r"impair|impaired|impairment|default|defaults|litigation|penalty|penalties|shortage|shortages|"
    r"disruption|disruptions|delay|delays|delayed|threat|threats|deficit|downturn|recession|"
    r"bottleneck|bottlenecks|erosion|erode|pressure|pressures|challenging|challenge|challenges|"
    r"constraint|constraints|damage|negative|negatively|vulnerability|vulnerable|attrition|shrink|shrinking|"
    r"underperform|severe|severely|crisis|crises|insolvency|bankruptcy|write[\s\-]*off)\b",
    re.I,
)
UNC_TONE_RE = re.compile(
    r"\b(?:uncertain|uncertainty|uncertainties|volatile|volatility|fluctuate|fluctuates|fluctuating|fluctuation|"
    r"unpredictable|unforeseen|contingent|contingency|doubt|doubtful|ambiguous|instability|unstable|"
    r"speculative|turbulence|turbulent|erratic|unclear|exposure|risk|risks|risky)\b",
    re.I,
)

# Hope, Hu & Lu (2016) Indian financial entity supplemental overlays (paired with spaCy CRF/CNN NER)
IND_MONEY_RE = re.compile(
    r"(?:₹|rs\.?|inr)\s*\d[\d,\.]*(?:\s*(?:crores?|crs?\.?|lakhs?|lacs?|millions?|billions?|thousands?|mn|bn))?|"
    r"\b\d[\d,\.]*\s*(?:crores?|crs?\.?|lakhs?|lacs?)\b",
    re.I,
)
IND_PCT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|per\s+cent|percentage\s+points?|bps|basis\s+points?)\b",
    re.I,
)
IND_DATE_RE = re.compile(
    r"\b(?:fy|f\.y\.|ay|a\.y\.)\s*(?:19|20)?\d{2}(?:[\-\–\/]\d{2,4})?\b|"
    r"\b(?:q[1-4]|h[1-2])\s*(?:fy)?\s*(?:19|20)\d{2}(?:[\-\–\/]\d{2})?\b",
    re.I,
)

SPACY_LABEL_TO_HOPE = {
    "PERSON": "Person",
    "GPE": "Location",
    "LOC": "Location",
    "FAC": "Location",
    "ORG": "Organization",
    "MONEY": "Money",
    "PERCENT": "Percent",
    "DATE": "Date",
    "TIME": "Time",
}

MONEY_UNIT_RE = re.compile(
    r"(?:₹|\$|€|£|¥|\b(?:rs|inr|usd|eur|gbp|jpy|rmb|chf|cad|aud|sgd|aed|dollars?|rupees?|euros?|pounds?|yen|yuan|crores?|crs?|lakhs?|lacs?|paise|paisa|cents?)\b)",
    re.I,
)
PCT_UNIT_RE = re.compile(
    r"(?:%|\b(?:percent|per\s+cent|percentage|bps|basis\s+points?)\b)",
    re.I,
)
# Generic non-specific corporate self-references and report section words (Hope et al. 2016, p. 13:
# generic references like 'the company' or 'the firm' are non-specific, unlike named entities like 'Microsoft').
GENERIC_CORP_NON_ENTITY_WORDS = {
    "company", "companies", "firm", "firms", "corporation", "enterprise", "enterprises", "group",
    "board", "director", "directors", "committee", "committees", "subcommittee",
    "management", "senior", "executive", "executives", "officer", "officers", "key", "managerial", "personnel",
    "audit", "auditing", "auditor", "auditors", "statutory", "secretarial", "internal", "external", "cost",
    "risk", "risks", "control", "controls", "system", "systems", "process", "processes", "governance", "corporate",
    "report", "reports", "reporting", "annual", "integrated", "sustainability", "business", "responsibility",
    "financial", "statement", "statements", "standalone", "consolidated", "balance", "sheet", "profit", "loss",
    "cash", "flow", "flows", "account", "accounts", "accounting", "note", "notes", "schedule", "schedules",
    "annexure", "annexures", "appendix", "section", "sections", "item", "part", "chapter", "table", "page",
    "regulation", "regulations", "clause", "act", "rule", "rules", "policy", "policies", "framework", "charter",
    "code", "conduct", "compliance", "vigil", "mechanism", "whistle", "blower", "remuneration", "nomination",
    "stakeholder", "stakeholders", "shareholder", "shareholders", "member", "members", "investor", "investors",
    "employee", "employees", "worker", "workers", "customer", "customers", "client", "clients",
    "vendor", "vendors", "supplier", "suppliers", "partner", "partners", "dealer", "dealers",
    "subsidiary", "subsidiaries", "associate", "associates", "joint", "venture", "ventures", "holding",
    "division", "divisions", "segment", "segments", "unit", "units", "plant", "plants", "factory", "department",
    "ceo", "cfo", "coo", "cio", "cro", "cso", "kmp", "md", "wtd", "chairman", "chairperson", "president",
    "vice", "managing", "whole", "time", "independent", "non", "woman", "secretary", "registrar", "promoter", "promoters",
    "bank", "banks", "banker", "bankers", "lender", "lenders", "institution", "institutions", "agency", "agencies",
    "government", "authority", "authorities", "regulator", "regulators", "ministry", "court", "courts", "tribunal",
    "discussion", "analysis", "overview", "outlook", "review", "operations", "operational", "performance",
    "opportunities", "threats", "concerns", "human", "resources", "industrial", "relations", "material", "developments",
    "cautionary", "forward", "looking", "disclaimer", "summary", "highlights", "particulars", "details",
    "the", "your", "our", "its", "their", "this", "that", "these", "those", "of", "and", "or", "for", "in", "to", "on", "by", "with", "at", "from", "as",
}

SENT_STARTER_NON_ENTITY = {
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "by", "with", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these", "those",
    "we", "our", "your", "they", "their", "there", "here", "however", "further", "furthermore",
    "moreover", "therefore", "thus", "hence", "consequently", "accordingly", "additionally", "also",
    "although", "though", "while", "whereas", "during", "under", "pursuant", "subject", "based",
    "all", "any", "each", "every", "some", "no", "not", "if", "when", "where", "which", "who",
    "after", "before", "since", "until", "upon", "within", "without", "through", "across", "among", "between",
}
DIGIT_OR_CURR_RE = re.compile(r"[0-9₹$€£¥%]")
CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z0-9\.\-&']*")

# General cleaning & acronym-safe sentence splitting
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
IMG_OR_HTML_RE = re.compile(r"<[^>]+>|!\[[^\]]*\]\([^\)]*\)")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]*\)")
SYM_RE = re.compile(r"[\*\#\_\`\~\|]+")
ALPHA_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")
ALNUM_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.I)

ABBREV_PROTECT_RE = re.compile(
    r"\b(?:Rs|Dr|Mr|Mrs|Ms|Prof|Shri|Smt|No|Nos|Sr|Jr|St|Ltd|Pvt|Inc|Corp|Co|vs|approx|Fig|Sec|Reg|Art|Ch|Vol|pp|viz|al|etc)\.",
    re.I,
)
DOTTED_ABBREV_RE = re.compile(r"\b(?:w\.e\.f|i\.e|e\.g|a\.m|p\.m|f\.y|a\.y|u\.s|u\.k)\.", re.I)
INITIAL_PROTECT_RE = re.compile(r"(?<=\b[A-Z])\.(?=\s+[A-Z])")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

# Worker-local globals initialized once per process
_WORKER_NLP = None
_WORKER_LI_NB = None


def _init_worker(li_nb_path):
    """Initialize spaCy NER pipeline and Li (2010) Naive Bayes parameters once per worker."""
    global _WORKER_NLP, _WORKER_LI_NB
    _WORKER_NLP = spacy.load(
        "en_core_web_sm",
        exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    if li_nb_path and os.path.exists(li_nb_path):
        with open(li_nb_path, "r", encoding="utf-8") as f:
            _WORKER_LI_NB = json.load(f)
    else:
        _WORKER_LI_NB = None


def split_sentences_safe(text):
    """Split narrative text into sentences without breaking on 'Rs. 500', 'No. 12', 'Mr. Rao', etc."""
    if not text:
        return []
    protected = ABBREV_PROTECT_RE.sub(lambda m: m.group(0)[:-1] + "\x00", text)
    protected = DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), protected)
    protected = INITIAL_PROTECT_RE.sub("\x00", protected)
    raw_sents = SENT_SPLIT_RE.split(protected)
    return [s.replace("\x00", ".").strip() for s in raw_sents if s.strip()]


def lines_to_narrative_with_qc(lines_list):
    """
    Convert raw Markdown lines into clean narrative text (excluding tables & images), while:
      1. Terminating unpunctuated Markdown headings with '.' so headings never merge into sentences.
      2. Stripping isolated PDF `/ToUnicode` CMap-garbled lines (e.g., 'vjg qh cpf vq kp ctg').
      3. Flagging `is_cmap_corrupted=True` if >25% of the section's words were lost to CMap font
         corruption or the surviving text has <18% English stopwords (non-English/font-corrupted).
    """
    out = []
    raw_alpha_words = 0
    garbled_alpha_words = 0
    clean_alpha_words = 0
    clean_sw_words = 0

    for l in lines_list:
        s = l.strip()
        if not s or s.startswith("|") or s.startswith("![") or s.startswith("<!--"):
            continue
        is_heading = s.startswith("#") or (s.startswith("**") and s.endswith("**") and len(s) <= 140)
        s = IMG_OR_HTML_RE.sub(" ", s)
        s = MD_LINK_RE.sub(r"\1", s)
        s = SYM_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue

        ws = ALPHA_WORD_RE.findall(s)
        n_w = len(ws)
        raw_alpha_words += n_w

        if n_w >= 8:
            sw_cnt = sum(1 for w in ws if w.lower() in NLTK_STOPWORDS)
            novow_low_cnt = sum(
                1 for w in ws if len(w) >= 3 and w.islower() and not VOWEL_GROUP_RE.search(w)
            )
            if (sw_cnt / n_w < 0.06 and novow_low_cnt >= 1) or (novow_low_cnt / n_w > 0.25):
                garbled_alpha_words += n_w
                continue

        clean_alpha_words += n_w
        clean_sw_words += sum(1 for w in ws if w.lower() in NLTK_STOPWORDS)

        if is_heading and s[-1] not in ".!?:;":
            s = s + "."
        out.append(s)

    is_corrupted = False
    if raw_alpha_words >= 30:
        if garbled_alpha_words / raw_alpha_words > 0.25:
            is_corrupted = True
        elif clean_alpha_words >= 30 and (clean_sw_words / clean_alpha_words) < 0.18:
            is_corrupted = True

    return re.sub(r"\s+", " ", " ".join(out)).strip(), is_corrupted


def parse_spans_str(spans_str, max_line):
    """Parse '100-150;220-260' into a list of (start, end) 1-based inclusive line tuples."""
    if not isinstance(spans_str, str) or not spans_str.strip():
        return []
    out = []
    for part in spans_str.split(";"):
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                s, e = max(1, int(a)), min(max_line, int(b))
                if e >= s:
                    out.append((s, e))
    return out


@lru_cache(maxsize=100000)
def count_syllables_word(w):
    """Cached syllable counter for Gunning Fog Index (>=3 syllables = complex word)."""
    w = w.lower()
    if len(w) <= 3:
        return 1
    if w.endswith(("es", "ed")) and not w.endswith(("ted", "ded", "ses", "zes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    groups = VOWEL_GROUP_RE.findall(w)
    return max(1, len(groups))


def compute_fog_and_words(text):
    """Return (word_count, sentence_count, complex_word_count, gunning_fog_index)."""
    if not text:
        return 0, 0, 0, np.nan
    words = ALPHA_WORD_RE.findall(text)
    n_words = len(words)
    sents = split_sentences_safe(text)
    n_sents = max(1, len(sents))
    if n_words < 20:
        return n_words, n_sents, 0, np.nan
    complex_words = sum(1 for w in words if count_syllables_word(w.lower()) >= 3)
    asl = min(n_words / n_sents, 80.0)
    pct_complex = 100.0 * (complex_words / n_words)
    fog = 0.4 * (asl + pct_complex)
    return n_words, n_sents, complex_words, round(fog, 3)


def count_campbell_keywords(text, check_india_ext=False):
    """
    Exact Campbell et al. (2014) 5-category non-overlapping longest-match-first keyword counter,
    honoring case-sensitivity on uppercase acronyms (Appendix A, PDF p. 39).
    """
    res = {
        "RF_FIN": 0, "RF_LR": 0, "RF_TAX": 0, "RF_SYS": 0, "RF_IDIO": 0,
        "RF_TOTAL": 0, "IND_EXT": 0,
    }
    if not text:
        return res
    for m in CAMPBELL_MASTER_RE.finditer(text):
        surf = " ".join(m.group(0).split())
        cat = CAMPBELL_EXACT_MAP.get(surf)
        if cat is None:
            cat = CAMPBELL_LOWER_MAP.get(surf.lower())
        if cat is not None:
            res[cat] += 1
    res["RF_TOTAL"] = res["RF_FIN"] + res["RF_LR"] + res["RF_TAX"] + res["RF_SYS"] + res["RF_IDIO"]
    if check_india_ext:
        res["IND_EXT"] = len(INDIA_RISK_EXT_RE.findall(text))
    return res


def _chunk_text_on_spaces(text, chunk_chars=35000):
    """Split long text into space-aligned chunks <= chunk_chars for fast low-memory spaCy NER."""
    if len(text) <= chunk_chars:
        return [(0, text)]
    chunks = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(n, pos + chunk_chars)
        if end < n:
            sp = text.rfind(" ", pos + chunk_chars // 2, end)
            if sp != -1:
                end = sp
        chunks.append((pos, text[pos:end]))
        pos = end + 1 if (end < n and text[end] == " ") else end
    return chunks


def _sentence_may_have_entity(sent):
    """Lossless fast filter: skip pure-lowercase non-numeric sentences starting only with a stopword."""
    if DIGIT_OR_CURR_RE.search(sent):
        return True
    caps = list(CAP_WORD_RE.finditer(sent))
    if not caps:
        return False
    if len(caps) == 1 and caps[0].start() <= 2 and caps[0].group(0).lower() in SENT_STARTER_NON_ENTITY:
        return False
    return True


def _is_valid_hope_span(span_txt, h_cat):
    """
    Validate spaCy NER spans against Hope, Hu & Lu (2016) Section 3.1 & Appendix C definitions:
      - Money spans must include a currency symbol or monetary unit (reject bare section numbers).
      - Percent spans must include '%' or percentage/bps unit.
      - Organization/Person/Location spans must contain at least one proper/specific token beyond
        generic corporate self-references ('The Company', 'Board of Directors', 'Audit Committee').
    """
    if h_cat == "Money":
        return bool(MONEY_UNIT_RE.search(span_txt))
    if h_cat == "Percent":
        return bool(PCT_UNIT_RE.search(span_txt))
    if h_cat in ("Organization", "Person", "Location"):
        alpha_toks = [w.lower() for w in ALPHA_WORD_RE.findall(span_txt)]
        if not alpha_toks:
            return False
        if all(w in GENERIC_CORP_NON_ENTITY_WORDS for w in alpha_toks):
            return False
    return True


def compute_hope_specificity(text, prefilter_long_text=False):
    """
    Exact Hope, Hu & Lu (2016, RAST) Specificity implementation (Section 3.1 & Appendix C):
      - Statistical NER (`spaCy en_core_web_sm` CRF/CNN + Indian currency/percent/FY overlays)
        mapped strictly to the 7 Stanford NER categories: Person, Location, Organization,
        Money, Percent, Date, Time.
      - Numerator: number of specific non-stopword tokens inside non-overlapping 7-category entities.
      - Denominator: total words in section AFTER `nltk` English stop-words removed.
    """
    empty = {
        "spec_money": 0, "spec_pct": 0, "spec_date": 0, "spec_time": 0,
        "spec_org": 0, "spec_loc": 0, "spec_person": 0,
        "spec_words": 0, "nonstop_words": 0, "specificity": np.nan,
    }
    if not text:
        return empty

    all_tokens = ALNUM_WORD_RE.findall(text)
    nonstop_words = sum(1 for w in all_tokens if w.lower() not in NLTK_STOPWORDS)
    if nonstop_words < 10:
        return empty

    ner_input_text = text
    if prefilter_long_text and len(text) > 10000:
        cand_sents = [s for s in split_sentences_safe(text) if _sentence_may_have_entity(s)]
        ner_input_text = " ".join(cand_sents)
        if not ner_input_text:
            empty["nonstop_words"] = nonstop_words
            empty["specificity"] = 0.0
            return empty

    # Collect candidate entity spans on `ner_input_text`: (priority, start_char, end_char, hope_category)
    # Priority 0 = explicit Indian Money/Percent/FY regex; Priority 1 = spaCy statistical NER
    cand_spans = []
    for m in IND_MONEY_RE.finditer(ner_input_text):
        cand_spans.append((0, m.start(), m.end(), "Money"))
    for m in IND_PCT_RE.finditer(ner_input_text):
        cand_spans.append((0, m.start(), m.end(), "Percent"))
    for m in IND_DATE_RE.finditer(ner_input_text):
        cand_spans.append((0, m.start(), m.end(), "Date"))

    chunks = _chunk_text_on_spaces(ner_input_text, 4500)
    chunk_offsets = [c[0] for c in chunks]
    chunk_strs = [c[1] for c in chunks]

    for base_off, doc in zip(chunk_offsets, _WORKER_NLP.pipe(chunk_strs, batch_size=24)):
        for ent in doc.ents:
            h_cat = SPACY_LABEL_TO_HOPE.get(ent.label_)
            if h_cat is not None and _is_valid_hope_span(ent.text, h_cat):
                cand_spans.append((1, base_off + ent.start_char, base_off + ent.end_char, h_cat))

    if not cand_spans:
        empty["nonstop_words"] = nonstop_words
        empty["specificity"] = 0.0
        return empty

    # Resolve overlapping spans greedily by (priority asc, length desc, start asc)
    cand_spans.sort(key=lambda x: (x[0], -(x[2] - x[1]), x[1]))
    accepted = []  # list of (s, e, cat)
    for _, s, e, cat in cand_spans:
        if e <= s:
            continue
        overlap = False
        for a_s, a_e, _ in accepted:
            if s < a_e and e > a_s:
                overlap = True
                break
        if not overlap:
            accepted.append((s, e, cat))

    cat_word_counts = defaultdict(int)
    for s, e, cat in accepted:
        span_toks = [w for w in ALNUM_WORD_RE.findall(ner_input_text[s:e]) if w.lower() not in NLTK_STOPWORDS]
        if span_toks:
            cat_word_counts[cat] += len(span_toks)

    spec_words = sum(cat_word_counts.values())
    spec_ratio = min(1.0, spec_words / nonstop_words)

    return {
        "spec_money": cat_word_counts["Money"],
        "spec_pct": cat_word_counts["Percent"],
        "spec_date": cat_word_counts["Date"],
        "spec_time": cat_word_counts["Time"],
        "spec_org": cat_word_counts["Organization"],
        "spec_loc": cat_word_counts["Location"],
        "spec_person": cat_word_counts["Person"],
        "spec_words": spec_words,
        "nonstop_words": nonstop_words,
        "specificity": round(spec_ratio, 5),
    }


def _nb_predict_class(stems, nb_dict):
    """Predict argmax class from exported MultinomialNB log-priors and stem log-probabilities."""
    classes = nb_dict["classes"]
    scores = list(nb_dict["class_log_prior"])
    stem_lp = nb_dict["stem_log_prob"]
    n_c = len(classes)
    for st in stems:
        lp = stem_lp.get(st)
        if lp is not None:
            for i in range(n_c):
                scores[i] += lp[i]
    best_i = 0
    best_v = scores[0]
    for i in range(1, n_c):
        if scores[i] > best_v:
            best_v = scores[i]
            best_i = i
    return classes[best_i]


def compute_li2010_fls(mda_text):
    """
    Exact Li (2010, JAR) Forward-Looking Statement (FLS) extraction & Naïve Bayesian
    Tone and Content classification (Sections 3-5, Appendices B-D).
    """
    nan_out = {
        "fls_count": 0, "fls_share": np.nan,
        "fls_tone": np.nan, "fls_tone_all": np.nan, "fls_tone_dict": np.nan,
        "fls_profit_tone": np.nan, "fls_liquidity_tone": np.nan, "fls_other_tone": np.nan,
        "fls_profit_pct": np.nan, "fls_liquidity_pct": np.nan, "fls_other_pct": np.nan,
        "fls_pos_share": np.nan, "fls_neu_share": np.nan, "fls_neg_share": np.nan, "fls_unc_share": np.nan,
    }
    if not mda_text:
        return nan_out

    sents = [s for s in split_sentences_safe(mda_text) if len(s) >= 25]
    if not sents:
        return nan_out

    fls_sents = []
    for s in sents:
        if len(s) > 650:
            continue
        alpha_chars = [c for c in s if c.isalpha()]
        if not alpha_chars or sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.80:
            continue
        if len(ALPHA_WORD_RE.findall(s)) < 5:
            continue
        if not FLS_TRIGGER_RE.search(s):
            continue
        if FLS_LEGAL_EXCL_RE.search(s):
            continue
        if FLS_RETRO_EXCL_RE.search(s):
            continue
        fls_sents.append(s)

    n_fls = len(fls_sents)
    if n_fls == 0:
        nan_out["fls_share"] = 0.0
        return nan_out

    nb_model = _WORKER_LI_NB
    pos_c = neu_c = neg_c = unc_c = 0
    prof_tones, liq_tones, oth_tones = [], [], []
    nb_tone_sum = 0.0
    dict_tone_sum = 0.0

    for s in fls_sents:
        # 1. Dictionary benchmark (Li 2010 Section 5.8)
        d_pos = len(POS_TONE_RE.findall(s))
        d_neg = len(NEG_TONE_RE.findall(s))
        d_unc = len(UNC_TONE_RE.findall(s))
        if d_pos > (d_neg + d_unc):
            dict_tone_sum += 1.0
        elif (d_neg + d_unc) > d_pos:
            dict_tone_sum -= 1.0

        # 2. Naïve Bayesian Machine Learning Prediction (Li 2010 Sections 3-4)
        stems = [stem_word(w.lower()) for w in ALPHA_WORD_RE.findall(s) if w.lower() not in LI_STOPWORDS]
        if nb_model is not None and stems:
            pred_t = _nb_predict_class(stems, nb_model["tone4"])
            pred_c = _nb_predict_class(stems, nb_model["cat3"])
        else:
            pred_t = "POS" if d_pos > (d_neg + d_unc) else ("NEG" if d_neg >= d_unc and (d_neg + d_unc) > d_pos else ("UNC" if (d_neg + d_unc) > d_pos else "NEU"))
            pred_c = "PROFIT"

        if pred_t == "POS":
            t_val = 1.0
            pos_c += 1
        elif pred_t == "NEU":
            t_val = 0.0
            neu_c += 1
        elif pred_t == "NEG":
            t_val = -1.0
            neg_c += 1
        else:
            t_val = -1.0
            unc_c += 1

        nb_tone_sum += t_val
        if pred_c == "PROFIT":
            prof_tones.append(t_val)
        elif pred_c == "LIQUIDITY":
            liq_tones.append(t_val)
        else:
            oth_tones.append(t_val)

    tone_all = round(nb_tone_sum / n_fls, 4)
    # Li (2010, p. 1064): require at least 5 FLS sentences for main TONE measure
    tone_min5 = tone_all if n_fls >= 5 else np.nan

    return {
        "fls_count": n_fls,
        "fls_share": round(n_fls / len(sents), 4),
        "fls_tone": tone_min5,
        "fls_tone_all": tone_all,
        "fls_tone_dict": round(dict_tone_sum / n_fls, 4) if n_fls >= 5 else np.nan,
        "fls_profit_tone": round(float(np.mean(prof_tones)), 4) if prof_tones and n_fls >= 5 else np.nan,
        "fls_liquidity_tone": round(float(np.mean(liq_tones)), 4) if liq_tones and n_fls >= 5 else np.nan,
        "fls_other_tone": round(float(np.mean(oth_tones)), 4) if oth_tones and n_fls >= 5 else np.nan,
        "fls_profit_pct": round(len(prof_tones) / n_fls, 4),
        "fls_liquidity_pct": round(len(liq_tones) / n_fls, 4),
        "fls_other_pct": round(len(oth_tones) / n_fls, 4),
        "fls_pos_share": round(pos_c / n_fls, 4),
        "fls_neu_share": round(neu_c / n_fls, 4),
        "fls_neg_share": round(neg_c / n_fls, 4),
        "fls_unc_share": round(unc_c / n_fls, 4),
    }


def process_single_row(row_dict):
    """Pass 1 worker: construct exact disjoint text partitions and compute all within-year metrics."""
    src_rel = row_dict["source_path"]
    full_path = os.path.join(AR_DIR, src_rel)

    lines = []
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()
    n_lines = len(lines)

    # Construct 1-based line masks for exact disjoint section partitioning
    in_risk = bytearray(n_lines + 1)
    in_mda = bytearray(n_lines + 1)
    in_notice = bytearray(n_lines + 1)

    if row_dict.get("risk_found"):
        for s, e in parse_spans_str(row_dict.get("risk_spans", ""), n_lines):
            for k in range(s, e + 1):
                in_risk[k] = 1

    if row_dict.get("mda_found") and pd.notna(row_dict.get("mda_start_line")) and pd.notna(row_dict.get("mda_end_line")):
        m_s = max(1, int(row_dict["mda_start_line"]))
        m_e = min(n_lines, int(row_dict["mda_end_line"]))
        for k in range(m_s, m_e + 1):
            in_mda[k] = 1

    for s, e in parse_spans_str(row_dict.get("notice_spans", ""), n_lines):
        for k in range(s, e + 1):
            in_notice[k] = 1

    # Single-pass line cleaning across the entire report (4x faster than cleaning overlapping slices)
    risk_buf, mda_full_buf, mda_ex_risk_buf, other_buf, full_ex_1a_buf, km_body_buf = [], [], [], [], [], []
    raw_w = [0, 0, 0, 0]   # [risk, mda, other, body]
    garb_w = [0, 0, 0, 0]
    cln_w = [0, 0, 0, 0]
    sw_w = [0, 0, 0, 0]

    for k in range(1, n_lines + 1):
        if in_notice[k]:
            continue
        s = lines[k - 1].strip()
        if not s or s.startswith("|") or s.startswith("![") or s.startswith("<!--"):
            continue
        is_heading = s.startswith("#") or (s.startswith("**") and s.endswith("**") and len(s) <= 140)
        s = IMG_OR_HTML_RE.sub(" ", s)
        s = MD_LINK_RE.sub(r"\1", s)
        s = SYM_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue

        ws = ALPHA_WORD_RE.findall(s)
        n_w = len(ws)
        sw_cnt = sum(1 for w in ws if w.lower() in NLTK_STOPWORDS)
        is_garb = False
        if n_w >= 8:
            novow_low = sum(1 for w in ws if len(w) >= 3 and w.islower() and not VOWEL_GROUP_RE.search(w))
            if (sw_cnt / n_w < 0.06 and novow_low >= 1) or (novow_low / n_w > 0.25):
                is_garb = True

        r_flag = in_risk[k]
        m_flag = in_mda[k]
        o_flag = (not r_flag) and (not m_flag)

        raw_w[3] += n_w
        if r_flag:
            raw_w[0] += n_w
        if m_flag:
            raw_w[1] += n_w
        if o_flag:
            raw_w[2] += n_w

        if is_garb:
            garb_w[3] += n_w
            if r_flag:
                garb_w[0] += n_w
            if m_flag:
                garb_w[1] += n_w
            if o_flag:
                garb_w[2] += n_w
            continue

        cln_w[3] += n_w
        sw_w[3] += sw_cnt
        if r_flag:
            cln_w[0] += n_w
            sw_w[0] += sw_cnt
        if m_flag:
            cln_w[1] += n_w
            sw_w[1] += sw_cnt
        if o_flag:
            cln_w[2] += n_w
            sw_w[2] += sw_cnt

        if is_heading and s[-1] not in ".!?:;":
            s = s + "."

        km_body_buf.append(s)
        if r_flag:
            risk_buf.append(s)
        else:
            full_ex_1a_buf.append(s)
        if m_flag:
            mda_full_buf.append(s)
            if not r_flag:
                mda_ex_risk_buf.append(s)
        elif not r_flag:
            other_buf.append(s)

    def _check_cmap_bad(idx_b):
        if raw_w[idx_b] < 30:
            return False
        if garb_w[idx_b] / raw_w[idx_b] > 0.25:
            return True
        if cln_w[idx_b] >= 30 and (sw_w[idx_b] / cln_w[idx_b]) < 0.18:
            return True
        return False

    risk_cmap_bad = _check_cmap_bad(0)
    mda_cmap_bad = _check_cmap_bad(1)
    other_cmap_bad = _check_cmap_bad(2)
    report_cmap_bad = _check_cmap_bad(3)

    risk_txt = "" if risk_cmap_bad else " ".join(risk_buf)
    mda_full_txt = "" if mda_cmap_bad else " ".join(mda_full_buf)
    mda_ex_risk_txt = "" if mda_cmap_bad else " ".join(mda_ex_risk_buf)
    other_txt = "" if other_cmap_bad else " ".join(other_buf)
    full_ex_1a_txt = "" if report_cmap_bad else " ".join(full_ex_1a_buf)
    km_body_txt = "" if report_cmap_bad else " ".join(km_body_buf)

    # 1. Word Counts, Sentence Counts, and Gunning Fog Readability (Campbell 2014 / Beatty 2019)
    r_words, r_sents, _, r_fog = compute_fog_and_words(risk_txt)
    m_full_words, m_full_sents, _, m_full_fog = compute_fog_and_words(mda_full_txt)
    m_ex_words, _, _, m_ex_fog = compute_fog_and_words(mda_ex_risk_txt)
    o_words, _, _, o_fog = compute_fog_and_words(other_txt)
    f_words, _, _, f_fog = compute_fog_and_words(km_body_txt)

    # Use MDA_ExRisk when non-empty (>=20 words), else fallback if MD&A was 100% risk
    use_m_words = m_ex_words if m_ex_words >= 20 else m_full_words
    use_m_fog = m_ex_fog if pd.notna(m_ex_fog) else m_full_fog

    # 2. Campbell et al. (2014) 5-Category Non-Overlapping Risk Keywords
    c_risk = count_campbell_keywords(risk_txt, check_india_ext=True)
    c_mda_ex = count_campbell_keywords(mda_ex_risk_txt, check_india_ext=False)
    c_mda_full = count_campbell_keywords(mda_full_txt, check_india_ext=False)
    c_other = count_campbell_keywords(other_txt, check_india_ext=False)

    rf_tot = c_risk["RF_TOTAL"]
    rf_fin_pct = round(c_risk["RF_FIN"] / rf_tot, 4) if rf_tot > 0 else np.nan
    rf_lr_pct = round(c_risk["RF_LR"] / rf_tot, 4) if rf_tot > 0 else np.nan
    rf_tax_pct = round(c_risk["RF_TAX"] / rf_tot, 4) if rf_tot > 0 else np.nan
    rf_sys_pct = round(c_risk["RF_SYS"] / rf_tot, 4) if rf_tot > 0 else np.nan
    rf_idio_pct = round(c_risk["RF_IDIO"] / rf_tot, 4) if rf_tot > 0 else np.nan

    # 3. Hope, Hu & Lu (2016) Statistical NER Specificity (`Risk_Section` and `Rest of 10-K`)
    spec_1a = compute_hope_specificity(risk_txt, prefilter_long_text=False) if r_words >= 15 else {
        "spec_money": 0, "spec_pct": 0, "spec_date": 0, "spec_time": 0,
        "spec_org": 0, "spec_loc": 0, "spec_person": 0, "spec_words": 0,
        "nonstop_words": 0, "specificity": np.nan,
    }
    spec_10k = compute_hope_specificity(full_ex_1a_txt, prefilter_long_text=True) if f_words >= 50 else {
        "specificity": np.nan,
    }

    # 4. Kravet & Muslu (2013) Sentence-Level Risk Disclosures & Section 4.6 Negative RD
    if f_words >= 50:
        km_sents_list = [s for s in split_sentences_safe(km_body_txt) if len(s) >= 15]
        km_risk_sents = 0
        km_neg_risk_sents = 0
        for s in km_sents_list:
            if KM_RISK_SENT_RE.search(s):
                km_risk_sents += 1
                if KM_NEG_MOD_RE.search(s):
                    km_neg_risk_sents += 1
        km_nonrisk_sents = max(0, len(km_sents_list) - km_risk_sents)
        km_other_risk_sents = km_risk_sents - km_neg_risk_sents
        km_total_sents = len(km_sents_list)
        li2006_words = len(LI2006_RISK_RE.findall(km_body_txt))
        li2006_log_words = round(math.log(1 + li2006_words), 4)
    else:
        km_risk_sents = np.nan
        km_neg_risk_sents = np.nan
        km_other_risk_sents = np.nan
        km_nonrisk_sents = np.nan
        km_total_sents = np.nan
        li2006_words = np.nan
        li2006_log_words = np.nan

    # 5. Li (2010) Forward-Looking Statements & Naïve Bayesian Tone/Content in MD&A
    li_fls = compute_li2010_fls(mda_full_txt)

    # 6. Brown & Tucker (2011) Porter-Stemmed Bag-of-Words:
    #    - Primary spec (p. 316 & Footnote 10): ALL words Porter-stemmed, NO stopword removal (pure IDF)
    #    - Footnote 10 robustness spec: Porter-stemmed AFTER English stopword removal (pure TF without IDF)
    if m_full_words >= 50:
        mda_low_toks = [w.lower() for w in ALPHA_WORD_RE.findall(mda_full_txt)]
        mda_bow = Counter(stem_word(w) for w in mda_low_toks)
        mda_bow_sw = Counter(stem_word(w) for w in mda_low_toks if w not in NLTK_STOPWORDS)
    else:
        mda_bow = Counter()
        mda_bow_sw = Counter()

    risk_bow = (
        Counter(stem_word(w.lower()) for w in ALPHA_WORD_RE.findall(risk_txt))
        if r_words >= 20
        else Counter()
    )

    out = dict(row_dict)
    out.update({
        # Section clean narrative word counts & PDF CMap font-encoding QC flags
        "mda_narrative_words": m_full_words,
        "mda_ex_risk_words": m_ex_words,
        "risk_narrative_words": r_words,
        "other_disjoint_words": o_words,
        "mda_cmap_corrupted": int(mda_cmap_bad),
        "risk_cmap_corrupted": int(risk_cmap_bad),
        "report_cmap_corrupted": int(report_cmap_bad),

        # Campbell et al. (2014) & Beatty et al. (2019) Level Variables
        "ALL_WORDS": round(math.log(1 + r_words), 4) if r_words > 0 else np.nan,
        "KEY_WORDS": round(math.log(1 + rf_tot), 4) if r_words > 0 else np.nan,
        "KW_1A": round(rf_tot / 1000.0, 4) if r_words > 0 else np.nan,
        "WC_1A": round(r_words / 1000.0, 4) if r_words > 0 else np.nan,
        "KW_1A_RAW": rf_tot,
        "KW_1A_FIN": c_risk["RF_FIN"],
        "KW_1A_LR": c_risk["RF_LR"],
        "KW_1A_TAX": c_risk["RF_TAX"],
        "KW_1A_SYS": c_risk["RF_SYS"],
        "KW_1A_IDIO": c_risk["RF_IDIO"],
        "KW_1A_INDIA_EXT": c_risk["IND_EXT"],
        "RF_FIN": rf_fin_pct,
        "RF_LR": rf_lr_pct,
        "RF_TAX": rf_tax_pct,
        "RF_SYS": rf_sys_pct,
        "RF_IDIO": rf_idio_pct,
        "MDA_DISC_WORDS": round(math.log(1 + use_m_words), 4) if use_m_words > 0 else np.nan,
        "MDA_DISC_KW": round(math.log(1 + c_mda_ex["RF_TOTAL"]), 4) if m_full_words > 0 else np.nan,
        "KW_MDA": round(c_mda_ex["RF_TOTAL"] / 1000.0, 4) if m_full_words > 0 else np.nan,
        "KW_MDA_Full": round(c_mda_full["RF_TOTAL"] / 1000.0, 4) if m_full_words > 0 else np.nan,
        "WC_MDA": round(use_m_words / 1000.0, 4) if use_m_words > 0 else np.nan,
        "KW_OTHER": round(c_other["RF_TOTAL"] / 1000.0, 4) if o_words > 0 else np.nan,
        "WC_OTHER": round(o_words / 1000.0, 4) if o_words > 0 else np.nan,
        "ALL_DISC": round(math.log(1 + f_words), 4) if f_words > 0 else np.nan,
        "RF_FOG": r_fog,
        "MDA_FOG": use_m_fog,
        "FOG_OTHER": o_fog,
        "ALL_FOG": f_fog,

        # Hope, Hu & Lu (2016) Statistical NER Specificity Variables
        "Specificity_1A": spec_1a["specificity"],
        "Specificity_10K": spec_10k["specificity"],
        "Spec_1A_Specific_Words": spec_1a["spec_words"],
        "Spec_1A_NonStop_Words": spec_1a["nonstop_words"],
        "Spec_1A_Money": spec_1a["spec_money"],
        "Spec_1A_Percent": spec_1a["spec_pct"],
        "Spec_1A_Date": spec_1a["spec_date"],
        "Spec_1A_Time": spec_1a["spec_time"],
        "Spec_1A_Org": spec_1a["spec_org"],
        "Spec_1A_Loc": spec_1a["spec_loc"],
        "Spec_1A_Person": spec_1a["spec_person"],

        # Kravet & Muslu (2013) & Li (2006) Variables
        "KM_Risk_Disclosure": km_risk_sents,
        "KM_Negative_RD": km_neg_risk_sents,
        "KM_Other_RD": km_other_risk_sents,
        "KM_Non_Risk_Disclosure": km_nonrisk_sents,
        "KM_Total_Sentences": km_total_sents,
        "Li2006_Risk_Words": li2006_words,
        "Li2006_Log_Risk_Words": li2006_log_words,

        # Li (2010) Forward-Looking Statements & Naïve Bayesian Variables
        "FLS_Count": li_fls["fls_count"],
        "FLS_Share": li_fls["fls_share"],
        "FLS_TONE": li_fls["fls_tone"],
        "FLS_TONE_All": li_fls["fls_tone_all"],
        "FLS_TONE_Dict": li_fls["fls_tone_dict"],
        "FLS_PROFIT_TONE": li_fls["fls_profit_tone"],
        "FLS_LIQUIDITY_TONE": li_fls["fls_liquidity_tone"],
        "FLS_OTHER_TONE": li_fls["fls_other_tone"],
        "FLS_PROFIT_PCT": li_fls["fls_profit_pct"],
        "FLS_LIQUIDITY_PCT": li_fls["fls_liquidity_pct"],
        "FLS_OTHER_PCT": li_fls["fls_other_pct"],
        "FLS_Pos_Share": li_fls["fls_pos_share"],
        "FLS_Neu_Share": li_fls["fls_neu_share"],
        "FLS_Neg_Share": li_fls["fls_neg_share"],
        "FLS_Unc_Share": li_fls["fls_unc_share"],

        # Temporary objects for Pass 2 longitudinal pairs (removed before CSV export)
        "_mda_bow": mda_bow,
        "_mda_bow_sw": mda_bow_sw,
        "_risk_bow": risk_bow,
        "_risk_txt": risk_txt,
    })
    return out


def compute_tf_cosine_rawscore(bow1, bow2):
    """Brown & Tucker (2011) Footnote 10 alternative Rawscore: pure TF without IDF after stopword removal."""
    if not bow1 or not bow2:
        return np.nan
    dot = sum(c1 * bow2[w] for w, c1 in bow1.items() if w in bow2)
    norm1 = sum(c1 * c1 for c1 in bow1.values())
    norm2 = sum(c2 * c2 for c2 in bow2.values())
    if norm1 <= 0 or norm2 <= 0:
        return np.nan
    cos_sim = max(0.0, min(1.0, dot / (math.sqrt(norm1) * math.sqrt(norm2))))
    return round(1.0 - cos_sim, 5)


def compute_tfidf_cosine_rawscore(bow1, bow2, idf_map):
    """Brown & Tucker (2011) Rawscore = 1 - cos(theta) between two Porter-stemmed TF-IDF vectors."""
    if not bow1 or not bow2:
        return np.nan
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for w, c1 in bow1.items():
        w_idf = idf_map.get(w, 0.0)
        v1 = c1 * w_idf
        norm1 += v1 * v1
        c2 = bow2.get(w, 0)
        if c2:
            dot += v1 * (c2 * w_idf)
    for w, c2 in bow2.items():
        w_idf = idf_map.get(w, 0.0)
        v2 = c2 * w_idf
        norm2 += v2 * v2
    if norm1 <= 0.0 or norm2 <= 0.0:
        return np.nan
    cos_sim = dot / (math.sqrt(norm1) * math.sqrt(norm2))
    cos_sim = max(0.0, min(1.0, cos_sim))
    return round(1.0 - cos_sim, 5)


def length_adjust_poly5(rawscore_series, length_series):
    """
    Brown & Tucker (2011) Appendix B (Table B1, p. 344): regress Rawscore on a 5th-order
    Taylor polynomial of current-year document word count (Length_t^1..5) and return residual `Score`.
    """
    valid = rawscore_series.notna() & length_series.notna() & (length_series > 0)
    scores = pd.Series(np.nan, index=rawscore_series.index, dtype=float)
    if valid.sum() < 30:
        return scores
    y = rawscore_series[valid].values.astype(float)
    # Scale Length by 1,000 for numerical conditioning (exact linear span is invariant to scaling)
    L = (length_series[valid].values.astype(float)) / 1000.0
    X = np.column_stack([np.ones_like(L), L, L**2, L**3, L**4, L**5])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    scores.loc[valid] = np.round(resid, 5)
    return scores


def _join_masked_contiguous_runs(tokens, mask, want_val=1):
    """
    Join contiguous runs of tokens where mask[i] == want_val, separating non-adjacent runs
    with ' ; ' so deleted/new gaps NEVER create false cross-gap multi-word phrase matches!
    """
    runs = []
    cur = []
    for tok, m in zip(tokens, mask):
        if m == want_val:
            cur.append(tok)
        else:
            if cur:
                runs.append(" ".join(cur))
                cur = []
    if cur:
        runs.append(" ".join(cur))
    return " ; ".join(runs)


def decompose_beatty_wcopyfind(text_prev, text_curr, kw_tot_prev, kw_tot_curr):
    """
    Exact Beatty, Cheng & Zhang (2019, CAR) Section 4 `WCopyfind v4.1.5` 6-word phrase
    decomposition (p. 145 & Figures 2-3):
      - Extract contiguous runs of >=6-word shared phrases into separate repeated documents
        (separated by ' ; ' so non-adjacent phrases never glue across gaps).
      - Following Beatty et al. (p. 145) verbatim:
          KW_1A_NEW  = KW_1A_t     - KW_1A_REP_t
          KW_1A_DISC = KW_1A_{t-1} - KW_1A_REP_{t-1}
        guaranteeing exact additivity `KW_1A_REP + KW_1A_NEW == KW_1A` for 100% of observations.
      - Also compute word-count equivalents (`WC_1A_REP`, `WC_1A_NEW`, `WC_1A_DISC`, `PCT_REPEATED_WC`).
    """
    nan8 = (np.nan,) * 8
    if not text_prev or not text_curr:
        return nan8

    words_prev = ALNUM_WORD_RE.findall(text_prev)
    words_curr = ALNUM_WORD_RE.findall(text_curr)
    if len(words_prev) < 10 or len(words_curr) < 10:
        return nan8

    low_prev = [w.lower() for w in words_prev]
    low_curr = [w.lower() for w in words_curr]

    K = 6
    shingles_prev = {tuple(low_prev[i : i + K]) for i in range(len(low_prev) - K + 1)}
    shingles_curr = {tuple(low_curr[i : i + K]) for i in range(len(low_curr) - K + 1)}
    shared = shingles_prev & shingles_curr

    rep_mask_curr = bytearray(len(words_curr))
    for i in range(len(low_curr) - K + 1):
        if tuple(low_curr[i : i + K]) in shared:
            for j in range(i, i + K):
                rep_mask_curr[j] = 1

    rep_mask_prev = bytearray(len(words_prev))
    for i in range(len(low_prev) - K + 1):
        if tuple(low_prev[i : i + K]) in shared:
            for j in range(i, i + K):
                rep_mask_prev[j] = 1

    rep_text_curr = _join_masked_contiguous_runs(words_curr, rep_mask_curr, 1)
    rep_text_prev = _join_masked_contiguous_runs(words_prev, rep_mask_prev, 1)

    kw_rep_curr = min(int(kw_tot_curr), count_campbell_keywords(rep_text_curr)["RF_TOTAL"])
    kw_new_curr = max(0, int(kw_tot_curr) - kw_rep_curr)

    kw_rep_prev = min(int(kw_tot_prev), count_campbell_keywords(rep_text_prev)["RF_TOTAL"])
    kw_disc_curr = max(0, int(kw_tot_prev) - kw_rep_prev)

    wc_rep_curr = sum(rep_mask_curr)
    wc_new_curr = len(words_curr) - wc_rep_curr
    wc_disc_curr = len(words_prev) - sum(rep_mask_prev)

    pct_rep_wc = round(wc_rep_curr / len(words_curr), 4)
    pct_rep_kw = round(kw_rep_curr / kw_tot_curr, 4) if kw_tot_curr > 0 else pct_rep_wc

    return (
        round(kw_rep_curr / 1000.0, 4),
        round(kw_new_curr / 1000.0, 4),
        round(kw_disc_curr / 1000.0, 4),
        pct_rep_kw,
        round(wc_rep_curr / 1000.0, 4),
        round(wc_new_curr / 1000.0, 4),
        round(wc_disc_curr / 1000.0, 4),
        pct_rep_wc,
    )


def compute_beatty_2factors(df):
    """
    Beatty et al. (2019) Section 5 & Appendix: extract 2 varimax-rotated principal factors
    from the 5 Campbell risk subcategories (`KW_1A_FIN`, `KW_1A_SYS`, `KW_1A_IDIO`, `KW_1A_LR`, `KW_1A_TAX`)
    capturing `FIN_SYS_1A` and `IDIO_LR_1A`.
    """
    cols = ["KW_1A_FIN", "KW_1A_SYS", "KW_1A_IDIO", "KW_1A_LR", "KW_1A_TAX"]
    valid = (df["risk_narrative_words"] > 0) & df[cols].notna().all(axis=1)
    f1 = pd.Series(np.nan, index=df.index, dtype=float)
    f2 = pd.Series(np.nan, index=df.index, dtype=float)
    if valid.sum() < 50:
        return f1, f2

    X = df.loc[valid, cols].values.astype(float) / 1000.0
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
    Z = (X - mu) / sd
    cov = np.cov(Z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    loadings = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 1e-9))

    # Varimax rotation on 2 factors
    L = loadings.copy()
    for _ in range(30):
        u, s, vt = np.linalg.svd(
            L.T @ (L**3 - (1.0 / L.shape[0]) * L @ np.diag(np.sum(L**2, axis=0)))
        )
        R = u @ vt
        L_new = L @ R
        if np.max(np.abs(L_new - L)) < 1e-7:
            L = L_new
            break
        L = L_new

    # Ensure positive sign orientation and match Factor 1 -> FIN+SYS, Factor 2 -> IDIO+LR
    for j in range(2):
        if L[:, j].sum() < 0:
            L[:, j] *= -1.0
    # Check which column loads more strongly on FIN(0)+SYS(1) vs IDIO(2)+LR(3)
    if (L[0, 1] + L[1, 1]) > (L[0, 0] + L[1, 0]):
        L = L[:, [1, 0]]

    # Regression factor scores: Z @ inv(cov) @ L
    inv_cov = np.linalg.pinv(cov)
    scores = Z @ inv_cov @ L
    f1.loc[valid] = np.round(scores[:, 0], 4)
    f2.loc[valid] = np.round(scores[:, 1], 4)
    return f1, f2


def main():
    parser = argparse.ArgumentParser(description="Step 1 (v2 - Audited): Compute 6-paper panel")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of valid rows (for testing)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Parallel worker processes")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    if os.path.exists(LI_NB_MODEL_JSON):
        print(f"[Step 1 v2] Found trained Li (2010) Naïve Bayesian classifier: {LI_NB_MODEL_JSON}")
    else:
        print(f"[Step 1 v2] WARNING: {LI_NB_MODEL_JSON} not found yet — using dictionary fallback.")

    manifest = pd.read_csv(MANIFEST_CSV, low_memory=False)
    ok_df = manifest[manifest["status"] == "ok"].copy()
    ok_df["year_int"] = pd.to_numeric(ok_df["year"], errors="coerce")
    ok_df = ok_df.sort_values(["bse_code", "year_int"]).reset_index(drop=True)

    if args.limit > 0:
        ok_df = ok_df.head(args.limit).copy()

    rows = ok_df.to_dict(orient="records")
    print(f"[Step 1 v2 - Pass 1] Computing audited within-year metrics + spaCy NER + NB FLS on {len(rows):,} reports ({args.workers} workers)...")

    pass1_results = []
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(LI_NB_MODEL_JSON,)) as pool:
        for idx, res in enumerate(pool.imap(process_single_row, rows, chunksize=12), 1):
            pass1_results.append(res)
            if idx % 1000 == 0 or idx == len(rows):
                el = time.time() - t0
                print(f"  Pass 1: {idx:,}/{len(rows):,} reports ({idx / max(0.1, el):.1f} reports/sec, {el:.1f}s elapsed)...")

    # -----------------------------------------------------------------------
    # Pass 2: Corpus-Wide & Cohort IDF + Longitudinal Year-over-Year Metrics
    # -----------------------------------------------------------------------
    print("[Step 1 v2 - Pass 2] Computing Porter-stemmed Pure IDF, Brown & Tucker (2011) Cosine Scores, and Beatty et al. (2019) WCopyfind decompositions...")

    df_mda_docfreq = Counter()
    df_risk_docfreq = Counter()
    cohort_mda_docfreq = defaultdict(Counter)
    cohort_mda_ndocs = Counter()
    n_mda_docs = 0
    n_risk_docs = 0

    for r in pass1_results:
        yr_i = int(r["year_int"]) if pd.notna(r["year_int"]) else 0
        if r["_mda_bow"]:
            n_mda_docs += 1
            df_mda_docfreq.update(r["_mda_bow"].keys())
            cohort_mda_ndocs[yr_i] += 1
            cohort_mda_docfreq[yr_i].update(r["_mda_bow"].keys())
        if r["_risk_bow"]:
            n_risk_docs += 1
            df_risk_docfreq.update(r["_risk_bow"].keys())

    # Exact Brown & Tucker (2011) p. 316-317: IDF = ln(M / m_k); word in all M docs gets ln(1)=0.0!
    mda_idf = {w: math.log(max(1, n_mda_docs) / df_c) for w, df_c in df_mda_docfreq.items()}
    risk_idf = {w: math.log(max(1, n_risk_docs) / df_c) for w, df_c in df_risk_docfreq.items()}
    cohort_mda_idf = {
        yr_i: {w: math.log(max(1, cohort_mda_ndocs[yr_i]) / df_c) for w, df_c in cnt.items()}
        for yr_i, cnt in cohort_mda_docfreq.items()
    }

    pass1_results.sort(key=lambda x: (str(x["bse_code"]), int(x["year_int"]) if pd.notna(x["year_int"]) else 0))

    yoy_nan_cols = [
        "MDA_Rawscore", "MDA_Rawscore_CohortIDF", "MDA_Rawscore_TF", "Risk_Rawscore",
        "KW_1A_REP", "KW_1A_NEW", "KW_1A_DISC", "PCT_REPEATED_KW",
        "WC_1A_REP", "WC_1A_NEW", "WC_1A_DISC", "PCT_REPEATED_WC",
        "Delta_KW_1A", "Delta_KW_MDA", "Delta_KW_OTHER",
        "Delta_WC_1A", "Delta_WC_MDA", "Delta_WC_OTHER",
        "Delta_FOG_1A", "Delta_FOG_MDA", "Delta_FOG_OTHER",
        "KM_Delta_RD", "KM_Delta_Negative_RD", "KM_Delta_Other_RD", "KM_Delta_Non_RD",
        "Li2006_Delta_Log_Risk",
    ]

    for i, r in enumerate(pass1_results):
        for c in yoy_nan_cols:
            r[c] = np.nan

        if i == 0:
            continue
        prev = pass1_results[i - 1]
        if str(prev["bse_code"]) != str(r["bse_code"]):
            continue
        yr_diff = (r["year_int"] - prev["year_int"]) if (pd.notna(r["year_int"]) and pd.notna(prev["year_int"])) else 99
        if yr_diff != 1:
            continue

        # 1. Brown & Tucker (2011) Rawscore:
        #    - Primary: Porter-stemmed, pure IDF, no stopword removal
        #    - Footnote 10: Porter-stemmed, stopword-removed TF without IDF
        if prev["_mda_bow"] and r["_mda_bow"]:
            r["MDA_Rawscore"] = compute_tfidf_cosine_rawscore(prev["_mda_bow"], r["_mda_bow"], mda_idf)
            r["MDA_Rawscore_TF"] = compute_tf_cosine_rawscore(prev["_mda_bow_sw"], r["_mda_bow_sw"])
            yr_i = int(r["year_int"])
            if cohort_mda_ndocs[yr_i] >= 20:
                r["MDA_Rawscore_CohortIDF"] = compute_tfidf_cosine_rawscore(
                    prev["_mda_bow"], r["_mda_bow"], cohort_mda_idf[yr_i]
                )
        if prev["_risk_bow"] and r["_risk_bow"]:
            r["Risk_Rawscore"] = compute_tfidf_cosine_rawscore(prev["_risk_bow"], r["_risk_bow"], risk_idf)

        # 2. Beatty, Cheng & Zhang (2019) WCopyfind 6-word phrase decomposition + Section Changes
        if prev["_risk_txt"] and r["_risk_txt"]:
            (
                kw_rep, kw_new, kw_disc, pct_rep_kw,
                wc_rep, wc_new, wc_disc, pct_rep_wc
            ) = decompose_beatty_wcopyfind(
                prev["_risk_txt"], r["_risk_txt"], prev["KW_1A_RAW"], r["KW_1A_RAW"]
            )
            r["KW_1A_REP"] = kw_rep
            r["KW_1A_NEW"] = kw_new
            r["KW_1A_DISC"] = kw_disc
            r["PCT_REPEATED_KW"] = pct_rep_kw
            r["WC_1A_REP"] = wc_rep
            r["WC_1A_NEW"] = wc_new
            r["WC_1A_DISC"] = wc_disc
            r["PCT_REPEATED_WC"] = pct_rep_wc

        for col_lvl, col_d, ndig in [
            ("KW_1A", "Delta_KW_1A", 4),
            ("KW_MDA", "Delta_KW_MDA", 4),
            ("KW_OTHER", "Delta_KW_OTHER", 4),
            ("WC_1A", "Delta_WC_1A", 4),
            ("WC_MDA", "Delta_WC_MDA", 4),
            ("WC_OTHER", "Delta_WC_OTHER", 4),
            ("RF_FOG", "Delta_FOG_1A", 3),
            ("MDA_FOG", "Delta_FOG_MDA", 3),
            ("FOG_OTHER", "Delta_FOG_OTHER", 3),
        ]:
            if pd.notna(r[col_lvl]) and pd.notna(prev[col_lvl]):
                r[col_d] = round(r[col_lvl] - prev[col_lvl], ndig)

        # 3. Kravet & Muslu (2013) & Li (2006) Annual Changes (only when neither year was CMap-corrupted)
        if pd.notna(r["KM_Risk_Disclosure"]) and pd.notna(prev["KM_Risk_Disclosure"]):
            r["KM_Delta_RD"] = r["KM_Risk_Disclosure"] - prev["KM_Risk_Disclosure"]
            r["KM_Delta_Negative_RD"] = r["KM_Negative_RD"] - prev["KM_Negative_RD"]
            r["KM_Delta_Other_RD"] = r["KM_Other_RD"] - prev["KM_Other_RD"]
            r["KM_Delta_Non_RD"] = r["KM_Non_Risk_Disclosure"] - prev["KM_Non_Risk_Disclosure"]
            r["Li2006_Delta_Log_Risk"] = round(r["Li2006_Log_Risk_Words"] - prev["Li2006_Log_Risk_Words"], 4)

    for r in pass1_results:
        r.pop("_mda_bow", None)
        r.pop("_mda_bow_sw", None)
        r.pop("_risk_bow", None)
        r.pop("_risk_txt", None)

    panel = pd.DataFrame(pass1_results)

    # 4. Brown & Tucker (2011) 5th-order Taylor polynomial length adjustment on current-year narrative word count
    panel["MDA_Score"] = length_adjust_poly5(panel["MDA_Rawscore"], panel["mda_narrative_words"])
    panel["MDA_Score_TF"] = length_adjust_poly5(panel["MDA_Rawscore_TF"], panel["mda_narrative_words"])
    panel["Risk_Score"] = length_adjust_poly5(panel["Risk_Rawscore"], panel["risk_narrative_words"])

    # 5. Beatty et al. (2019) 2-Factor Risk Subcategory Scores (`FIN_SYS_1A`, `IDIO_LR_1A`)
    panel["FIN_SYS_1A"], panel["IDIO_LR_1A"] = compute_beatty_2factors(panel)

    # 6. Kravet & Muslu (2013) Industry-Level vs Firm-Level RD Decomposition (require >=3 peer firms in cell!)
    panel["NIC_2digit"] = (
        panel["NIC code"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan}).str[:2]
    )
    valid_km = panel["KM_Delta_RD"].notna() & panel["NIC_2digit"].notna()
    cell_counts = panel.loc[valid_km].groupby(["NIC_2digit", "year_int"])["KM_Delta_RD"].transform("count")
    cell_medians = panel.loc[valid_km].groupby(["NIC_2digit", "year_int"])["KM_Delta_RD"].transform("median")

    panel["KM_Industry_Peer_N"] = np.nan
    panel.loc[valid_km, "KM_Industry_Peer_N"] = cell_counts
    panel["KM_Delta_Industry_RD"] = np.nan
    ge3_idx = cell_counts[cell_counts >= 3].index
    panel.loc[ge3_idx, "KM_Delta_Industry_RD"] = np.round(cell_medians.loc[ge3_idx], 3)
    panel["KM_Delta_Firm_RD"] = np.round(panel["KM_Delta_RD"] - panel["KM_Delta_Industry_RD"], 3)

    # 7. Regulatory / Crisis Regime Indicators
    panel["POST_2009_Crisis"] = (panel["year_int"] >= 2009).astype(int)
    panel["POST_2015_LODR_CA2013"] = (panel["year_int"] >= 2015).astype(int)

    if args.limit == 0:
        out_csv = os.path.join(OUT_DIR, "india_corporate_risk_disclosure_panel_1997_2025.csv")
        panel.to_csv(out_csv, index=False)
    else:
        out_csv = os.path.join(OUT_DIR, f"test_panel_limit_{args.limit}.csv")
        panel.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"\n[Step 1 v2 Complete in {elapsed:.1f}s]")
    print(f"  Saved panel CSV ({len(panel):,} rows x {len(panel.columns)} columns): {out_csv}")
    summary_cols = [
        ("MDA_Rawscore", "Brown & Tucker (2011) Raw Cosine Dissimilarity (Porter + Pure IDF)"),
        ("MDA_Rawscore_TF", "Brown & Tucker (2011, Fn 10) Raw Cosine Dissimilarity (Porter + Stopword-Removed TF)"),
        ("MDA_Score", "Brown & Tucker (2011) Length-Adjusted Score"),
        ("KW_1A", "Campbell/Beatty Risk Keyword Count (/1,000)"),
        ("RF_SYS", "Campbell (2014) Systematic Risk Share"),
        ("RF_IDIO", "Campbell (2014) Idiosyncratic Risk Share"),
        ("RF_FIN", "Campbell (2014) Financial Risk Share"),
        ("RF_LR", "Campbell (2014) Legal & Regulatory Risk Share"),
        ("RF_TAX", "Campbell (2014) Tax Risk Share"),
        ("KW_1A_REP", "Beatty et al. (2019) Repeated Risk Keywords (/1,000)"),
        ("KW_1A_NEW", "Beatty et al. (2019) New Risk Keywords (/1,000)"),
        ("PCT_REPEATED_WC", "Beatty et al. (2019) % Repeated Risk Words"),
        ("Specificity_1A", "Hope et al. (2016) 7-Class NER Specificity (Risk)"),
        ("Specificity_10K", "Hope et al. (2016) 7-Class NER Specificity (Rest of Report)"),
        ("KM_Risk_Disclosure", "Kravet & Muslu (2013) Risk Sentences"),
        ("KM_Delta_RD", "Kravet & Muslu (2013) Annual Change in Risk Sentences"),
        ("FLS_TONE", "Li (2010) Naive Bayes FLS Tone [-1, +1]"),
        ("FLS_PROFIT_PCT", "Li (2010) Naive Bayes Profit/Ops Content Share"),
        ("FLS_LIQUIDITY_PCT", "Li (2010) Naive Bayes Liquidity/Cap Content Share"),
    ]
    for col, desc in summary_cols:
        s = panel[col].dropna()
        if len(s) > 0:
            print(f"    {col:22s} | N={len(s):6,d} | mean={s.mean():8.4f} | sd={s.std():8.4f} | p50={s.median():8.4f} | {desc}")


if __name__ == "__main__":
    main()
