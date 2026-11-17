#!/usr/bin/env python3
"""
Step 0 (v2 - Audited): Automated Section Segmentation for Indian Annual Reports (1997-2025)
===========================================================================================
Partitions each <BSE_Scrip_Code>/markdown/<Year>.md in:
    ./annual_reports_corpus
into structured, non-overlapping sections required by the 6 benchmark papers
(Li 2010; Brown & Tucker 2011; Kravet & Muslu 2013; Campbell et al. 2014;
Hope, Hu & Lu 2016; Beatty, Cheng & Zhang 2019):

  - Block A (MDA_Full): Management Discussion & Analysis section
      -> step0_sections/mda/<bse_code>/<year>_mda.md
  - Block B (Risk_Section): Substantive Narrative Risk & Threat subsections (India's Item 1A
      equivalent: "Risks and Concerns", "Opportunities and Threats",
      "Risk Management Policy/Framework/Mitigation", "Principal/Key Risks",
      "Internal Control Systems", "SWOT") — strictly excluding Ind-AS 107 financial
      instrument footnote disclosures and Corporate Governance committee-attendance tables.
      -> step0_sections/risk/<bse_code>/<year>_risk.md
  - Block C (Directors_Report): Board's / Directors' Report
      -> step0_sections/directors/<bse_code>/<year>_directors.md
  - Block D (Other_Narrative) & AGM Notice coordinates (`risk_spans`, `notice_spans`):
      Exact line coordinates exported to `section_manifest.csv` so Step 1 constructs
      strictly disjoint `MDA_ExRisk`, `OTHER` (non-1A, non-MD&A, non-AGM-Notice),
      and `Full_Ex1A` sections.
"""

import os
import re
import sys
import glob
import time
import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AR_DIR = os.path.join(BASE_DIR, "annual_reports")
CMIE_CSV = os.path.join(BASE_DIR, "05_cmie_data", "cmie_identity_indicators.csv")
OUT_DIR = os.path.join(BASE_DIR, "04_outputs", "step0_sections")

# Pre-compiled regular expressions
IMG_OR_HTML_RE = re.compile(r"<[^>]+>|!\[[^\]]*\]\([^\)]*\)")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]*\)")
SYM_RE = re.compile(r"[\*\#\_\`\~\|]+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

# Acronym/abbreviation-safe sentence splitter (Li 2010 Footnote 7: e.g. "Rs. 500", "No. 123", "Mr. X")
ABBREV_PROTECT_RE = re.compile(
    r"\b(?:Rs|Dr|Mr|Mrs|Ms|Prof|Shri|Smt|No|Nos|Sr|Jr|St|Ltd|Pvt|Inc|Corp|Co|vs|approx|Fig|Sec|Reg|Art|Ch|Vol|pp|viz|al|etc)\.",
    re.I,
)
DOTTED_ABBREV_RE = re.compile(r"\b(?:w\.e\.f|i\.e|e\.g|a\.m|p\.m|f\.y|a\.y|u\.s|u\.k)\.", re.I)
INITIAL_PROTECT_RE = re.compile(r"(?<=\b[A-Z])\.(?=\s+[A-Z])")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences_safe(text):
    """Split narrative text into sentences without breaking on 'Rs. 500', 'No. 12', 'Mr. Rao', etc."""
    if not text:
        return []
    protected = ABBREV_PROTECT_RE.sub(lambda m: m.group(0)[:-1] + "\x00", text)
    protected = DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), protected)
    protected = INITIAL_PROTECT_RE.sub("\x00", protected)
    raw_sents = SENT_SPLIT_RE.split(protected)
    return [s.replace("\x00", ".").strip() for s in raw_sents if s.strip()]


# Sub-headings characteristic of MD&A interior content (SEBI Clause 49 / LODR Schedule V)
MDA_INTERIOR_RE = re.compile(
    r"industry\s+structure\s+(?:and|&)\s+developments|"
    r"opportunities\s+(?:and|&)\s+threats|"
    r"risks?\s+(?:and|&)\s+concerns?|"
    r"segment[\s\-]*(?:wise|level)\s+(?:or\s+product[\s\-]*wise\s+)?performance|"
    r"business\s+segment\s+analysis|"
    r"operating\s+results\s+of\s+the\s+company|"
    r"outlook\s+for\s+the\s+company|"
    r"internal\s+control\s+systems?(?:\s+(?:and|&)\s+their\s+adequacy)?|"
    r"material\s+developments\s+in\s+human\s+resources|"
    r"industrial\s+relations|"
    r"cautionary\s+statement|"
    r"economic\s+overview|global\s+economy|indian\s+economy|"
    r"operational\s+performance|financial\s+performance|"
    r"key\s+financial\s+ratios",
    re.I,
)

MDA_CAUTIONARY_ANCHOR_RE = re.compile(
    r"statements?\s+in\s+(?:this|the)\s+management\s+discussion\s+(?:and|&)\s+analysis\s+describing",
    re.I,
)

# Cross-reference stub phrases when MD&A or Corporate Governance is only mentioned in 1-2 sentences inside Directors' Report
CROSS_REF_STUB_RE = re.compile(
    r"separate\s+section\s+forming\s+part|"
    r"annexed\s+hereto\s+and\s+forms\s+an?\s+integral\s+part|"
    r"is\s+attached\s+as\s+annexure|"
    r"is\s+given\s+in\s+a\s+separate\s+section|"
    r"forms\s+part\s+of\s+(?:this|the)\s+annual\s+report|"
    r"is\s+presented\s+(?:separately|in\s+a\s+separate\s+section)|"
    r"is\s+annexed\s+to\s+(?:this|the)\s+report",
    re.I,
)

# Substantive Risk subsection headings (Block B) - excludes Corporate Governance "Risk Management Committee"
RISK_HEADING_RE = re.compile(
    r"\b(?:"
    r"risks?\s*(?:and|&|,|/)\s*concerns?|"
    r"concerns?\s*(?:and|&)\s*risks?|"
    r"opportunities\s*(?:and|&|,|/)\s*threats?|"
    r"threats?\s*(?:and|&|,|/)\s*opportunities|"
    r"opportunities.{0,25}risks?.{0,25}concerns?|"
    r"threats?\s*(?:and|&|,|/)\s*risks?|"
    r"risk\s*management(?:\s*policy|\s*framework|\s*system|\s*process|\s*architecture|\s*(?:and|&)\s*mitigation)?|"
    r"enterprise\s*risk\s*management|"
    r"risk\s*assessment(?:\s*(?:and|&)\s*mitigation|\s*framework|\s*and\s*minimization)?|"
    r"risk\s*mitigation|"
    r"key\s*risks?(?:\s*(?:and|&)\s*mitigation)?|"
    r"principal\s*risks?(?:\s*(?:and|&)\s*uncertainties)?|"
    r"major\s*risks?|"
    r"business\s*risks?|"
    r"operational\s*risks?|"
    r"strategic\s*risks?|"
    r"internal\s*control\s*systems?\s*(?:and|&)\s*their\s*adequacy|"
    r"swot\s*analysis|"
    r"threats\b"
    r")",
    re.I,
)

FIN_RISK_NOTE_RE = re.compile(
    r"\bfinancial\s+(?:instruments?.{0,30})?risk\s+management\b|"
    r"\bcapital\s+(?:and\s+)?(?:financial\s+)?risk\s+management\b",
    re.I,
)

RISK_EXCLUDE_RE = re.compile(
    r"auditor.{0,25}risk|"
    r"risks?\s+of\s+material\s+misstatement|"
    r"identify\s+and\s+assess\s+the\s+risks|"
    r"table\s+of\s+contents|"
    r"\brisk\s*management\s*committee\b|"
    r"terms\s+of\s+reference",
    re.I,
)


def check_text_quality(text):
    """Detect empty files, font-encoding mojibake, or truncated/stub files."""
    if not text or len(text.strip()) < 200:
        return "empty_file", 0.0
    sample = text[:20000]
    alpha_ascii = sum(1 for c in sample if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    non_ascii = sum(1 for c in sample if ord(c) > 127)
    total_chars = len(sample)
    ascii_ratio = alpha_ascii / max(1, total_chars)
    if non_ascii > alpha_ascii * 0.35 or ascii_ratio < 0.25:
        return "corrupted_font_encoding", round(ascii_ratio, 3)
    return "ok", round(ascii_ratio, 3)


def clean_markdown_inline(s):
    """Strip HTML tags, images, Markdown link targets `[txt](#anchor)` -> `txt`, and Markdown symbols."""
    s = IMG_OR_HTML_RE.sub(" ", s)
    s = MD_LINK_RE.sub(r"\1", s)
    s = SYM_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_heading_line(line):
    """
    If `line` is a Markdown heading (#...) or standalone bold heading (**...**),
    return (clean_text, strip_prefix_text, level). Otherwise return None.
    """
    s = line.strip()
    if not s or len(s) > 220 or s.startswith("|") or s.startswith("!["):
        return None
    level = 0
    if s.startswith("#"):
        m = re.match(r"^(#{1,6})\s+", s)
        if not m:
            return None
        level = len(m.group(1))
    elif s.startswith("**") and (s.endswith("**") or s.endswith("**:") or s.endswith("**.")):
        level = 4
    else:
        return None

    clean = clean_markdown_inline(s)
    if not clean or len(clean) < 3 or len(clean) > 140:
        return None

    # Exclude Table-of-Contents lines that end with dots or standalone page numbers
    if re.search(r"\.{3,}\s*\d*\s*$", clean):
        return None
    toks = clean.split()
    if len(toks) >= 2 and toks[-1].isascii() and toks[-1].isdigit():
        num = int(toks[-1])
        if 1 <= num <= 450 and not re.search(
            r"\b(?:act|section|clause|rule|fy|year|part|annexure|attachment|note|item|schedule|table|figure|no)\.?$",
            toks[-2],
            re.I,
        ):
            return None

    # Strip leading numbering / Annexure prefixes for canonical classification
    core = re.sub(
        r"^\s*(?:annexure\s*[a-z0-9\-\.]*\s*(?:to\s*(?:the\s*)?(?:directors?'?s?|board'?s?)\s*report)?[\s:\-\.]*)",
        "",
        clean,
        flags=re.I,
    )
    core = re.sub(
        r"^\s*(?:section|item|part)?\s*\(?[a-z0-9IVX]{1,4}[\.\)\-\:]\s*",
        "",
        core,
        flags=re.I,
    ).strip()
    return clean, (core or clean), level


def classify_major_section(clean, core):
    """
    Classify a heading into one of the top-level Indian Annual Report sections:
      - 'MDA_AND_DIRECTORS'
      - 'MDA'
      - 'DIRECTORS'
      - 'CORP_GOV'
      - 'BRSR'
      - 'AUDITORS_OR_FIN'
      - 'NOTICE'
      - None (regular subheading)
    """
    low = core.lower()
    low_full = clean.lower()
    low_nospace = re.sub(r"[\s\-\_]+", "", low)

    if len(low.split()) > 16:
        return None
    if any(
        p in low_full
        for p in [
            "forms part of",
            "is annexed",
            "refer to",
            "as required under",
            "pursuant to",
            "in terms of",
        ]
    ):
        return None

    # Combined Directors' Report and MD&A
    if (
        ("director" in low or "board" in low)
        and "report" in low
        and ("management" in low and ("discussion" in low or "analysis" in low))
    ):
        return "MDA_AND_DIRECTORS"

    # Management Discussion & Analysis (including when attached as `Annexure I - Management Discussion and Analysis Report`)
    if (
        re.search(r"^management.{0,6}discussion.{0,6}(?:and|&).{0,6}analysis", low)
        or low_nospace.startswith("managementdiscussionandanalysis")
        or low_nospace.startswith("managementsdiscussionandanalysis")
        or re.match(r"^md\s*&\s*a(?:\s+report)?$", low)
    ):
        # Exclude combined stub heading `Management Discussion and Analysis Report & Corporate Governance` if short
        return "MDA"

    # Directors' / Board's Report
    if (
        re.match(
            r"^(?:the\s+)?(?:directors?'?s?|board'?s?)\s*report(?:\s+to\s+(?:the\s+)?(?:shareholders|members)|\s*[\-\–\:]?\s*(?:19|20)\d{2}[\-\d]*)?$",
            low,
        )
        or re.match(r"^report\s+of\s+the\s+(?:board\s+of\s+)?directors?(?:\s+to\s+the\s+members)?$", low)
        or low_nospace in ("directorsreport", "boardsreport", "directorreport", "boardreport")
    ):
        if "annexure" not in low_full:
            return "DIRECTORS"

    # Corporate Governance Report
    if (
        re.match(r"^(?:report\s+on\s+)?corporate\s+governance(?:\s+report)?$", low)
        or low_nospace in ("corporategovernancereport", "reportoncorporategovernance")
    ):
        return "CORP_GOV"

    # Business Responsibility / Sustainability Report
    if re.match(
        r"^business\s+responsibility(?:\s+(?:and|&)\s+sustainability)?\s+report", low
    ):
        return "BRSR"

    # Auditors' Report / Secretarial Audit / CEO-CFO Certification / Financial Statements
    if (
        re.match(r"^(?:independent\s+)?auditors?'?s?\s+(?:re[np]ort|certificate)", low)
        or re.match(r"^re[np]ort\s+of\s+the\s+(?:independent\s+)?auditors?", low)
        or re.match(r"^secretarial\s+audit\s+re[np]ort", low)
        or re.match(r"^form\s+no\.?\s*mr[\s\-]*3", low)
        or re.match(r"^form\s+no\.?\s*mgt[\s\-]*9", low)
        or re.match(r"^ceo\s*(?:/|and|&)\s*cfo\s+certificat", low)
        or re.match(r"^declaration\s+(?:by\s+the\s+(?:managing\s+director|ceo)|regarding\s+compliance)", low)
        or re.match(
            r"^(?:standalone\s+|consolidated\s+)?(?:balance\s+sheet|statement\s+of\s+profit\s+(?:and|&)\s+loss|profit\s+(?:and|&)\s+loss\s+account|cash\s+flow\s+statement|schedule\s+1\s*[\-\:]\s*share\s+capital)(?:\s+as\s+at.*)?$",
            low,
        )
    ):
        return "AUDITORS_OR_FIN"

    # Notice of AGM / Proxy Form
    if re.match(
        r"^notice(?:\s+of\s+(?:the\s+)?(?:\d+[a-z]{2}\s+)?annual\s+general\s+meeting|\s+to\s+(?:the\s+)?(?:members|shareholders))?$|"
        r"^(?:proxy\s+form|attendance\s+slip|e[\-\s]*voting\s+instructions)$",
        low,
    ):
        return "NOTICE"

    return None


def count_words(lines_slice):
    """Count alphanumeric word tokens in a list of lines (excluding markdown table separator lines)."""
    total = 0
    for l in lines_slice:
        s = l.strip()
        if not s or s.startswith("![") or re.match(r"^\|[\s\-\:\|]+\|$", s):
            continue
        total += len(WORD_RE.findall(s))
    return total


def count_narrative_words_and_sentences(lines_slice):
    """Count words and acronym-safe sentences in narrative lines (excluding markdown tables `|...|`)."""
    words = 0
    narrative_chunks = []
    for l in lines_slice:
        s = l.strip()
        if not s or s.startswith("|") or s.startswith("![") or s.startswith("<!--"):
            continue
        clean = clean_markdown_inline(s)
        w = WORD_RE.findall(clean)
        if w:
            words += len(w)
            narrative_chunks.append(clean)
    if not narrative_chunks:
        return 0, 0
    joined = " ".join(narrative_chunks)
    sents = len(split_sentences_safe(joined))
    return words, sents


def segment_single_report(args):
    """Worker function to segment a single <BSE_Code>/markdown/<Year>.md file."""
    md_path, bse_code, year, write_extracts = args
    rel_src = f"{bse_code}/markdown/{year}.md"
    try:
        with open(md_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        return {
            "bse_code": bse_code,
            "year": year,
            "source_path": rel_src,
            "status": f"read_error:{type(e).__name__}",
            "ascii_ratio": 0.0,
            "total_lines": 0,
            "total_words": 0,
        }

    q_status, ascii_ratio = check_text_quality(text)
    lines = text.splitlines()
    n_lines = len(lines)
    total_words = count_words(lines)

    if q_status != "ok":
        return {
            "bse_code": bse_code,
            "year": int(year) if year.isdigit() else year,
            "source_path": rel_src,
            "status": q_status,
            "ascii_ratio": ascii_ratio,
            "total_lines": n_lines,
            "total_words": total_words,
            "mda_found": False,
            "mda_start_line": None,
            "mda_end_line": None,
            "mda_words": 0,
            "mda_sentences": 0,
            "risk_found": False,
            "risk_sections_count": 0,
            "risk_words": 0,
            "risk_sentences": 0,
            "risk_headings": "",
            "risk_spans": "",
            "fin_risk_note_words": 0,
            "directors_found": False,
            "directors_start_line": None,
            "directors_end_line": None,
            "directors_words": 0,
            "notice_spans": "",
            "other_narrative_words": 0,
            "other_narrative_sentences": 0,
        }

    # 1. Scan all headings (1-based line numbers)
    all_headings = []  # (line_no, level, clean, core, major_type)
    for idx, line in enumerate(lines, 1):
        parsed = parse_heading_line(line)
        if parsed:
            clean, core, level = parsed
            mtype = classify_major_section(clean, core)
            all_headings.append((idx, level, clean, core, mtype))

    major_headings = [h for h in all_headings if h[4] is not None]

    # Filter out mini-TOC clusters where >=3 major headings appear within 12 lines of each other
    filtered_major = []
    for i, h in enumerate(major_headings):
        line_no = h[0]
        neighbors = sum(
            1
            for j, h2 in enumerate(major_headings)
            if i != j and abs(h2[0] - line_no) <= 12
        )
        if neighbors >= 2:
            if i + 1 < len(major_headings):
                next_l = major_headings[i + 1][0]
                if count_words(lines[line_no - 1 : next_l - 1]) < 40:
                    continue
        filtered_major.append(h)

    # Merge consecutive identical major section types (e.g. running page headers `# DIRECTORS' REPORT`)
    merged_major = []
    for h in filtered_major:
        if merged_major and merged_major[-1][4] == h[4]:
            continue
        merged_major.append(h)

    # Identify AGM Notice spans (for clean legal-boilerplate exclusion in Step 1)
    notice_blocks = []
    for i, h in enumerate(merged_major):
        if h[4] == "NOTICE":
            n_s = h[0]
            n_e = merged_major[i + 1][0] - 1 if (i + 1 < len(merged_major)) else n_lines
            # Cap notice span if at end of report or before Directors' Report
            n_e = min(n_e, n_s + 1200)
            if n_e >= n_s:
                notice_blocks.append((n_s, n_e))

    # 2. Identify candidate MD&A spans (Block A)
    mda_candidates = []
    for i, h in enumerate(merged_major):
        l_start, lvl, clean, core, mtype = h
        if mtype in ("MDA", "MDA_AND_DIRECTORS"):
            l_end = n_lines
            for j in range(i + 1, len(merged_major)):
                if merged_major[j][4] not in ("MDA",):
                    l_end = merged_major[j][0] - 1
                    break
            span_lines = lines[l_start - 1 : l_end]
            w_cnt, s_cnt = count_narrative_words_and_sentences(span_lines)
            span_txt = "\n".join(span_lines)
            is_stub = (w_cnt < 180 and bool(CROSS_REF_STUB_RE.search(span_txt))) or w_cnt < 80
            interior_hits = len(MDA_INTERIOR_RE.findall(span_txt))
            score = w_cnt + interior_hits * 300 - (5000 if is_stub else 0)
            mda_candidates.append((score, l_start, l_end, w_cnt, s_cnt, mtype, is_stub))

    mda_found = False
    mda_start, mda_end, mda_words, mda_sents = None, None, 0, 0
    if mda_candidates:
        mda_candidates.sort(key=lambda x: x[0], reverse=True)
        best = mda_candidates[0]
        if best[3] >= 100 and not best[6]:
            mda_found = True
            _, mda_start, mda_end, mda_words, mda_sents, _, _ = best

    # 3. Identify candidate Directors' Report spans (Block C)
    dir_candidates = []
    for i, h in enumerate(merged_major):
        l_start, lvl, clean, core, mtype = h
        if mtype in ("DIRECTORS", "MDA_AND_DIRECTORS"):
            l_end = n_lines
            for j in range(i + 1, len(merged_major)):
                next_mtype = merged_major[j][4]
                next_l = merged_major[j][0]
                if next_mtype in ("MDA", "CORP_GOV"):
                    next_end = merged_major[j + 1][0] - 1 if (j + 1 < len(merged_major)) else n_lines
                    sub_w = count_words(lines[next_l - 1 : next_end])
                    sub_txt = "\n".join(lines[next_l - 1 : min(next_end, next_l + 30)])
                    if sub_w < 180 or CROSS_REF_STUB_RE.search(sub_txt):
                        continue
                if next_mtype != "DIRECTORS":
                    l_end = next_l - 1
                    break
            span_lines = lines[l_start - 1 : l_end]
            w_cnt, _ = count_narrative_words_and_sentences(span_lines)
            if w_cnt >= 120:
                dir_candidates.append((w_cnt, l_start, l_end))

    dir_found = False
    dir_start, dir_end, dir_words = None, None, 0
    if dir_candidates:
        dir_candidates.sort(key=lambda x: x[0], reverse=True)
        dir_words, dir_start, dir_end = dir_candidates[0]
        dir_found = True
    else:
        open_pat = re.compile(
            r"^\s*(?:[\*\#\>\_\s])*(?:your|the)\s+directors\s+(?:have\s+(?:great\s+)?pleasure|are\s+pleased|take\s+(?:great\s+)?pleasure|present\s+their)\b",
            re.I,
        )
        for idx_l, l in enumerate(lines, 1):
            if open_pat.search(l):
                l_start = max(1, idx_l - 2)
                l_end = n_lines
                for mh in merged_major:
                    if mh[0] > idx_l + 15 and mh[4] in ("MDA", "CORP_GOV", "BRSR", "AUDITORS_OR_FIN"):
                        l_end = mh[0] - 1
                        break
                w_cnt, _ = count_narrative_words_and_sentences(lines[l_start - 1 : l_end])
                if w_cnt >= 120:
                    dir_found = True
                    dir_start, dir_end, dir_words = l_start, l_end, w_cnt
                    break

    # Fallback A for MD&A: cluster of >=2 SEBI Clause 49 MD&A interior subheadings
    if not mda_found:
        mda_sub_lines = [
            h[0] for h in all_headings if h[4] is None and MDA_INTERIOR_RE.search(h[2])
        ]
        if len(mda_sub_lines) >= 2:
            c_start = mda_sub_lines[0]
            l_end = n_lines
            for mh in merged_major:
                if mh[0] > c_start + 10 and mh[4] in ("CORP_GOV", "BRSR", "AUDITORS_OR_FIN", "DIRECTORS"):
                    l_end = mh[0] - 1
                    break
            l_end = min(l_end, c_start + 750)
            w_cnt, s_cnt = count_narrative_words_and_sentences(lines[c_start - 1 : l_end])
            if w_cnt >= 150:
                mda_found = True
                mda_start, mda_end, mda_words, mda_sents = c_start, l_end, w_cnt, s_cnt

    # Fallback B for MD&A: graphic-banner MD&As ending with canonical MD&A Cautionary Statement
    if not mda_found:
        for idx_l, l in enumerate(lines, 1):
            if MDA_CAUTIONARY_ANCHOR_RE.search(l):
                # End of MD&A is end of this cautionary block (next major heading or +15 lines)
                c_end = min(n_lines, idx_l + 12)
                for mh in merged_major:
                    if mh[0] > idx_l:
                        c_end = min(c_end, mh[0] - 1)
                        break
                # Start is immediately after preceding major section (e.g. Directors' Report / MGT-9 / AOC-2), bounded to <=600 lines back
                prev_majors = [mh[0] for mh in merged_major if mh[0] < idx_l - 25]
                if prev_majors:
                    # Find end of previous major section or start from last major boundary if within 600 lines
                    c_start = max(idx_l - 500, prev_majors[-1])
                    if dir_found and dir_end and dir_end < idx_l - 20:
                        c_start = max(c_start, dir_end + 1)
                else:
                    c_start = max(1, idx_l - 400)
                w_cnt, s_cnt = count_narrative_words_and_sentences(lines[c_start - 1 : c_end])
                if w_cnt >= 250:
                    mda_found = True
                    mda_start, mda_end, mda_words, mda_sents = c_start, c_end, w_cnt, s_cnt
                    break

    # 4. Identify Risk Subsections (Block B: Substantive Narrative Risk vs Ind-AS 107 Financial Risk Note)
    # Determine where Financial Statements / Auditor's Report begin so footnotes never leak into Block B
    first_fin_line = n_lines + 1
    for mh in merged_major:
        if mh[4] == "AUDITORS_OR_FIN" and mh[0] > (mda_end or 0) and mh[0] > (dir_end or 0):
            # Verify it is not an MGT-9 / MR-3 annexure before MD&A
            if "auditor" in mh[2].lower() or "balance sheet" in mh[2].lower() or mh[0] > n_lines * 0.42:
                first_fin_line = mh[0]
                break

    narrative_risk_blocks = []  # list of (start_line, end_line, heading_str, parent_sec, words, sents)
    fin_risk_note_words = 0

    for idx_h, (l_start, lvl, clean, core, mtype) in enumerate(all_headings):
        if mtype is not None:
            continue
        if RISK_EXCLUDE_RE.search(clean):
            continue
        is_fin_note = bool(FIN_RISK_NOTE_RE.search(clean))
        is_narr_risk = bool(RISK_HEADING_RE.search(clean)) and not is_fin_note

        if not (is_fin_note or is_narr_risk):
            continue

        # Find end of this subsection
        l_end = n_lines
        for j in range(idx_h + 1, len(all_headings)):
            nh_line, nh_lvl, nh_clean, nh_core, nh_mtype = all_headings[j]
            if nh_mtype is not None:
                l_end = nh_line - 1
                break
            # Stop at clearly non-risk MD&A / Directors / Governance sections regardless of heading level
            if re.search(
                r"\b(?:human\s+resources?|industrial\s+relations?|cautionary\s+statement|outlook|internal\s+audit|financial\s+performance|segment\s+performance|dividend|fixed\s+deposits|subsidiaries|board\s+meetings?|acknowledg|corporate\s+social\s+responsibility|directors?'?\s+responsibility|related\s+party|conservation\s+of\s+energy|particulars\s+of\s+employees|vigil\s+mechanism|whistle\s*blower|audit\s+committee|stakeholders?\s+relationship|nomination\s+and\s+remuneration)\b",
                nh_clean,
                re.I,
            ) and not RISK_HEADING_RE.search(nh_clean):
                l_end = nh_line - 1
                break
            # If next heading is same or higher importance, stop UNLESS it is an explicit sub-risk / mitigation heading
            if nh_lvl <= lvl and not re.search(
                r"\b(?:risks?|credit|liquidity|market|currency|foreign\s+exchange|interest\s+rate|commodity|operational|cyber|mitigation|threats?|concerns?)\b",
                nh_clean,
                re.I,
            ):
                l_end = nh_line - 1
                break

        # Cap single risk subsection span to 800 lines max (and never cross into Financial Statements)
        if l_start < first_fin_line:
            l_end = min(l_end, first_fin_line - 1, l_start + 800)
        else:
            l_end = min(l_end, l_start + 800)

        if l_end < l_start:
            continue

        sub_lines = lines[l_start - 1 : l_end]
        w_cnt, s_cnt = count_narrative_words_and_sentences(sub_lines)
        if w_cnt < 15:
            continue

        # Determine parent section context
        if mda_found and mda_start <= l_start <= mda_end:
            parent = "MDA"
        elif dir_found and dir_start <= l_start <= dir_end:
            parent = "DIRECTORS_REPORT"
        elif l_start >= first_fin_line or (l_start > max(mda_end or 0, dir_end or 0) and l_start > n_lines * 0.45):
            parent = "FINANCIAL_NOTES_OR_OTHER"
        else:
            parent = "OTHER_REPORT_SECTION"

        # Strictly exclude Ind-AS 107 / Financial Statement Footnotes from Block B (`Risk_Section`)
        if is_fin_note or parent == "FINANCIAL_NOTES_OR_OTHER":
            if is_fin_note or "risk" in clean.lower():
                fin_risk_note_words += w_cnt
            continue

        # Avoid duplicate overlapping risk spans
        if narrative_risk_blocks and l_start <= narrative_risk_blocks[-1][1]:
            continue

        narrative_risk_blocks.append((l_start, l_end, clean, parent, w_cnt, s_cnt))

    risk_found = len(narrative_risk_blocks) > 0
    risk_words = sum(b[4] for b in narrative_risk_blocks)
    risk_sents = sum(b[5] for b in narrative_risk_blocks)
    risk_headings_str = " | ".join(f"{b[2]} [{b[3]}:L{b[0]}-{b[1]}]" for b in narrative_risk_blocks)
    risk_spans_str = ";".join(f"{b[0]}-{b[1]}" for b in narrative_risk_blocks)
    notice_spans_str = ";".join(f"{s}-{e}" for s, e in notice_blocks)

    # 5. Compute Block D (Other_Narrative: remaining non-table narrative outside MDA, Risk, Directors, and AGM Notice)
    covered = bytearray(n_lines + 1)
    if mda_found:
        for k in range(mda_start, mda_end + 1):
            covered[k] = 1
    if dir_found:
        for k in range(dir_start, dir_end + 1):
            covered[k] = 1
    for b_s, b_e, _, _, _, _ in narrative_risk_blocks:
        for k in range(b_s, b_e + 1):
            covered[k] = 1
    for n_s, n_e in notice_blocks:
        for k in range(n_s, n_e + 1):
            covered[k] = 1

    other_lines = [lines[k - 1] for k in range(1, n_lines + 1) if not covered[k]]
    other_words, other_sents = count_narrative_words_and_sentences(other_lines)

    # 6. Write extracted Markdown files if requested
    if write_extracts:
        if mda_found:
            mda_out_dir = os.path.join(OUT_DIR, "mda", bse_code)
            os.makedirs(mda_out_dir, exist_ok=True)
            mda_out_path = os.path.join(mda_out_dir, f"{year}_mda.md")
            header_comment = f"<!-- source: {rel_src} | lines {mda_start}-{mda_end} | words: {mda_words} | sentences: {mda_sents} -->\n"
            with open(mda_out_path, "w", encoding="utf-8") as fo:
                fo.write(header_comment + "\n".join(lines[mda_start - 1 : mda_end]) + "\n")

        if risk_found:
            risk_out_dir = os.path.join(OUT_DIR, "risk", bse_code)
            os.makedirs(risk_out_dir, exist_ok=True)
            risk_out_path = os.path.join(risk_out_dir, f"{year}_risk.md")
            chunks = [
                f"<!-- source: {rel_src} | blocks: {len(narrative_risk_blocks)} | total_words: {risk_words} | total_sentences: {risk_sents} -->"
            ]
            for idx_b, (b_s, b_e, b_hdr, b_parent, b_w, b_s_cnt) in enumerate(narrative_risk_blocks, 1):
                chunks.append(
                    f"\n<!-- risk_block_{idx_b}: parent={b_parent} | heading={b_hdr} | lines {b_s}-{b_e} | words={b_w} -->\n"
                    + "\n".join(lines[b_s - 1 : b_e])
                )
            with open(risk_out_path, "w", encoding="utf-8") as fo:
                fo.write("\n".join(chunks) + "\n")

        if dir_found:
            dir_out_dir = os.path.join(OUT_DIR, "directors", bse_code)
            os.makedirs(dir_out_dir, exist_ok=True)
            dir_out_path = os.path.join(dir_out_dir, f"{year}_directors.md")
            header_comment = f"<!-- source: {rel_src} | lines {dir_start}-{dir_end} | words: {dir_words} -->\n"
            with open(dir_out_path, "w", encoding="utf-8") as fo:
                fo.write(header_comment + "\n".join(lines[dir_start - 1 : dir_end]) + "\n")

    return {
        "bse_code": bse_code,
        "year": int(year) if year.isdigit() else year,
        "source_path": rel_src,
        "status": "ok",
        "ascii_ratio": ascii_ratio,
        "total_lines": n_lines,
        "total_words": total_words,
        "mda_found": mda_found,
        "mda_start_line": mda_start,
        "mda_end_line": mda_end,
        "mda_words": mda_words,
        "mda_sentences": mda_sents,
        "risk_found": risk_found,
        "risk_sections_count": len(narrative_risk_blocks),
        "risk_words": risk_words,
        "risk_sentences": risk_sents,
        "risk_headings": risk_headings_str,
        "risk_spans": risk_spans_str,
        "fin_risk_note_words": fin_risk_note_words,
        "directors_found": dir_found,
        "directors_start_line": dir_start,
        "directors_end_line": dir_end,
        "directors_words": dir_words,
        "notice_spans": notice_spans_str,
        "other_narrative_words": other_words,
        "other_narrative_sentences": other_sents,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 0 (v2): Segment Indian Annual Reports")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (for testing)")
    parser.add_argument("--no-write", action="store_true", help="Only build manifest without writing section .md files")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Number of parallel workers")
    args = parser.parse_args()

    # Clean stale risk/mda/directors files before regenerating so removed false-positive extracts do not linger
    if not args.no_write and args.limit == 0:
        import shutil
        for sub in ("mda", "risk", "directors"):
            p_sub = os.path.join(OUT_DIR, sub)
            if os.path.exists(p_sub):
                shutil.rmtree(p_sub)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_paths = sorted(glob.glob(os.path.join(AR_DIR, "*", "markdown", "*.md")))
    if args.limit > 0:
        all_paths = all_paths[: args.limit]

    tasks = []
    for p in all_paths:
        parts = p.split(os.sep)
        bse_code = parts[-3]
        year = os.path.splitext(parts[-1])[0]
        tasks.append((p, bse_code, year, not args.no_write))

    t0 = time.time()
    print(f"[Step 0 v2] Segmenting {len(tasks)} annual report Markdown files using {args.workers} workers...")
    results = []
    with Pool(processes=args.workers) as pool:
        for idx, res in enumerate(pool.imap_unordered(segment_single_report, tasks, chunksize=50), 1):
            results.append(res)
            if idx % 2000 == 0 or idx == len(tasks):
                elapsed = time.time() - t0
                print(f"  Processed {idx}/{len(tasks)} files ({idx / max(0.1, elapsed):.1f} files/sec)...")

    df = pd.DataFrame(results)
    df = df.sort_values(["bse_code", "year"]).reset_index(drop=True)

    # Merge CMIE Prowess metadata if available
    if os.path.exists(CMIE_CSV):
        cmie = pd.read_csv(CMIE_CSV, dtype={"BSE scrip code": str})
        cmie["bse_code"] = cmie["BSE scrip code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        keep_cols = [
            "bse_code",
            "Company Name",
            "Prowess company code",
            "CIN code",
            "NIC code",
            "NIC name",
            "Industry group",
            "Ownership group",
            "Size decile",
            "ISIN code",
            "NSE symbol",
            "BSE scrip id",
        ]
        keep_cols = [c for c in keep_cols if c in cmie.columns]
        cmie_sub = cmie[keep_cols].drop_duplicates(subset=["bse_code"])
        df = df.merge(cmie_sub, on="bse_code", how="left")

    manifest_path = os.path.join(OUT_DIR, "section_manifest.csv")
    df.to_csv(manifest_path, index=False)

    elapsed = time.time() - t0
    ok_df = df[df["status"] == "ok"]
    print(f"\n[Step 0 v2 Complete in {elapsed:.1f}s]")
    print(f"  Manifest saved to: {manifest_path}")
    print(f"  Total files scanned:         {len(df):,}")
    print(f"  Valid English text ('ok'):   {len(ok_df):,} ({100 * len(ok_df) / max(1, len(df)):.1f}%)")
    print(f"  Empty / stub files:          {(df['status'] == 'empty_file').sum():,}")
    print(f"  Corrupted font encoding:     {(df['status'] == 'corrupted_font_encoding').sum():,}")
    if len(ok_df) > 0:
        print(f"  Block A (MD&A extracted):    {ok_df['mda_found'].sum():,} / {len(ok_df):,} ({100 * ok_df['mda_found'].mean():.1f}%) | Median words: {int(ok_df.loc[ok_df['mda_found'], 'mda_words'].median() if ok_df['mda_found'].any() else 0):,}")
        print(f"  Block B (Risk extracted):    {ok_df['risk_found'].sum():,} / {len(ok_df):,} ({100 * ok_df['risk_found'].mean():.1f}%) | Median words: {int(ok_df.loc[ok_df['risk_found'], 'risk_words'].median() if ok_df['risk_found'].any() else 0):,}")
        print(f"  Block C (Directors Report):  {ok_df['directors_found'].sum():,} / {len(ok_df):,} ({100 * ok_df['directors_found'].mean():.1f}%) | Median words: {int(ok_df.loc[ok_df['directors_found'], 'directors_words'].median() if ok_df['directors_found'].any() else 0):,}")


if __name__ == "__main__":
    main()
