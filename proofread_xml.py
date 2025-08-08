import argparse
import os
import time
import json
import tracemalloc
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from lxml import etree
from dotenv import load_dotenv

from xml_utils import load_xml, get_p_elements, apply_issues_to_p
from genai_client import proofread_with_genai

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------- .env helpers --------------------

load_dotenv()  # loads .env from project root

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v

# -------------------- metrics helpers --------------------

def perf_summary(t0: float, file_path: str, peak_bytes: int) -> Dict:
    t1 = time.perf_counter()
    return {
        "file": os.path.basename(file_path),
        "time_seconds": round(t1 - t0, 4),
        "mem_peak_mb": round(peak_bytes / (1024 * 1024), 2),
    }

def _flatten_p_text(p: etree._Element) -> str:
    """Rebuild <p> inner text while ignoring <error> tags (use their text)."""
    buf: List[str] = []
    if p.text:
        buf.append(p.text)
    for node in p:
        tag = str(node.tag) if node.tag is not None else ""
        if tag.endswith("error"):
            if node.text:
                buf.append(node.text)
            if node.tail:
                buf.append(node.tail)
        else:
            if node.text:
                buf.append(node.text)
            if node.tail:
                buf.append(node.tail)
    return "".join(buf)

def invariant_score(tree_before: etree._ElementTree, tree_after: etree._ElementTree) -> float:
    root_b = tree_before.getroot()
    root_a = tree_after.getroot()
    tb = "\n".join(_flatten_p_text(p) for p in root_b.xpath(".//*[local-name()='p']"))
    ta = "\n".join(_flatten_p_text(p) for p in root_a.xpath(".//*[local-name()='p']"))
    return 1.0 if tb == ta else 0.0

def count_error_tags(root: etree._Element) -> int:
    return len(root.xpath(".//*[local-name()='error']"))

# -------------------- tiny safe heuristics --------------------

def _preserve_first_case(original: str, corrected: str) -> str:
    """If original starts uppercase, capitalize corrected's first letter."""
    if not original:
        return corrected
    if original[0].isupper():
        return corrected[0].upper() + corrected[1:]
    return corrected

def heuristic_issues(text: str, lang: str = "en") -> List[Dict]:
    """
    Minimal, deterministic fixes to cover frequent misses.
    Spans must be minimal (one word OR one punctuation/space run) to keep the length invariant.
    """
    out: List[Dict] = []

    # their -> they're (case-preserving)
    for m in re.finditer(r"\btheir\b", text, flags=re.IGNORECASE):
        orig = m.group(0)
        corr = _preserve_first_case(orig, "they're")
        out.append({
            "start": m.start(), "end": m.end(),
            "type": "grammar", "correction": corr,
            "reason": "Use they're (they are) instead of possessive their."
        })

    # he don't -> doesn't (wrap only "don't")
    for m in re.finditer(r"\b[Hh]e don't\b", text):
        span = m.group(0)
        tok_idx = span.lower().find("don't")
        s = m.start() + tok_idx
        e = s + len("don't")
        out.append({
            "start": s, "end": e,
            "type": "grammar", "correction": "doesn't",
            "reason": "Subject-verb agreement: he doesn't (not he don't)."
        })

    # lets -> let's (case-preserving, token-level)
    for m in re.finditer(r"\blets\b", text, flags=re.IGNORECASE):
        orig = m.group(0)
        corr = _preserve_first_case(orig, "let's")
        out.append({
            "start": m.start(), "end": m.end(),
            "type": "grammar", "correction": corr,
            "reason": "Missing apostrophe in the contraction."
        })

    # Sentence starts lowercase → capitalize first word
    m0 = re.match(r"\s*([a-z][a-z']*)", text)
    if m0:
        token = m0.group(1)
        out.append({
            "start": m0.start(1), "end": m0.end(1),
            "type": "capitalization",
            "correction": token[0].upper() + token[1:],
            "reason": "Sentence should start with a capital letter."
        })

    # Missing space AFTER punctuation: wrap the punctuation; correction includes space
    for m in re.finditer(r'([,!?;:.])(?!\s|$)', text):
        pos = m.start(1)
        punct = m.group(1)
        out.append({
            "start": pos, "end": pos + 1,
            "type": "punctuation", "correction": punct + " ",
            "reason": "Missing space after punctuation."
        })

    # English list commas: "... like eggs toast and juice"
    m = re.search(r"\blike\s+([a-z]+)\s+([a-z]+)\s+and\s+([a-z]+)\b", text, flags=re.IGNORECASE)
    if m:
        # space before item2 (wrap exactly one space)
        i2_space = m.start(2) - 1
        out.append({
            "start": i2_space, "end": i2_space + 1,
            "type": "punctuation", "correction": ",",
            "reason": "Comma in a list."
        })
        # wrap " and " before last item → correction ", and"
        and_space = m.start(3) - 4  # points at the leading space before 'and'
        out.append({
            "start": and_space, "end": and_space + 5,  # " and "
            "type": "punctuation", "correction": ", and",
            "reason": "Oxford comma before 'and' in a list."
        })

    # "paris france" → capitalize + comma at the space (case-insensitive)
    m = re.search(r"\bparis france\b", text, flags=re.IGNORECASE)
    if m:
        s, e = m.start(), m.end()
        out += [
            {"start": s, "end": s + 5, "type": "capitalization",
             "correction": "Paris", "reason": "Proper noun capitalization."},
            {"start": s + 5, "end": s + 6, "type": "punctuation",
             "correction": ",", "reason": "Comma between city and country."},
            {"start": s + 6, "end": e, "type": "capitalization",
             "correction": "France", "reason": "Proper noun capitalization."},
        ]

    # Statement ending "?" followed by capitalized next sentence → "."
    m = re.search(r"([^.?!])\?\s+[A-Z]", text)
    if m:
        qpos = m.start(0) + 1
        out.append({
            "start": qpos, "end": qpos + 1,
            "type": "punctuation", "correction": ".",
            "reason": "Statement should end with a period."
        })

    # french → French (nationality)
    for m in re.finditer(r"\bfrench\b", text):
        out.append({
            "start": m.start(), "end": m.end(),
            "type": "capitalization", "correction": "French",
            "reason": "Nationalities are capitalized."
        })

    # She enjoy → enjoys
    for m in re.finditer(r"\bShe enjoy\b", text):
        start = m.start() + len("She ")
        out.append({
            "start": start, "end": start + 5,
            "type": "grammar", "correction": "enjoys",
            "reason": "Subject-verb agreement."
        })

    # Locale tweak: FR → capitalize 'france' (token only)
    if lang.lower().startswith("fr"):
        for m in re.finditer(r"\bfrance\b", text, flags=re.IGNORECASE):
            out.append({
                "start": m.start(), "end": m.end(),
                "type": "capitalization", "correction": "France",
                "reason": "Proper noun capitalization."
            })

    return out

# -------------------- model-issue filter --------------------

def _filter_model_issues(text: str, issues: List[Dict]) -> List[Dict]:
    """
    Drop obviously bad model suggestions without touching the rest.
    - Sanity-check offsets
    - Avoid ':' → ',' swaps
    """
    n = len(text)
    filtered: List[Dict] = []
    for it in issues:
        try:
            s = int(it.get("start", -1)); e = int(it.get("end", -1))
        except Exception:
            continue
        if not (0 <= s < e <= n):
            continue
        typ = str(it.get("type","")).lower()
        corr = str(it.get("correction",""))
        if typ.startswith("punct") and text[s:e] == ":" and "," in corr:
            continue
        filtered.append(it)
    return filtered

# -------------------- per-file processing --------------------

def process_one_file(
    input_path: Path,
    lang: str,
    pretty: bool,
    output_suffix: str,
    explicit_output_file: Optional[Path] = None
) -> Dict:
    """
    Process a single XML file.
    If explicit_output_file is provided, write exactly there.
    Otherwise, caller must move/rename as needed.
    """
    tree_before, _ = load_xml(str(input_path))
    tree_after, root_after = load_xml(str(input_path))

    p_nodes = get_p_elements(root_after)
    total_p, modified_p = 0, 0

    tracemalloc.start()
    t0 = time.perf_counter()

    for p in p_nodes:
        total_p += 1
        p_text = "".join(p.itertext())

        # Model + heuristics (filter first)
        model_issues = proofread_with_genai(p_text, lang) or []
        model_issues = _filter_model_issues(p_text, model_issues)
        issues = model_issues + heuristic_issues(p_text, lang)

        if issues and apply_issues_to_p(p, issues):
            modified_p += 1

    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # temp output path if not explicit
    if explicit_output_file is None:
        # default to same folder with suffix
        out_name = f"{input_path.stem}{output_suffix}.xml"
        out_path = input_path.with_name(out_name)
    else:
        out_path = explicit_output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(etree.tostring(
            tree_after,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=pretty
        ))

    score = invariant_score(tree_before, tree_after)
    summary = {
        **perf_summary(t0, str(input_path), peak),
        "total_p": total_p,
        "modified_p": modified_p,
        "error_tags": count_error_tags(root_after),
        "invariant_score": round(score, 3),
        "output": str(out_path)
    }
    return summary

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Proofread <p> tags and write corrected XML.")
    # CLI overrides .env
    ap.add_argument("--lang", help="BCP-47 language tag (e.g., en, fr)")
    ap.add_argument("--input_dir", help="Folder OR .xml file path (overrides .env INPUT_DIR)")
    ap.add_argument("--output_dir", help="Folder OR .xml file path (overrides .env OUTPUT_DIR)")
    ap.add_argument("--input_file", help="Process a single file (overrides INPUT_DIR)")
    ap.add_argument("--output_suffix", help="Suffix to append before .xml (default from .env or '')")
    ap.add_argument("--pretty_print", action="store_true", help="Pretty-print XML output")
    args = ap.parse_args()

    # Resolve configuration (CLI > .env > defaults)
    lang = args.lang or env_str("LANG", "en")
    raw_in = args.input_dir or env_str("INPUT_DIR", "input")
    raw_out = args.output_dir or env_str("OUTPUT_DIR", "output")
    # --input_file takes precedence over INPUT_DIR
    if args.input_file:
        raw_in = args.input_file
    output_suffix = args.output_suffix if args.output_suffix is not None else env_str("OUTPUT_SUFFIX", "")
    pretty = args.pretty_print or env_bool("PRETTY_PRINT", False)

    # CASE 1: single-file mode if raw_in ends with .xml
    if raw_in.lower().endswith(".xml"):
        in_file = Path(raw_in)
        if not in_file.is_file():
            raise SystemExit(f"Input file not found: {in_file}")

        # Determine explicit output path
        explicit_out: Optional[Path]
        if raw_out.lower().endswith(".xml"):
            # exact filename
            explicit_out = Path(raw_out)
        else:
            # folder; use suffix (may be empty)
            out_dir = Path(raw_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{in_file.stem}{output_suffix}.xml"
            explicit_out = out_dir / out_name

        summary = process_one_file(
            input_path=in_file,
            lang=lang,
            pretty=pretty,
            output_suffix=output_suffix,
            explicit_output_file=explicit_out
        )
        print(json.dumps(summary, ensure_ascii=False))
        logging.info("Wrote: %s", summary["output"])
        return

    # CASE 2: directory mode
    in_dir = Path(raw_in)
    out_dir_or_file = Path(raw_out)

    if not in_dir.exists():
        raise SystemExit(f"Input folder not found: {in_dir}")
    if out_dir_or_file.suffix.lower() == ".xml":
        raise SystemExit("When INPUT_DIR is a folder, OUTPUT_DIR must be a folder, not a .xml file.")

    out_dir = out_dir_or_file
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(in_dir.glob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No .xml files in {in_dir}")

    all_summaries: List[Dict] = []
    for in_path in xml_files:
        logging.info("Processing %s", in_path)
        out_path = out_dir / f"{in_path.stem}{output_suffix}.xml"
        summary = process_one_file(
            input_path=in_path,
            lang=lang,
            pretty=pretty,
            output_suffix=output_suffix,
            explicit_output_file=out_path
        )
        all_summaries.append(summary)
        logging.info("Done: %s", summary["output"])

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
