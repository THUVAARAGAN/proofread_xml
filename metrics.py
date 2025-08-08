import time
from pathlib import Path
from typing import Dict
from lxml import etree

# -------- Performance summary --------

def perf_summary(start_time: float, file_path: str, peak_bytes: int) -> Dict:
    """
    Build a one-file performance summary.
    - start_time: value from time.perf_counter() taken before processing
    - file_path:  path to the input file (any string)
    - peak_bytes: peak memory in bytes from tracemalloc.get_traced_memory()[1]
    """
    duration = time.perf_counter() - start_time
    return {
        "file": Path(file_path).name,
        "time_seconds": round(duration, 4),
        "mem_peak_mb": round(peak_bytes / (1024 * 1024), 2),
    }

# -------- Invariant scoring --------

def _flatten_p_text(p: etree._Element) -> str:
    """
    Concatenate the visible text inside <p>, treating <error> like plain text:
    - include p.text
    - for each child:
        - if child.tag local-name == 'error': include child.text and child.tail
        - otherwise: include child.text and child.tail
    This effectively ignores tags/attributes and reconstructs the original string.
    """
    buf = []
    if p.text:
        buf.append(p.text)
    for node in p:
        tag = node.tag
        # local-name() check without XPath for speed/simplicity
        local = tag.split('}', 1)[1] if isinstance(tag, str) and '}' in tag else str(tag)
        # For <error>, we still include its inner text (original slice) and tail
        if local == "error":
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
    """
    Fraction of <p> elements whose reconstructed text (ignoring <error> tags/attrs)
    is EXACTLY identical before vs after. 1.0 == perfect.
    Returns 0.0 if counts of <p> differ or there are no <p> elements.
    """
    root_b = tree_before.getroot()
    root_a = tree_after.getroot()
    before_ps = root_b.xpath(".//*[local-name()='p']")
    after_ps  = root_a.xpath(".//*[local-name()='p']")

    if len(before_ps) != len(after_ps) or not before_ps:
        return 0.0

    ok = 0
    for pb, pa in zip(before_ps, after_ps):
        if _flatten_p_text(pb) == _flatten_p_text(pa):
            ok += 1
    return ok / len(before_ps)

# -------- Counts --------

def count_error_tags(root: etree._Element) -> int:
    """Count all <error> tags anywhere in the document."""
    return len(root.xpath(".//*[local-name()='error']"))
