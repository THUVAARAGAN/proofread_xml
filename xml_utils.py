from lxml import etree
from dataclasses import dataclass
from typing import List, Tuple
import re

# --------- BE-verb guard (prevents bogus "hard -> are" swaps) ---------
_BE = {"am", "is", "are", "was", "were", "be", "been", "being"}

def _reject_bogus_be_swap(text: str, s: int, e: int, corr: str) -> bool:
    """
    If the correction is a BE-verb but the original slice isn't a BE-verb token,
    treat it as bogus and drop the span.
    """
    orig = text[s:e].strip().lower()
    return corr.strip().lower() in _BE and orig not in _BE

# ---------------------------------------------------------------------

@dataclass
class TextPart:
    node: etree._Element
    slot: str   # "text" | "tail"
    text: str
    start: int
    end: int

def load_xml(file_path: str):
    parser = etree.XMLParser(remove_blank_text=False, strip_cdata=False, recover=True, ns_clean=False)
    tree = etree.parse(file_path, parser)
    return tree, tree.getroot()

def get_p_elements(root):
    # namespace-agnostic
    return root.xpath(".//*[local-name()='p']")

def _collect_text_parts(p: etree._Element) -> tuple[str, List[TextPart]]:
    """
    Flatten the visible text INSIDE <p>. We intentionally DO NOT include p.tail,
    because that text is outside the <p> element and would cause span leakage.
    """
    parts: List[TextPart] = []
    buf: List[str] = []
    pos = 0

    def push(node, slot, s):
        nonlocal pos
        if s:
            start = pos
            buf.append(s)
            pos += len(s)
            parts.append(TextPart(node=node, slot=slot, text=s, start=start, end=pos))

    for elem in p.iter():
        if elem is p:
            # include only p.text (not p.tail)
            push(elem, "text", elem.text)
        else:
            # for descendants, include both text and tail
            push(elem, "text", elem.text)
            push(elem, "tail", elem.tail)

    return "".join(buf), parts

def _set_slot(node, slot, value):
    if slot == "text":
        node.text = value
    else:
        node.tail = value

def _get_slot(node, slot):
    return node.text if slot == "text" else node.tail

def _recollect(p):
    return _collect_text_parts(p)

def _insert_holder_after_node_or_inside_p(p: etree._Element, target_node: etree._Element, slot: str, holder: etree._Element):
    """
    Insert the temporary split-holder safely:
    - If splitting p.text (target_node is p and slot=='text'), place holder as a child INSIDE <p>.
    - Otherwise, insert as the next sibling of the target node.
    """
    if target_node is p and slot == "text":
        if len(p) > 0:
            p.insert(0, holder)
        else:
            p.append(holder)
        return

    parent = target_node.getparent()
    if parent is None:
        parent = p
    siblings = list(parent)
    idx = siblings.index(target_node) if target_node in siblings else -1
    if idx >= 0:
        parent.insert(idx + 1, holder)
    else:
        target_node.addnext(holder)

def _insert_error_after_tail_node_in_order(base_node: etree._Element, new_err: etree._Element, p: etree._Element):
    """
    Ensure multiple <error> nodes inserted after the same base node keep their
    original order: insert after the last consecutive <error> sibling (if any).
    """
    parent = base_node.getparent() or p
    # Walk forward over consecutive <error> siblings
    ref = base_node
    nxt = ref.getnext()
    last = ref
    while nxt is not None:
        tag = nxt.tag
        local = tag.split('}', 1)[1] if isinstance(tag, str) and '}' in tag else str(tag)
        if local != "error":
            break
        last = nxt
        nxt = nxt.getnext()
    last.addnext(new_err)

def _wrap_span(p: etree._Element, parts: List[TextPart], span: Tuple[int, int], err):
    s0, e0 = span
    if s0 >= e0:
        return

    # Find all parts that intersect [s0, e0)
    affected = [tp for tp in parts if not (e0 <= tp.start or s0 >= tp.end)]
    if not affected:
        return

    # ---------- Align first boundary ----------
    first = affected[0]
    first_text = _get_slot(first.node, first.slot) or ""
    left_len = s0 - first.start  # chars to keep on the LEFT side (outside span)
    if 0 < left_len < len(first_text):
        _set_slot(first.node, first.slot, first_text[:left_len])

        holder = etree.Element("split-holder")
        if first.slot == "text":
            holder.text = first_text[left_len:]
            holder.tail = None
        else:
            holder.text = None
            holder.tail = first_text[left_len:]

        _insert_holder_after_node_or_inside_p(p, first.node, first.slot, holder)

        # Recompute after structural change
        _, new_parts = _recollect(p)
        parts[:] = new_parts
        affected[:] = [tp for tp in parts if not (e0 <= tp.start or s0 >= tp.end)]

    # ---------- Align last boundary ----------
    if affected:
        last = affected[-1]
        last_text = _get_slot(last.node, last.slot) or ""
        right_len = last.end - e0  # chars to keep on the RIGHT side (outside span)
        if 0 < right_len < len(last_text):
            _set_slot(last.node, last.slot, last_text[:-right_len])

            holder = etree.Element("split-holder")
            if last.slot == "text":
                holder.text = last_text[-right_len:]
                holder.tail = None
            else:
                holder.text = None
                holder.tail = last_text[-right_len:]

            _insert_holder_after_node_or_inside_p(p, last.node, last.slot, holder)

            _, new_parts = _recollect(p)
            parts[:] = new_parts
            affected[:] = [tp for tp in parts if not (e0 <= tp.start or s0 >= tp.end)]

    # ---------- Now the span should be whole parts ----------
    span_parts = [tp for tp in parts if s0 <= tp.start and tp.end <= e0]
    if not span_parts:
        return

    # Wrap EACH part separately to avoid crossing element boundaries
    for tp in span_parts:
        if not tp.text:
            continue

        err_el = etree.Element("error")
        err_el.set("type", err["type"])
        err_el.set("correction", err.get("correction", ""))
        if err.get("reason"):
            err_el.set("reason", err["reason"])
        err_el.text = tp.text  # preserve exact original slice

        # zero out original slice
        _set_slot(tp.node, tp.slot, "")

        # insert error element adjacent to the slice, preserving order
        if tp.slot == "text":
            # Append to keep insertion order for multiple slices from the same node
            tp.node.append(err_el)
        else:
            # Insert after the last consecutive <error> sibling after this node
            _insert_error_after_tail_node_in_order(tp.node, err_el, p)

    # ---------- Clean up temporary split-holder nodes ----------
    for holder in list(p.iterfind(".//split-holder")):
        prev = holder.getprevious()
        nxt = holder.getnext()
        par = holder.getparent()

        if holder.text:
            if prev is not None:
                prev.tail = (prev.tail or "") + holder.text
            else:
                par.text = (par.text or "") + holder.text
        if holder.tail:
            if nxt is not None:
                nxt.tail = (nxt.tail or "") + holder.tail
            else:
                par.tail = (par.tail or "") + holder.tail

        par.remove(holder)

# -------- Normalization helpers (prevent mid-word slices) --------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_']+")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_SET = set(",.;:?!")

def _norm_type(t: str) -> str:
    t = (t or "").strip().lower()
    if t in {"grammar", "verb agreement", "agreement", "syntax"}:
        return "grammar"
    if t in {"spelling", "spell", "typo"}:
        return "spelling"
    if t in {"punctuation", "punct"}:
        return "punctuation"
    if t in {"capitalization", "caps", "case"}:
        return "capitalization"
    if t in {"clarity", "style"}:
        return "clarity"
    return t or "grammar"

def _expand_token_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start,end) to cover the whole token if offsets are inside a token."""
    s, e = start, end
    # expand left
    while s > 0 and (text[s-1].isalnum() or text[s-1] in "_'"):
        s -= 1
    # expand right
    n = len(text)
    while e < n and (text[e].isalnum() or text[e] in "_'"):
        e += 1
    return s, e

def _nearest_space_or_punct(text: str, start: int, end: int) -> tuple[int, int] | None:
    """
    For punctuation fixes: prefer a punctuation mark first, then a space run.
    Never return letters.
    """
    n = len(text)

    # 1) nearest single punctuation near the offsets
    for k in range(max(0, start - 1), min(n, end + 1)):
        if text[k] in _PUNCT_SET:
            return k, k + 1

    # 2) space run starting at end
    m = _SPACE_RE.match(text, end)
    if m:
        return m.start(), m.end()

    # 3) space run ending at start
    i = start - 1
    while i >= 0 and text[i] == " ":
        i -= 1
    if i + 1 < start:
        return i + 1, start

    return None

def _normalize_spans_for_leaf(text: str, issues: List[dict]) -> List[dict]:
    """
    Normalize spans:
    - word-like types -> expand to whole tokens (no mid-word)
    - punctuation -> target a punctuation mark or a space run; never letters
    - canonicalize type; drop overlaps
    - drop bogus BE-verb swaps (e.g., 'hard' -> 'are')
    """
    cleaned: List[dict] = []
    for it in issues:
        try:
            s = int(it["start"]); e = int(it["end"])
        except Exception:
            continue
        if not (0 <= s < e <= len(text)):
            continue

        typ = _norm_type(it.get("type"))
        corr = str(it.get("correction", ""))
        reason = str(it.get("reason") or "")

        if typ in {"grammar", "spelling", "capitalization", "clarity"}:
            ns, ne = _expand_token_bounds(text, s, e)
            if ns == ne:
                continue
            # avoid single-letter slices for alpha content
            if ne - ns == 1 and text[ns].isalpha():
                ns, ne = _expand_token_bounds(text, ns, ne)

        elif typ == "punctuation":
            pick = _nearest_space_or_punct(text, s, e)
            if pick:
                ns, ne = pick
            else:
                # fallback: single non-letter if possible; else skip
                if not text[s].isalpha():
                    ns, ne = s, min(s + 1, len(text))
                else:
                    # try to find nearest space in a small window
                    left = s
                    while left > 0 and text[left-1] != " ":
                        left -= 1
                    right = e
                    while right < len(text) and text[right] != " ":
                        right += 1
                    if left < s:
                        ns, ne = left, s
                    elif right > e:
                        ns, ne = e, right
                    else:
                        continue  # give up if we’d wrap letters

            # FINAL GUARD: ensure punctuation span has no letters
            if any(ch.isalpha() for ch in text[ns:ne]):
                # shrink to nearest single punctuation if present
                found = None
                for k in range(ns, ne):
                    if text[k] in _PUNCT_SET:
                        found = (k, k+1); break
                if found:
                    ns, ne = found
                else:
                    continue

        else:
            typ = "grammar"
            ns, ne = _expand_token_bounds(text, s, e)
            if ns == ne:
                continue

        if ns >= ne:
            continue

        # Drop bogus BE-verb swaps like "hard" -> "are"
        if typ == "grammar" and _reject_bogus_be_swap(text, ns, ne, corr):
            continue

        cleaned.append({
            "start": ns,
            "end": ne,
            "type": typ,
            "correction": corr,
            "reason": reason
        })

    # de-overlap: keep left-most non-overlapping
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    out: List[dict] = []
    last_end = -1
    for c in cleaned:
        if c["start"] < last_end:
            continue
        out.append(c)
        last_end = c["end"]
    return out

# -------- Leaf-mode helpers (no text mutation) --------

def _p_is_leaf(p: etree._Element) -> bool:
    """True if <p> has no child elements (pure text only)."""
    for _ in p.iterchildren():
        return False
    return True

def _apply_leaf_wrapping(p: etree._Element, issues: List[dict]) -> bool:
    """
    Safe path for <p> with only text.
    Rebuilds mixed content (text + <error>) strictly from the original text and spans.
    Never changes characters; corrections live only in attributes.
    """
    original = p.text or ""
    if not original:
        return False

    spans = sorted(issues, key=lambda x: (int(x["start"]), int(x["end"])))
    spans = [s for s in spans if 0 <= int(s["start"]) < int(s["end"]) <= len(original)]

    out_chunks: List[object] = []
    cursor = 0
    for s in spans:
        a, b = int(s["start"]), int(s["end"])
        if a < cursor:
            continue

        if cursor < a:
            out_chunks.append(original[cursor:a])

        err_el = etree.Element("error")
        err_el.set("type", s["type"])
        err_el.set("correction", s["correction"])
        reason = s.get("reason")
        if reason:
            err_el.set("reason", reason)
        err_el.text = original[a:b]
        out_chunks.append(err_el)

        cursor = b

    if cursor < len(original):
        out_chunks.append(original[cursor:])

    if not any(isinstance(x, etree._Element) for x in out_chunks):
        return False

    # Rebuild <p> content
    for child in list(p):
        p.remove(child)

    idx = 0
    if idx < len(out_chunks) and isinstance(out_chunks[0], str):
        p.text = out_chunks[0]
        idx = 1
    else:
        p.text = None

    last_el = None
    while idx < len(out_chunks):
        item = out_chunks[idx]
        if isinstance(item, etree._Element):
            p.append(item)
            last_el = item
            item.tail = None
        else:
            if last_el is not None:
                last_el.tail = (last_el.tail or "") + item
            else:
                p.text = (p.text or "") + item
        idx += 1

    return True

# -------------- Public apply function --------------

def apply_issues_to_p(p: etree._Element, issues: List[dict]) -> bool:
    if not issues:
        return False

    # Skip whitespace-only paragraphs entirely
    p_text_all = "".join(p.itertext())
    if p_text_all.strip() == "":
        return False

    # Prefer leaf-mode (no child elements) — safest and never mutates characters.
    if _p_is_leaf(p):
        original = p.text or ""
        norm = _normalize_spans_for_leaf(original, issues)
        if not norm:
            return False
        return _apply_leaf_wrapping(p, norm)

    # Fallback to split-based approach for nested content:
    _, parts = _collect_text_parts(p)
    changed = False
    for it in sorted(issues, key=lambda x: (int(x["start"]), int(x["end"])), reverse=True):
        _wrap_span(p, parts, (int(it["start"]), int(it["end"])), it)
        _, parts = _collect_text_parts(p)
        changed = True
    return changed

def text_without_error_tags(p: etree._Element) -> str:
    full, _ = _collect_text_parts(p)
    return full
