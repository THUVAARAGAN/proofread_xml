import os, json, logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
client = OpenAI(api_key=api_key)

_ALLOWED_TYPES = {"grammar","spelling","punctuation","capitalization","clarity"}

def _norm_type(t: str) -> str:
    t = (t or "").strip().lower()
    if t in {"verb agreement","agreement","syntax"}: return "grammar"
    if t in {"spell","typo"}: return "spelling"
    if t in {"punct","punc"}: return "punctuation"
    if t in {"caps","case"}: return "capitalization"
    if t not in _ALLOWED_TYPES: return "grammar"
    return t

def _postprocess(text: str, issues: list[dict]) -> list[dict]:
    n = len(text)
    cleaned = []
    for it in issues:
        try:
            s = int(it["start"]); e = int(it["end"])
        except Exception:
            continue
        if not (0 <= s < e <= n):
            continue
        cleaned.append({
            "start": s,
            "end": e,
            "type": _norm_type(it.get("type","")),
            "correction": str(it.get("correction","")),
            "reason": str(it.get("reason","")),
        })
    # de-overlap: keep leftmost non-overlapping
    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    out, last_end = [], -1
    for it in cleaned:
        if it["start"] < last_end:  # overlap -> drop later item
            continue
        out.append(it); last_end = it["end"]
    return out

def proofread_with_genai(text: str, lang: str = "en") -> list[dict]:
    """
    Returns [{"start":int,"end":int,"type":str,"correction":str,"reason":str}, ...]
    Offsets are 0-based, end-exclusive.
    """
    system = (
        f"You are a strict proofreading engine for {lang}. "
        "Given ONE paragraph string, detect only genuine errors in grammar, spelling, punctuation, "
        "capitalization, and clarity. Respond ONLY with a JSON object containing an array under the key "
        "\"issues\". Each item must have keys: \"start\" (int), \"end\" (int, exclusive), "
        "\"type\" (one of grammar|spelling|punctuation|capitalization|clarity), "
        "\"correction\" (string), and \"reason\" (string). "
        "Character offsets are over the EXACT provided text. "
        "Choose the SMALLEST span: a single word for word-like errors; a single punctuation mark "
        "or a contiguous whitespace run for punctuation; never select letters for punctuation spans. "
        "Do not rewrite the whole sentence."
    )
    user_prompt = {"lang": lang, "text": text}

    try:
        logging.info("Sending request to OpenAI...")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)}
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        issues = data.get("issues", [])
        logging.info("Received issues: %s", len(issues))

        # Coerce + sanitize
        coerced = []
        for it in issues:
            try:
                coerced.append({
                    "start": int(it["start"]),
                    "end": int(it["end"]),
                    "type": str(it.get("type","")),
                    "correction": str(it.get("correction","")),
                    "reason": str(it.get("reason","")),
                })
            except Exception:
                continue

        return _postprocess(text, coerced)

    except Exception as e:
        logging.error("GenAI Error: %s", e)
        return []
