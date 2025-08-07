import os
import re
import json
from pathlib import Path
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ---------- classification constants & helpers ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Environment-configurable defaults
CLASSIFY_MODEL_DEFAULT = os.getenv("CLASSIFY_MODEL", "gpt-5-mini")
CLASSIFY_REASONING_EFFORT_DEFAULT = os.getenv("CLASSIFY_REASONING_EFFORT")  # minimal|low|medium|high
CLASSIFY_VERBOSITY_DEFAULT = os.getenv("CLASSIFY_VERBOSITY")  # low|medium|high

_ALLOWED_EFFORT = {"minimal", "low", "medium", "high"}
_ALLOWED_VERBOSITY = {"low", "medium", "high"}

# Boilerplate prompt: header/footer are immutable; only criteria text is user-editable
DEFAULT_PROMPT_HEADER = (
    "For each numbered Instagram bio below, reply **yes** or **no**.\n\n"
)
DEFAULT_PROMPT_FOOTER = (
    "\nIf the bio does **not** clearly show no affiliation with what we desire, reply **no**.\n\n"
    "Return a single space-separated list of yes/no in the same order as the bios."
)

# Default editable criteria text
DEFAULT_CRITERIA_TEXT = (
    "**Say yes** if the bio contains an explicit Christian signal – e.g. the words Jesus, Christ, Christian, Bible, a Scripture reference (John 3:16, 1 Cor 13:4-8, etc.), ✝️ cross emoji, 'saved by grace', 'follower of Christ', or similar.\n"
    "Jesus, Christ, Christian, Bible, a Scripture reference (John 3:16, 1 Cor 13:4-8, etc.), ✝️ cross emoji, 'saved by grace', 'follower of Christ', or similar.\n"
)

# File path for storing the prompt
PROMPT_FILE_PATH = Path(__file__).parent.parent / "data" / "classification_prompt.json"

def _ensure_data_directory():
    """Ensure the data directory exists"""
    PROMPT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _build_full_prompt(criteria_text: str) -> str:
    return f"{DEFAULT_PROMPT_HEADER}{criteria_text}\n{DEFAULT_PROMPT_FOOTER}"

def _load_prompt_from_file() -> str:
    """Load the FULL prompt (header + criteria + footer)."""
    try:
        if PROMPT_FILE_PATH.exists():
            with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'criteria' in data:
                    return _build_full_prompt(data['criteria'])
                if 'prompt' in data:
                    # Legacy support
                    return data['prompt']
                return _build_full_prompt(DEFAULT_CRITERIA_TEXT)
        else:
            print(f"ℹ️ No saved prompt found at {PROMPT_FILE_PATH}, using default")
            return _build_full_prompt(DEFAULT_CRITERIA_TEXT)
    except Exception as e:
        print(f"⚠️ Error loading prompt from file: {e}, using default")
        return _build_full_prompt(DEFAULT_CRITERIA_TEXT)

def _save_criteria_to_file(criteria_text: str) -> bool:
    """Persist only the editable criteria to disk."""
    try:
        _ensure_data_directory()
        data = {
            'criteria': criteria_text,
            'last_updated': str(Path(__file__).stat().st_mtime)  # Simple timestamp
        }
        with open(PROMPT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Prompt saved to {PROMPT_FILE_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error saving prompt to file: {e}")
        return False

_current_prompt = _load_prompt_from_file()
_current_criteria = DEFAULT_CRITERIA_TEXT
try:
    if PROMPT_FILE_PATH.exists():
        with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            _current_criteria = data.get('criteria', DEFAULT_CRITERIA_TEXT)
except Exception:
    _current_criteria = DEFAULT_CRITERIA_TEXT

def get_classification_prompt() -> str:
    """Get the current classification prompt"""
    return _current_prompt

def get_editable_criteria() -> str:
    """Return only the editable criteria portion for UI editing."""
    return _current_criteria

def update_classification_prompt(new_criteria: str) -> str:
    """Update only the editable criteria; rebuild and persist the full prompt."""
    global _current_prompt, _current_criteria
    _current_criteria = new_criteria
    _current_prompt = _build_full_prompt(_current_criteria)
    if _save_criteria_to_file(_current_criteria):
        print("✅ Classification criteria updated and persisted")
    else:
        print("⚠️ Criteria updated but failed to persist")
    return _current_prompt

def reset_to_default_prompt() -> str:
    """Reset criteria to default (boilerplate remains fixed)."""
    return update_classification_prompt(DEFAULT_CRITERIA_TEXT)

CHRISTIAN_WORDS = {
    "jesus", "christ", "christian", "god", "lord", "bible",
    "believer", "disciple", "faith", "saved", "born again",
    "church", "worship"
}

BIBLE_BOOKS = {
    "genesis","exodus","leviticus","numbers","deuteronomy",
    "joshua","judges","ruth","samuel","kings","chronicles","ezra","nehemiah","esther",
    "job","psalm","psalms","proverbs","ecclesiastes","song","songs","canticles",
    "isaiah","jeremiah","lamentations","ezekiel","daniel",
    "hosea","joel","amos","obadiah","jonah","micah","nahum",
    "habakkuk","zephaniah","haggai","zechariah","malachi",
    "matthew","mark","luke","john","acts","romans",
    "corinthians","galatians","ephesians","philippians","colossians",
    "thessalonians","timothy","titus","philemon","hebrews",
    "james","peter","jude","revelation","rev"
}

BIBLE_PATTERN = re.compile(r"\b(" + "|".join(BIBLE_BOOKS) + r")(?:'s|s)?\b", re.I)

# quick single‑word flags
QUICK_KEYWORDS = {"†", "cross", "amen", "agtg", "jesusfreak", "bibleverse"} | CHRISTIAN_WORDS | BIBLE_BOOKS

def classify_profiles(
    profile_data,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
):
    """
    Adds 'is_christian' = 'yes' / 'no' to each dict in `profile_data`.
    Uses fast keyword heuristics first, then batches uncertain bios to GPT.
    """
    # Stage 1: Quick keyword check
    quick_results = {}
    unsure_bios = []
    unsure_indices = []
    
    for i, item in enumerate(profile_data):
        uname = item["username"]
        bio = (item["bio"] or "").strip()
        clean = re.sub(r"[^\w\s]", " ", bio).strip()
        
        # Check for obvious Christian keywords
        has_christian_keywords = (
            any(k in bio.lower() for k in QUICK_KEYWORDS) or 
            BIBLE_PATTERN.search(bio)
        )
        
        if has_christian_keywords:
            quick_results[uname] = "yes"
        else:
            # No obvious keywords found - need LLM analysis
            unsure_bios.append(clean)
            unsure_indices.append(i)
            quick_results[uname] = "no"  # Default to no, will be updated by LLM

    # Stage 2: LLM classification for uncertain bios
    if unsure_bios:
        print(f"🔍 Quick check found {len(profile_data) - len(unsure_bios)} obvious matches")
        print(f"🤔 Sending {len(unsure_bios)} uncertain bios to LLM for analysis")
        
        prompt = get_classification_prompt()
        payload = "\n".join(f"{i+1}) {b}" for i, b in enumerate(unsure_bios))
        try:
            # Resolve configuration from parameters or environment
            selected_model = (model or CLASSIFY_MODEL_DEFAULT).strip()
            effort = (reasoning_effort or CLASSIFY_REASONING_EFFORT_DEFAULT)
            effort = effort.lower().strip() if isinstance(effort, str) else None
            if effort not in _ALLOWED_EFFORT:
                effort = None

            verb = (verbosity or CLASSIFY_VERBOSITY_DEFAULT)
            verb = verb.lower().strip() if isinstance(verb, str) else None
            if verb not in _ALLOWED_VERBOSITY:
                verb = None

            request_kwargs = {
                "model": selected_model,
                # Combine instructions and payload for the Responses API input
                "input": f"{prompt}\n\n{payload}",
            }
            if effort:
                request_kwargs["reasoning"] = {"effort": effort}
            if verb:
                request_kwargs["text"] = {"verbosity": verb}

            resp = client.responses.create(**request_kwargs)
            print(f"✅ OpenAI OK — model {resp.model}")

            # Extract plain text output from Responses API
            output_text = (getattr(resp, "output_text", None) or "").strip()
            if not output_text:
                # Fallback: attempt to reconstruct from output items if needed
                try:
                    output_items = getattr(resp, "output", [])
                    chunks = []
                    for item in output_items or []:
                        content = item.get("content") if isinstance(item, dict) else None
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "output_text":
                                    chunks.append(c.get("text", ""))
                    output_text = "".join(chunks).strip()
                except Exception:
                    output_text = ""

            flags = (output_text or "").strip().split()
            
            # Validate length
            if len(flags) != len(unsure_bios):
                print(f"⚠️ Length mismatch: got {len(flags)}, expected {len(unsure_bios)}")
                flags = ["no"] * len(unsure_bios)  # Default to no for mismatched results
            # Enforce backend rule: only explicit 'yes' counts as yes; everything else => 'no'
            flags = [
                "yes" if isinstance(f, str) and f.strip().lower().startswith("y") else "no"
                for f in flags
            ]
                
        except Exception as e:
            print(f"❗️GPT batch failed ({type(e).__name__}: {e}); using no for all uncertain bios.")
            flags = ["no"] * len(unsure_bios)

        # Update results for uncertain bios
        for i, flag in enumerate(flags):
            original_index = unsure_indices[i]
            username = profile_data[original_index]["username"]
            quick_results[username] = flag.lower()

    # attach the flag onto each item
    for item in profile_data:
        item["is_christian"] = quick_results.get(item["username"], "no")

    return [i["username"] for i in profile_data
                    if i["is_christian"] == "yes"]

