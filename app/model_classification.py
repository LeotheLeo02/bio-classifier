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

# Default classification prompt - can be updated via API
DEFAULT_CLASSIFICATION_PROMPT = (
    "For each numbered Instagram bio below, reply **yes** or **no**.\n"
    "\n"
    "**Say yes** if the bio contains an explicit Christian signal – e.g. the words Jesus, Christ, Christian, Bible, a Scripture reference (John 3:16, 1 Cor 13:4-8, etc.), ✝️ cross emoji, 'saved by grace', 'follower of Christ', or similar.\n"
    "Jesus, Christ, Christian, Bible, a Scripture reference (John 3:16, 1 Cor 13:4-8, etc.), "
    "✝️ cross emoji, 'saved by grace', 'follower of Christ', or similar.\n"
    "\n"
    "If the bio does **not** clearly show Christian affiliation, reply **no**.\n"
    "\n"
    "Return a single space-separated list of yes/no in the same order as the bios."
)

# File path for storing the prompt
PROMPT_FILE_PATH = Path(__file__).parent.parent / "data" / "classification_prompt.json"

def _ensure_data_directory():
    """Ensure the data directory exists"""
    PROMPT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_prompt_from_file() -> str:
    """Load the prompt from the persistent file"""
    try:
        if PROMPT_FILE_PATH.exists():
            with open(PROMPT_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('prompt', DEFAULT_CLASSIFICATION_PROMPT)
        else:
            print(f"ℹ️ No saved prompt found at {PROMPT_FILE_PATH}, using default")
            return DEFAULT_CLASSIFICATION_PROMPT
    except Exception as e:
        print(f"⚠️ Error loading prompt from file: {e}, using default")
        return DEFAULT_CLASSIFICATION_PROMPT

def _save_prompt_to_file(prompt: str) -> bool:
    """Save the prompt to the persistent file"""
    try:
        _ensure_data_directory()
        data = {
            'prompt': prompt,
            'last_updated': str(Path(__file__).stat().st_mtime)  # Simple timestamp
        }
        with open(PROMPT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Prompt saved to {PROMPT_FILE_PATH}")
        return True
    except Exception as e:
        print(f"❌ Error saving prompt to file: {e}")
        return False

# Initialize the prompt from file or default
_current_prompt = _load_prompt_from_file()

def get_classification_prompt() -> str:
    """Get the current classification prompt"""
    return _current_prompt

def update_classification_prompt(new_prompt: str) -> str:
    """Update the classification prompt and persist it"""
    global _current_prompt
    _current_prompt = new_prompt
    
    # Save to file for persistence
    if _save_prompt_to_file(new_prompt):
        print("✅ Classification prompt updated and persisted")
    else:
        print("⚠️ Classification prompt updated but failed to persist")
    
    return _current_prompt

def reset_to_default_prompt() -> str:
    """Reset the prompt to the default and persist it"""
    return update_classification_prompt(DEFAULT_CLASSIFICATION_PROMPT)

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

