import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# ---------- classification constants & helpers ----------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def classify_profiles(profile_data, model: str = "gpt-4o"):
    """
    Adds 'is_christian' = 'yes' / 'no' to each dict in `profile_data`.
    Uses fast keyword heuristics first, then batches uncertain bios to GPT.
    """
    unsure_bios = []
    order = []

    for item in profile_data:
        uname = item["username"]
        bio = (item["bio"] or "").strip()
        clean = re.sub(r"[^\w\s]", " ", bio).strip()
        order.append(uname)
        unsure_bios.append(clean)

    # map of username → flag
    verdicts = {}

    if unsure_bios:
        prompt = (
            "For each tagged Instagram bio below answer **yes** or **no**.\n"
            "Say **yes** **only** when BOTH of the following are true:\n"
            "  1. The bio clearly belongs to a *student* (college).\n"
            "     - clues: \"class of 2027\", \"'28\", \"freshman\", \"senior\", etc.\n"
            "  2. The bio contains an explicit Christian signal (e.g. Jesus, Christ, ✝️, Bible verse).\n"
            "All other cases – including churches, ministries, businesses, adults, or students without\n"
            "Christian references – must be **no**.\n"
            "Return one space-separated list of tag|label pairs in the same order, e.g. #A01#|yes #A02#|no ..."
        )
        
        # Create tagged payload with unambiguous IDs
        payload_lines = []
        for i, bio in enumerate(unsure_bios, 1):
            tag = f"#A{i:02d}#"  # → #A01#, #A02#, etc.
            payload_lines.append(f"{tag} {bio}")
        payload = "\n".join(payload_lines)
        
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ],
            )
            print(f"✅ OpenAI OK — model {resp.model}")
            
            # Parse tag|label pairs
            response_text = resp.choices[0].message.content.strip()
            pairs = response_text.split()
            
            # Validate length before trusting the output
            if len(pairs) != len(unsure_bios):
                raise ValueError(f"Model returned {len(pairs)} flags for {len(unsure_bios)} bios")
            
            # Parse each pair and recover original index
            for pair in pairs:
                if "|" not in pair:
                    raise ValueError(f"Invalid pair format: {pair}")
                tag, label = pair.split("|", 1)
                if not tag.startswith("#A") or not tag.endswith("#"):
                    raise ValueError(f"Invalid tag format: {tag}")
                try:
                    idx = int(tag[2:-1]) - 1  # Extract number from #A01# → 0
                    if 0 <= idx < len(unsure_bios):
                        verdicts[order[idx]] = label.lower()
                except ValueError:
                    raise ValueError(f"Invalid tag number: {tag}")
                    
        except Exception as e:
            print(f"❗️GPT batch failed ({type(e).__name__}: {e}); fallback to keywords.")
            # Fallback to keyword-based classification
            for i, bio in enumerate(unsure_bios):
                is_christian = "yes" if any(k in bio.lower() for k in CHRISTIAN_WORDS) or BIBLE_PATTERN.search(bio) else "no"
                verdicts[order[i]] = is_christian

    # attach the flag onto each item
    for item in profile_data:
        item["is_christian"] = verdicts.get(item["username"], "no")

    return [i["username"] for i in profile_data
                    if i["is_christian"] == "yes"]
