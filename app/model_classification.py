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

def classify_profiles(profile_data, model: str = "gpt-4.1-mini"):
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
        
        prompt = (
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
        payload = "\n".join(f"{i+1}) {b}" for i, b in enumerate(unsure_bios))
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": payload},
                ],
            )
            print(f"✅ OpenAI OK — model {resp.model}")
            flags = resp.choices[0].message.content.strip().split()
            
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

