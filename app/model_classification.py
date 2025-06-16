import re
from openai import OpenAI

# ---------- classification constants & helpers ----------
client = OpenAI()

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
    definite = []        # (username, flag)
    unsure_bios = []     # bios queued for GPT
    order = []           # usernames in GPT order

    for item in profile_data:
        uname = item["username"]
        bio = (item["bio"] or "").strip()
        bio_lc = bio.lower()

        if not bio:
            definite.append((uname, "no"))
            continue

        if any(k in bio_lc for k in QUICK_KEYWORDS) or BIBLE_PATTERN.search(bio):
            definite.append((uname, "yes"))
            continue

        # possible hit → send to GPT
        clean = re.sub(r"[^\w\s]", " ", bio).strip()
        order.append(uname)
        unsure_bios.append(clean)

    # map of username → flag
    verdicts = {u: f for u, f in definite}

    if unsure_bios:
        prompt = (
            "Classify each numbered Instagram bio below as yes or no "
            "(Christian affiliation detected). "
            "Return a space‑separated list of 'yes' or 'no' in the same order."
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
        except Exception as e:
            print(f"❗️GPT batch failed ({type(e).__name__}: {e}); fallback to keywords.")
            flags = [
                "yes" if any(k in b.lower() for k in CHRISTIAN_WORDS) or BIBLE_PATTERN.search(b) else "no"
                for b in unsure_bios
            ]

        for uname, flag in zip(order, flags):
            verdicts[uname] = flag.lower()

    # attach the flag onto each item
    for item in profile_data:
        item["is_christian"] = verdicts.get(item["username"], "no")

    return [i["username"] for i in profile_data
                    if i["is_christian"] == "yes"]