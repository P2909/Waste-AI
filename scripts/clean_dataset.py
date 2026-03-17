# scripts/clean_dataset.py
import os, hashlib
from PIL import Image
from pathlib import Path

RAW = Path("data/raw/delhi")
MIN_SIDE = 64  # tiny images are not useful

def file_hash(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not RAW.exists():
        print("❌ RAW path not found:", RAW.resolve()); return

    print("📂 cleaning:", RAW.resolve())
    for cls in sorted(x for x in RAW.iterdir() if x.is_dir()):
        print(f"\n➡️ class: {cls.name}")
        seen = set()
        removed = {"corrupt":0, "small":0, "dupe":0}

        for p in list(cls.glob("*")):
            if not p.is_file(): continue
            # remove corrupt
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                p.unlink(missing_ok=True); removed["corrupt"]+=1; continue
            # reopen to check size
            with Image.open(p) as im:
                w,h = im.size
                if min(w,h) < MIN_SIDE:
                    p.unlink(missing_ok=True); removed["small"]+=1; continue
            # remove duplicates (by hash)
            hsh = file_hash(p)
            if hsh in seen:
                p.unlink(missing_ok=True); removed["dupe"]+=1
            else:
                seen.add(hsh)

        print(f"   🧹 removed -> corrupt:{removed['corrupt']} small:{removed['small']} dupes:{removed['dupe']}")

    print("\n✅ cleaning complete.")

if __name__ == "__main__":
    main()
