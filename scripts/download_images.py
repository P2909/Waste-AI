from icrawler.builtin import BingImageCrawler
from pathlib import Path
import time

# Queries for each waste category
queries = {
    "organic": [
        "banana peels waste", "vegetable scraps waste", "food waste compost"
    ],
    "hazardous": [
        "battery waste", "medical waste syringes", "pesticide bottle waste", "paint can waste"
    ],
    "recyclable": [
        "plastic bottle recycling", "cardboard recycling", "paper recycling",
        "metal can recycling", "glass bottle recycling"
    ],
    "general": [
        "mixed trash", "non recyclable waste", "dirty tissue waste", "broken ceramic waste"
    ]
}

def download_images(category, queries_list, base_dir, max_images=80):
    """
    Downloads images for the given category and list of queries using BingImageCrawler.
    Retries up to 3 times if an error occurs.
    """
    out_dir = base_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    for query in queries_list:
        print(f"Downloading '{query}' into '{out_dir}'...")
        success = False
        attempts = 0

        while not success and attempts < 3:  # Retry up to 3 times
            try:
                crawler = BingImageCrawler(storage={"root_dir": str(out_dir)})
                crawler.crawl(keyword=query, max_num=max_images)
                success = True
                print(f"✅ Done: {query}")
            except Exception as e:
                attempts += 1
                print(f"⚠️ Error downloading '{query}': {e}")
                if attempts < 3:
                    print(f"🔄 Retrying... ({attempts}/3)")
                    time.sleep(5)
                else:
                    print(f"❌ Skipping '{query}' after 3 failed attempts.")

        # Short delay between queries to avoid rate limiting
        time.sleep(2)

if __name__ == "__main__":
    base_path = Path("data/raw/delhi")
    for cls, qs in queries.items():
        download_images(cls, qs, base_path)
