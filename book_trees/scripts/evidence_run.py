import requests
import json

BASE_URL = "http://localhost:8000/api/epubs"
EPUB_IDS = [1, 15, 17, 18, 20, 23]

sentences = []
seen = set()

for epub_id in EPUB_IDS:
    response = requests.get(f"{BASE_URL}/{epub_id}/relationships/")
    relationships = response.json()

    for rel in relationships:
        for ev in rel.get("evidence", []):
            text = ev.get("evidence", "").strip()
            if text and text not in seen:
                seen.add(text)
                sentences.append({
                    "text": text,
                    "meta": {
                        "epub_id": epub_id,
                        "chapter": ev.get("chapter"),
                        "specific_type": ev.get("specific_type")
                    }
                })

with open("label_studio_input.json", "w") as f:
    json.dump(sentences, f, indent=2)

print(f"Exported {len(sentences)} unique sentences across {len(EPUB_IDS)} EPUBs.")