import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(BASE_DIR, 'animal_summaries.json')

_animal_data = None

def _load_data():
    global _animal_data
    if _animal_data is None:
        try:
            with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
                _animal_data = json.load(f)
        except FileNotFoundError as e:
            print(f"Error loading animal data: {e}")
            _animal_data = {}
    return _animal_data

def get_summary(animal_class: str) -> str:
    data = _load_data()
    summary = data.get(animal_class)

    if summary:
        return summary
    else:
        return "No summary available for this animal."