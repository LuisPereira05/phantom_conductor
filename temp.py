import json

with open("gesture_templates.json") as f:
    data = json.load(f)
for g in data:
    print(g["name"], "-> pool[0] length:", len(g["pool"][0]))
