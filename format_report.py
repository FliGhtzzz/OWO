
import json

with open('animal_counts.json', 'r', encoding='utf-16') as f:
    data = json.load(f)

# Sort by count descending
sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

print("# Animal Photo Counts\n")
print("| Animal | Count |")
print("| :--- | :--- |")
for animal, count in sorted_data:
    print(f"| {animal} | {count} |")
