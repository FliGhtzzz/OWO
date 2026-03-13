
import os
import sys

# Ensure database.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import _get_es, ES_INDEX

# Provided lists
animals1 = [
  "Aardvark", "Agouti", "Aye-aye", "Babirusa", "Bactrian Camel", "Binturong (Bearcat)", "Bilby", "Blue Whale", "Bushbaby", "Chachalaca", "Clouded Leopard", "Coati", "Coral Snake", "Crab-eating Macaque", "Crested Auklet", "Crowned Crane", "Dall Sheep", "Demoiselle Crane", "Dhole", "Dugong", "Eared Grebe", "Emperor Penguin", "Fossa", "Frigatebird", "Galago", "Ghost Shark", "Goeldi's Monkey", "Greater One-horned Rhino", "Guianacara", "Harpy Eagle", "Humpback Dolphin", "Iberian Lynx", "Ili Pika", "Jerboa", "Kiang", "Kinkajou", "Kob (antelope)", "Lammergeier (Bearded Vulture)", "Lechwe", "Lesser Grison", "Lion's Mane Jellyfish", "Long-eared Jerboa", "Lyrebird", "Manatee", "Marmoset", "Magellanic Penguin", "Mantis Shrimp", "Marbled Murrelet", "Mekong Giant Catfish", "Monarch Butterfly", "Narwhal", "Nene (Hawaiian Goose)", "Okapi", "Olive Baboon", "Osprey", "Paca", "Pangolin", "Pika (American)", "Potoo", "Pronghorn", "Proboscis Monkey", "Quokka", "Rafflesia (Parasitic Flower)", "Red Panda", "Rodrigues Flying Fox", "Saiga", "Saltwater Crocodile", "Secretary Bird", "Shoebill", "Snow Leopard", "Spectacled Bear", "Springbok", "Squirrel Monkey", "Starfish", "Sumatran Orangutan", "Tapir", "Tarsier", "Thorny Devil (lizard)", "Tree Kangaroo", "Tuatara", "Uakari", "Umbrella Cockatoo", "Vampire Bat", "Viscacha", "Walrus", "White-faced Capuchin Monkey", "Wolf Spider", "Wrangell Island Caribou", "X-Ray Tetra (fish)", "Yak", "Yaranacu", "Zebra Longwing (butterfly)", "Zorse", "Axolotl", "Narambai", "Blobfish"
]

animals2 = [
    "Lion", "Tiger", "Elephant", "Giraffe", "Zebra", "Panda", "Kangaroo", "Koala", "Cheetah", "Hippo", "Rhino", "Camel", "Wolf", "Fox", "Monkey", "Gorilla", "Chimpanzee", "Otter", "Raccoon", "Hedgehog", "Bat", "Whale", "Dolphin", "Seal", "Sea Lion", "Walrus", "Platypus", "Anteater", "Armadillo", "Sloth", "Reindeer", "Moose", "Antelope", "Bison", "Yak", "Polar Bear", "Brown Bear", "Meerkat", "Groundhog", "Mole", "Birds", "Eagle", "Owl", "Penguin", "Ostrich", "Flamingo", "Peacock", "Parrot", "Hummingbird", "Swan", "Kingfisher", "Woodpecker", "Pelican", "Seagull", "Swallow", "Sparrow", "Crow", "Pigeon", "Crane", "Stork", "Falcon", "Crocodile", "Turtle", "Tortoise", "Chameleon", "Iguana", "Gecko", "Rattlesnake", "Cobra", "Python", "Frog", "Toad", "Salamander", "Newt", "Great White Shark", "Hammerhead Shark", "Stingray", "Sea Horse", "Clownfish", "Goldfish", "Eel", "Octopus", "Squid", "Jellyfish", "Starfish", "Sea Urchin", "Lobster", "Crab", "Butterfly", "Bee", "Dragonfly", "Mantis", "Firefly", "Ant", "Ladybug", "Cicada", "Spider", "Scorpion", "Centipede", "Dog", "Cat"
]

all_animals = sorted(list(set(animals1 + animals2)))

def generate_report():
    es = _get_es()
    if not es:
        print("Elasticsearch not available")
        return

    report_lines = []
    total_matches = 0
    animals_with_photos = 0
    
    for animal in all_animals:
        # Use term query for exact keyword matching (case-insensitive because analyzer is lowercase)
        body = {"query": {"term": {"keywords": animal.lower()}}}
        res = es.count(index=ES_INDEX, body=body)
        count = res['count']
        report_lines.append((animal, count))
        total_matches += count
        if count > 0:
            animals_with_photos += 1

    # Sort report: count desc, then name asc
    report_lines.sort(key=lambda x: (-x[1], x[0]))

    print(f"# 全球動物標籤統計報告\n")
    print(f"- **提供的動物種類總數**: {len(all_animals)}")
    print(f"- **有照片的動物種類**: {animals_with_photos}")
    print(f"- **無照片的動物種類**: {len(all_animals) - animals_with_photos}")
    print(f"- **總計標籤匹配次數 (Total Label Count)**: {total_matches}\n")
    
    print("| 動物名稱 | 標籤數量 (cnt) |")
    print("| :--- | :--- |")
    for animal, count in report_lines:
        print(f"| {animal} | {count} |")

if __name__ == "__main__":
    generate_report()
