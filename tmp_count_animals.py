
import os
import sys
import json

# Ensure database.py can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import _get_es, ES_INDEX

animals1 = [
  "Aardvark", "Agouti", "Aye-aye", "Babirusa", "Bactrian Camel", "Binturong (Bearcat)", "Bilby", "Blue Whale", "Bushbaby", "Chachalaca", "Clouded Leopard", "Coati", "Coral Snake", "Crab-eating Macaque", "Crested Auklet", "Crowned Crane", "Dall Sheep", "Demoiselle Crane", "Dhole", "Dugong", "Eared Grebe", "Emperor Penguin", "Fossa", "Frigatebird", "Galago", "Ghost Shark", "Goeldi's Monkey", "Greater One-horned Rhino", "Guianacara", "Harpy Eagle", "Humpback Dolphin", "Iberian Lynx", "Ili Pika", "Jerboa", "Kiang", "Kinkajou", "Kob (antelope)", "Lammergeier (Bearded Vulture)", "Lechwe", "Lesser Grison", "Lion's Mane Jellyfish", "Long-eared Jerboa", "Lyrebird", "Manatee", "Marmoset", "Magellanic Penguin", "Mantis Shrimp", "Marbled Murrelet", "Mekong Giant Catfish", "Monarch Butterfly", "Narwhal", "Nene (Hawaiian Goose)", "Okapi", "Olive Baboon", "Osprey", "Paca", "Pangolin", "Pika (American)", "Potoo", "Pronghorn", "Proboscis Monkey", "Quokka", "Rafflesia (Parasitic Flower)", "Red Panda", "Rodrigues Flying Fox", "Saiga", "Saltwater Crocodile", "Secretary Bird", "Shoebill", "Snow Leopard", "Spectacled Bear", "Springbok", "Squirrel Monkey", "Starfish", "Sumatran Orangutan", "Tapir", "Tarsier", "Thorny Devil (lizard)", "Tree Kangaroo", "Tuatara", "Uakari", "Umbrella Cockatoo", "Vampire Bat", "Viscacha", "Walrus", "White-faced Capuchin Monkey", "Wolf Spider", "Wrangell Island Caribou", "X-Ray Tetra (fish)", "Yak", "Yaranacu", "Zebra Longwing (butterfly)", "Zorse", "Axolotl", "Narambai", "Blobfish"
]

animals2 = [
    "Lion", "Tiger", "Elephant", "Giraffe", "Zebra", "Panda", "Kangaroo", "Koala", "Cheetah", "Hippo", "Rhino", "Camel", "Wolf", "Fox", "Monkey", "Gorilla", "Chimpanzee", "Otter", "Raccoon", "Hedgehog", "Bat", "Whale", "Dolphin", "Seal", "Sea Lion", "Walrus", "Platypus", "Anteater", "Armadillo", "Sloth", "Reindeer", "Moose", "Antelope", "Bison", "Yak", "Polar Bear", "Brown Bear", "Meerkat", "Groundhog", "Mole", "Birds", "Eagle", "Owl", "Penguin", "Ostrich", "Flamingo", "Peacock", "Parrot", "Hummingbird", "Swan", "Kingfisher", "Woodpecker", "Pelican", "Seagull", "Swallow", "Sparrow", "Crow", "Pigeon", "Crane", "Stork", "Falcon", "Crocodile", "Turtle", "Tortoise", "Chameleon", "Iguana", "Gecko", "Rattlesnake", "Cobra", "Python", "Frog", "Toad", "Salamander", "Newt", "Great White Shark", "Hammerhead Shark", "Stingray", "Sea Horse", "Clownfish", "Goldfish", "Eel", "Octopus", "Squid", "Jellyfish", "Starfish", "Sea Urchin", "Lobster", "Crab", "Butterfly", "Bee", "Dragonfly", "Mantis", "Firefly", "Ant", "Ladybug", "Cicada", "Spider", "Scorpion", "Centipede", "Dog", "Cat"
]

all_animals = sorted(list(set(animals1 + animals2)))

def count_animals():
    es = _get_es()
    if not es:
        print("Error: Elasticsearch not available")
        return

    report = {}
    for animal in all_animals:
        # Use exact match on keywords field
        body = {
            "query": {
                "term": {
                    "keywords": animal.lower()
                }
            }
        }
        res = es.count(index=ES_INDEX, body=body)
        count = res['count']
        if count > 0:
            report[animal] = count

    # Also check if any are missing or have common variations
    # For reporting, just print non-zero counts
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    count_animals()
