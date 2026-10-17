"""Pinned, small HotpotQA-style fixture corpus used by both pipelines.

The exercise intentionally keeps the corpus local so a clean checkout can run
without a hosted model or a network download.  Every evaluation question has
two labelled supporting document titles, just like the supporting-facts slice
used by HotpotQA.
"""

from __future__ import annotations

from typing import Final


DOCUMENTS: Final[dict[str, str]] = {
    "Ada Lovelace": "Ada Lovelace wrote the first published algorithm intended for Charles Babbage's Analytical Engine.",
    "Charles Babbage": "Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer.",
    "Apple Inc.": "Apple Inc. was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
    "Microsoft": "Microsoft was founded in 1975 by Bill Gates and Paul Allen.",
    "Earth": "Earth completes an orbit around the Sun in about 365 days and is the third planet from the Sun.",
    "Mars": "Mars is the fourth planet from the Sun and takes about 687 Earth days to orbit it.",
    "Nile River": "The Nile is commonly listed as the world's longest river, flowing through northeastern Africa.",
    "Amazon River": "The Amazon River is one of the world's largest rivers by discharge and flows through South America.",
    "The Beatles": "The Beatles were an English rock band formed in Liverpool in 1960.",
    "Elvis Presley": "Elvis Presley began recording and performing professionally in the 1950s and is known as the King of Rock and Roll.",
    "Python": "Python is a programming language first released in 1991 and designed by Guido van Rossum.",
    "Java": "Java is a programming language first released by Sun Microsystems in 1995.",
    "Mount Everest": "Mount Everest is the highest mountain above sea level, with its summit in the Himalayas.",
    "K2": "K2 is the second-highest mountain above sea level, after Mount Everest.",
    "Apollo 11": "Apollo 11 was the first crewed mission to land humans on the Moon in 1969.",
    "Apollo 13": "Apollo 13 was a 1970 lunar mission whose crew returned safely after an onboard oxygen-tank failure.",
    "Marie Curie": "Marie Curie pioneered research on radioactivity and was the first person to win two Nobel Prizes.",
    "Albert Einstein": "Albert Einstein developed the theories of special and general relativity.",
    "William Shakespeare": "William Shakespeare was an English playwright and poet associated with the late sixteenth and early seventeenth centuries.",
    "Homer": "Homer is the traditionally named ancient Greek poet associated with the Iliad and the Odyssey.",
    "Pacific Ocean": "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Atlantic Ocean": "The Atlantic Ocean is the second-largest ocean and separates the Americas from Europe and Africa.",
    "University of Oxford": "The University of Oxford is an ancient university in England, with teaching documented in the twelfth century.",
    "Harvard University": "Harvard University was established in 1636 in Massachusetts and is the oldest institution of higher learning in the United States.",
    "World Wide Web": "The World Wide Web was invented by Tim Berners-Lee in 1989 as a system of linked documents.",
    "Internet": "The Internet is a global network of interconnected computer networks that predates the World Wide Web.",
    "Johannes Gutenberg": "Johannes Gutenberg introduced mechanical movable-type printing to Europe in the fifteenth century.",
    "Tim Berners-Lee": "Tim Berners-Lee invented the World Wide Web while working at CERN in 1989.",
    "Ludwig van Beethoven": "Ludwig van Beethoven was a German composer whose career followed that of Wolfgang Amadeus Mozart.",
    "Wolfgang Amadeus Mozart": "Wolfgang Amadeus Mozart was born in 1756 and became a prominent classical composer.",
    "World War I": "World War I began in 1914 and ended in 1918.",
    "World War II": "World War II began in 1939 and ended in 1945.",
    "Moon": "The Moon is Earth's natural satellite and is much closer to Earth than Mars.",
    "DNA": "DNA stores hereditary information in living organisms and is generally more chemically stable than RNA.",
    "RNA": "RNA helps express genetic information and is generally less chemically stable than DNA.",
    "Sahara Desert": "The Sahara is the largest hot desert in the world and covers much of North Africa.",
    "Gobi Desert": "The Gobi is a large cold desert spanning parts of northern China and southern Mongolia.",
    "Amazon rainforest": "The Amazon rainforest is the largest tropical rainforest by area and lies mainly in South America.",
    "Congo rainforest": "The Congo rainforest is the second-largest tropical rainforest and lies in central Africa.",
    "Tesla Motors": "Tesla Motors was founded in 2003 and develops electric vehicles and energy products.",
    "Ford Motor Company": "Ford Motor Company was founded in 1903 by Henry Ford and his associates.",
    "Rome": "Rome was founded, according to traditional accounts, in 753 BCE and became the centre of the Roman Republic and Empire.",
    "Athens": "Athens is an ancient Greek city with a history extending back more than three thousand years.",
    "Linux": "The Linux kernel was first released by Linus Torvalds in 1991.",
    "Windows": "Microsoft Windows was first released as a graphical operating environment in 1985.",
    "Eiffel Tower": "The Eiffel Tower was completed in Paris in 1889 for the Exposition Universelle.",
    "Statue of Liberty": "The Statue of Liberty was dedicated in New York Harbor in 1886.",
    "Gold": "Gold is a dense, yellow chemical element with the symbol Au and atomic number 79.",
    "Silver": "Silver is a bright metallic chemical element with the symbol Ag and atomic number 47.",
    "Venus": "Venus has the hottest planetary surface in the Solar System because of its dense carbon-dioxide atmosphere.",
    "Mercury": "Mercury is the closest planet to the Sun and has a very thin exosphere.",
    "Jupiter": "Jupiter is the largest planet in the Solar System.",
    "Saturn": "Saturn is the sixth planet from the Sun and is famous for its prominent ring system.",
    "French Revolution": "The French Revolution began in 1789.",
    "American Revolution": "The American Revolution began in 1775 and led to the independence of the United States.",
    "Mona Lisa": "Leonardo da Vinci painted the Mona Lisa in the early sixteenth century.",
    "The Starry Night": "Vincent van Gogh painted The Starry Night in 1889.",
    "Modern Olympic Games": "The first modern Olympic Games were held in Athens in 1896.",
    "FIFA World Cup": "The first FIFA World Cup was held in Uruguay in 1930.",
    "JavaScript": "JavaScript was first released in 1995, several years after Python.",
    "Himalayas": "The Himalayas contain the world's highest peaks, including Mount Everest.",
    "Andes": "The Andes are the longest continental mountain range and run along western South America.",
    "Antarctica": "Antarctica is the fifth-largest continent and is larger than Greenland.",
    "Greenland": "Greenland is the world's largest island and is covered mostly by an ice sheet.",
    "Bitcoin": "Bitcoin's first block was mined in 2009.",
    "Ethereum": "Ethereum's network launched in 2015, after Bitcoin.",
    "Stonehenge": "Stonehenge is a prehistoric monument in England whose construction began thousands of years ago.",
    "Pyramids of Giza": "The Pyramids of Giza were built in ancient Egypt during the Fourth Dynasty.",
    "Boeing": "Boeing was founded in 1916 as the Pacific Aero Products Company.",
    "Airbus": "Airbus was formed in 1970 as a European aerospace consortium.",
}


def context_for(title: str) -> str:
    """Return the pinned text for a labelled document title."""

    try:
        return DOCUMENTS[title]
    except KeyError as exc:  # fail loudly if the fixture and corpus diverge
        raise KeyError(f"No corpus document is pinned for title {title!r}") from exc
