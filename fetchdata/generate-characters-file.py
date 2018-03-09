
from common_fns import getCharacters

with open("/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/tng-all-episodes.txt", 'r') as f:
    data = f.read()

characters = getCharacters(data.split("\n"))

with open("/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/tng-all-chars.txt", 'w') as f:
    f.write("\n".join(characters))

