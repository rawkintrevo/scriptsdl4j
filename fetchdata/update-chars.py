import os
from common_fns import getCharacters

episode_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/star-trek-tng"
scene_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scenes/star-trek-tng"


all_char_file = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/tng-all-chars.txt"
with open(all_char_file, 'r') as f:
    valid_chars = f.read().split("\n")

episode_files = os.listdir(episode_dir)
scene_files = os.listdir(scene_dir)

max_guests = 0
for i_f in range(0, len(episode_files)):
    with open(episode_dir + "/" + episode_files[i_f], 'rb') as fin:
        lines = fin.read()
    guest_chars = [c for c in getCharacters(lines.split("\n")) if c not in valid_chars if len(c) > 2]
    guest_chars.sort(key=len, reverse=True)
    if "" in guest_chars:
        guest_chars.remove("")
    char_map = {guest_chars[i] : "GUEST%i" % i for i in range(0, len(guest_chars))}
    scenes_in_this_episode = [f for f in scene_files if episode_files[i_f].split(".")[0] in f]
    max_guests = max(max_guests, len(guest_chars))
    for scene_fname in scenes_in_this_episode:
        with open(scene_dir + "/" + scene_fname, 'r') as f_scene:
            data = f_scene.read()
        for k, v in char_map.iteritems():
            data = data.replace(k, v)
        with open(scene_dir + "/" + scene_fname, 'w') as f_scene:
            f_scene.write(data)

print("Max guests in any episode: %i" % max_guests)
print("adding to tng-all-chars.txt")

with open(all_char_file, 'w') as fout:
    all_chars = valid_chars + ["GUEST%i" % i for i in range(0, max_guests)]
    fout.write("\n".join(all_chars) )

################

for sf in scene_files:
    with open(scene_dir + "/" + sf, "r") as fin:
        lines = fin.readlines()
        if " T\n" in lines:
            print(sf)
            break


