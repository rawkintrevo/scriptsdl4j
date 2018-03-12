
import os
from common_fns import getCharacters

MIN_SCENE_LEN = 1024


## For individual scene topic modeling
def getTextBySpeakerLine(script):
    characters = getCharacters(script)
    out = "".join(scene).split(":")
    for c in characters:
        out = [l.replace(c, "") for l in out]
    return out




episode_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/star-trek-tng"
scene_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scenes/star-trek-tng"

files = os.listdir(episode_dir)
for i_f in range(0, len(files)):
    with open(episode_dir + "/" + files[i_f], 'rb') as fin:
        lines = fin.readlines()

    episode_title = lines[0]

    scenes_idx = []
    for i in range(0, len(lines)):
        if lines[i].strip().startswith("["):
            scenes_idx.append(i)

    scenes = []
    for idx in range(0, len(scenes_idx)-1):
        scenes.append(lines[scenes_idx[idx]: scenes_idx[idx+1]])


    # ## for full script
    # sceneText = ["".join(getTextOnly(scene)) for scene in scenes]
    # ## for individual scene
    # sceneText = getTextBySpeakerLine(scene)

    grab_last_n_scenes = 0
    scene_counter = 0
    for s_i in range(0, len(scenes)):
        scene = "".join(["".join(s) for s in scenes[s_i - grab_last_n_scenes : s_i+1]])
        if len(scene) < MIN_SCENE_LEN:
            grab_last_n_scenes += 1
            continue
        scene_counter += 1
        grab_last_n_scenes = 0
        sceneText = getTextBySpeakerLine(scene)
        scene.split("\n")
        characters = " ".join(getCharacters(scene.split("\n")))
        with open(scene_dir + "/" + files[i_f].replace(".txt", "") + "-scene-%i.txt" % scene_counter , "wb") as fout:
            fout.write(characters.strip() + "\n\n" + "".join(scene) )
            # fout.write(characters + "\n\n" + "".join(scene) )



