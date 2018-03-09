
import os
from common_fns import getCharacters

MIN_SCENE_LEN = 1024

# for modeling an entire script / act
def getTextOnly(script):
    return [line.split(":")[1] if ":" in line else line for line in script if "[" not in line]

## For individual scene topic modeling
def getTextBySpeakerLine(script):
    characters = getCharacters(script)
    out = "".join(scene).split(":")
    for c in characters:
        out = [l.replace(c, "") for l in out]
    return out

def extractTopics(sceneText, use="nmf", no_features = 1000, no_topics = 6, no_top_words = 15, min_df=2, max_df=0.95):
    # Blog on Topic Modeling with Sk-learn
    # https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation

    def display_topics(model, feature_names, no_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        return topics

    if use=="nmf":
        # NMF is able to use tf-idf
        tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(sceneText)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        # Run NMF
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        return display_topics(nmf, tfidf_feature_names, no_top_words)

    if use=="lda":
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(sceneText)
        tf_feature_names = tf_vectorizer.get_feature_names()
        # Run LDA
        lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        return display_topics(lda, tf_feature_names, no_top_words)


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
    for s_i in range(0, len(scenes)):
        scene = "".join(["".join(s) for s in scenes[s_i - grab_last_n_scenes : s_i+1]])
        if len(scene) < MIN_SCENE_LEN:
            grab_last_n_scenes += 1
            continue
        sceneText = getTextBySpeakerLine(scene)
        scene.split("\n")
        characters = " ".join(getCharacters(scene.split("\n")))
        # topics = ""
        # try:
        #     topics = " ".join(list(set(" ".join(extractTopics(sceneText, use="lda", no_features=len(" ".join(sceneText))/2, no_topics= 3, no_top_words=3, max_df=1, min_df=1)).split(" "))))
        # except:
        #     print("unable to create topics in episode %s scene %i.  Text:\n%s" % (files[i_f], s_i, "".join(scene)))
        with open(scene_dir + "/" + files[i_f].replace(".txt", "") + "-scene-%i.txt" % s_i , "wb") as fout:
            # fout.write(characters + "\n" + topics + "\n\n" + "".join(scene) )
            fout.write(characters + "\n\n" + "".join(scene) )

