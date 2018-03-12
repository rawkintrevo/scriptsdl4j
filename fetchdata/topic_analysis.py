
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation


NO_TOPICS= 60
# for modeling an entire script / act
def getTextOnly(script):
    return [line.split(":")[1] if ":" in line else line for line in script if "[" not in line]

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topics

def extractTopics(sceneText, use="nmf", no_features = 1000, no_topics = 6, no_top_words = 15, min_df=2, max_df=0.95):
    # Blog on Topic Modeling with Sk-learn
    # https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation

    if use=="nmf":
        # NMF is able to use tf-idf
        tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(sceneText)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        # Run NMF
        nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
        results = nmf.transform(tfidf)
        return nmf, tfidf_feature_names, no_top_words, results

    if use=="lda":
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(sceneText)
        tf_feature_names = tf_vectorizer.get_feature_names()
        # Run LDA
        lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
        results = lda.transform(tf)
        return lda, tf_feature_names, no_top_words, results


scene_dir = "/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scenes/star-trek-tng"
documents = []
fnames = []
for scene_fname in os.listdir(scene_dir):
    with open(scene_dir + "/" + scene_fname, 'r') as fin:
        doc = fin.readlines()
    doc = getTextOnly(doc[1:])
    fnames.append(scene_fname)
    documents.append("".join(doc))

clf, feature_names, no_top_words, results =  extractTopics(documents, no_topics=NO_TOPICS)
topics = display_topics(clf, feature_names, no_top_words)

# Pick an arbitrary scene and see topics assosciated with the scene, including topic words

# i = 2024
# top_n_topics = 4
# import numpy as np
# print(documents[i])
# for t in list(np.argpartition(results[i], -top_n_topics)[-top_n_topics:]):
#     print(results[i][t], ":", topics[t] )

# Plot counts of each topic- start high then reduce until you don't have any 0s.
# NOTE: 60 was a quick hip shot.

# import matplotlib.pylab as plt
# import numpy
#
# no_topics = 60
# clf, feature_names, no_top_words, results =  extractTopics(documents, no_topics= no_topics)
# counts = {i : 0 for i in range(0, no_topics)}
# for i in range(0, len(results)):
#     for j in np.where(results[i] > 0)[0].tolist():
#         counts[j] += 1
#
# plt.bar(*zip(*sorted(counts.items())))

for i in range(0, len(fnames)):
    with open(scene_dir + "/" + fnames[i], 'r') as fin:
        lines = fin.read().split("\n")
    lines.insert(1, " ".join(["%.6f" % v for v in results[1].tolist()]))
    with open(scene_dir + "/" + fnames[i], 'w') as fout:
        fout.write("\n".join(lines))


with open("/home/rawkintrevo/gits/scriptsdl4j/fetchdata/scripts/topics.txt", "w") as fout:
    for i in range(0,len(topics)):
        fout.write("%i : %s \n" % (i, topics[i]))