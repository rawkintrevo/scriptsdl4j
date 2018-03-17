from flask import Flask, jsonify, request
import gensim
import numpy as np

app = Flask(__name__)


@app.route('/vec2word')
def vec2word():
    args = request.args.to_dict()
    arrayList = []
    for i in range(0,300):
        arrayList.append(float(args['a%i' % i]))
    vec = np.array(arrayList)
    word = model.most_similar(positive=[vec], topn=1)[0][0]
    return jsonify({"word" : word})

@app.route('/word2vec/<word>')
def word2vec(word):
    if word in model:
        return jsonify({"vec" : model[word].tolist()})
    else:
        return False


if __name__ == '__main__':
    print("loading model...")
    model = gensim.models.KeyedVectors.load_word2vec_format('/home/rawkintrevo/gits/scriptsdl4j/data/GoogleNews-vectors-negative300.bin.gz',
                                                            binary=True)
    print("model loaded")
    app.run(port = 9001)


