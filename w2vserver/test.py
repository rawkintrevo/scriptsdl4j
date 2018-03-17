from requests import get
import numpy as np

vecArray = np.random.rand(300)
vec = list(vecArray)
params = {"a%i" % i : vec[i] for i in range(0, len(vec))}
r = get("http://localhost:9001/vec2word", params= params)
print(r.json()['word'])

vec = get("http://localhost:9001/word2vec/Starfleet").json()["vec"]
params = {"a%i" % i : "%.10f" % vec[i] for i in range(0, len(vec))}
r = get("http://localhost:9001/vec2word", params= params)
print(r.json()['word'])


