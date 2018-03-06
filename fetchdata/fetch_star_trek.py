## Python 2.7
from HTMLParser import HTMLParser
from requests import get



class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def processWebPage(html):
    title = strip_tags(html.split('tbody')[0]).split( "-")[1].split('\n')[0].strip()
    body = "".join([c for c in strip_tags(html.split('tbody')[1]).replace("\r", '').replace("\ufffd", '') if ord(c) < 128])
    output = title + "\n" + body
    return output

base_url = "http://www.chakoteya.net/NextGen/"
r = get(base_url+ "episodes.htm")
pages = [line.split('"')[0] for line in r.text.split('href="') if line.split('"')[0][0].isdigit()]

for p in pages:
    print(p)
    r = get(base_url + p)
    out = processWebPage(r.text)
    with open("scripts/star-trek-tng/" + p.split(".")[0] + ".txt", 'wb') as f:
        f.write(out)



html = r.text

