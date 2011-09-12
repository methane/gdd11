import requests
import re
import json

def solve(url, cookie):
    setup = re.compile(r"setup\((.*?)\)")

    r = requests.get(url, cookies=cookie)

    while True:
        d = r.read()
        m = setup.search(d)
        if not m:
            return r
        colors = json.loads(m.group(1))

        print "colors:", colors

        answer = []
        for i in xrange(len(colors)):
            c = colors[i]
            for j in xrange(len(colors)):
                if i == j: continue
                if colors[j] == c:
                    answer.append(j)
                    break

        answer = ','.join(map(str, answer))
        print "answer:", answer

        r = requests.post(url, data=dict(answer=answer), cookies=cookie)
