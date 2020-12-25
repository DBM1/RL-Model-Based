import os

ns = [100, 300, 500]
ms = [100, 500, 1000]
hs = [1, 2, 3]
sps = [0, 5, 10, 20]

for n in ns:
    for m in ms:
        for h in hs:
            for sp in sps:
                os.popen('python main.py --model="nn" --n={} --m={} --h={} --sp={}'.format(n, m, h, sp))
