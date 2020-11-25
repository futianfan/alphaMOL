import sys
from props import *

for line in sys.stdin:
    x,y = line.split()
    if y == "None": y = None
    sim2D = similarity(x, y)
    try:
        #print x, y, sim2D, qed(y), qed(x), drd2(y), drd2(x)
        print x, y, sim2D, qed(y)
    except Exception as e:
        print x, y, sim2D, 0.0