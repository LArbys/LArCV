import os,sys
import numpy as np

FILE=sys.argv[1]

data = None
with open(FILE,'r') as f:
    data = f.read()
data = data.split("\n")[:-1]

from multiprocessing.dummy import Pool as ThreadPool

def work(d):
    SS=str(d)
    print 
    os.system(SS)

# Make the Pool of workers
pool = ThreadPool(12)

results = pool.map(work, data)

pool.close()
pool.join()
