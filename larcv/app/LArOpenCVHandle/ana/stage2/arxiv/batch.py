import os,sys

from multiprocessing.dummy import Pool as ThreadPool

def work(d):
    SS = "python optimum.py {}".format(d)
    os.system(SS)

# Make the Pool of workers
pool = ThreadPool(12)

from random import shuffle
data = [i for i in xrange(1,25)]
shuffle(data)
results = pool.map(work, data)

pool.close()
pool.join()
