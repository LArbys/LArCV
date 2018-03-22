import os,sys

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

def main(argv):
    
    OUTDIR = str(argv[1])
    num = int(os.path.basename(argv[2]).split(".")[0].split("_")[-1])

    SS = "hadd -f -k %s " % os.path.join(OUTDIR,"dllee_vertex_%d.root " % num)
    for file_ in argv[2:]:
        SS += file_
        SS += " "

    print SS
    os.system(SS)

if __name__ == '__main__':

    if len(sys.argv) <= 2:
        print
        print "OUTDIR = str(sys.argv[1])"
        print "FLIST  = str(sys.argv[2:])"
        print
        sys.exit(1)
    
    main(sys.argv)
    
    sys.exit(0)
