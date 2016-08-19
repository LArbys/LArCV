from ROOT import TChain
import sys,os

TARGET_DIR=sys.argv[1]

if not TARGET_DIR.startswith('/'):
    TARGET_DIR = os.getcwd() + '/' + TARGET_DIR

PREFIX='larlite'
SUFFIX='.root'

job_files={}
flavors=[]
for f in [x for x in os.listdir(TARGET_DIR) if x.startswith(PREFIX) and x.endswith(SUFFIX)]:

    jobid = int(f.split('_')[-1].replace(SUFFIX,''))
    flavor = f.split('_')[-2]
    if not flavor in flavors:
        flavors.append(flavor)
    if not jobid in job_files:
        job_files[jobid]=[]
    job_files[jobid].append('%s/%s' % (TARGET_DIR,f))

bad_files={}
good_files={}
for jobid,files in job_files.iteritems():

    if not len(files) == len(flavors):
        print 'JobID %d missing files (%d/%d)' % (jobid,len(files),len(flavors))
        bad_files[jobid]=files
        continue

    good=True
    for f in files:
        ch=TChain("larlite_id_tree")
        ch.AddFile(f)
        if ch.GetEntries() > 0: continue
        good=False
        break

    if not good:
        print 'JobID %d has corrupted file(s)!' % jobid
        bad_files[jobid]=files
    else:
        good_files[jobid]=files

print 'Found',len(bad_files),'bad jobs'
print bad_files.keys()
print

#
# Ask if we want to remove them
#
user_input = ''
while len(bad_files) and not user_input:
    sys.stdout.write('Remove bad files?[y/n]: ')
    sys.stdout.flush()
    user_input=sys.stdin.readline().strip('\n')
    if not user_input in ['y','yes','n','no']:
        user_input = ''
if user_input in ['y','yes']:
    try:
        for key,value in bad_files.iteritems():
            for f in value:
                os.remove(f)
    except OSError:
        print 'Failed to remove bad files...'
        pass

#
# Ask if we want to register to production database
#
user_input = ''
while not user_input:
    sys.stdout.write('Register in production DB?[y/n]: ')
    sys.stdout.flush()
    user_input=sys.stdin.readline().strip('\n')
    if not user_input in ['y','yes','n','no']:
        user_input = ''
if user_input in ['n','no']:
    sys.exit(0)

project = ''
while not project:
    sys.stdout.write('Define a project name: ')
    sys.stdout.flush()
    project=sys.stdin.readline().strip('\n')
    
try:
    from proddb.table import table
except ImportError:
    'You do not have proddb. Check $LARCV_BASEDIR/python dir.'
    sys.exit(1)

from proddb.table import table
t=table(project)
if t.exist():
    user_input = ''
    while not user_input:
        sys.stdout.write('Project name already exists. Want to really use it? [y/n]: ')
        sys.stdout.flush()
        user_input=sys.stdin.readline().strip('\n')
        if not user_input in ['y','yes','n','no']:
            user_input = ''
else:
    t.create()
if user_input in ['n','no']:
    sys.exit(0)

keys=good_files.keys()
keys.sort()
for k in keys:
    files=good_files[k]
    t.register_session(files_v=files,status=1,check=False)

#
# Ask if we want to define files for job submission
#
user_input = ''
while not user_input:
    sys.stdout.write('Prepare project for job submission? [y/n]: ')
    sys.stdout.flush()
    user_input=sys.stdin.readline().strip('\n')
    if not user_input in ['y','yes','n','no']:
        user_input = ''

if user_input in ['n','no']: 
    sys.exit(0)

t.unlock()
t.reset_job_index(status=1)
