import commands,os

from random import random, seed

seed(str(1))

keys=('extbnb','extunb','bnb')

cmd_temp_m = { 'extbnb' : 'data_tier=raw and ub_project.stage=mergeext_bnb and ub_project.version=prod_v04_26_04_05',
               'extunb' : 'data_tier=raw and ub_project.stage=mergeext_unbiased and ub_project.version=prod_v04_26_04_05',
               'bnb'    : 'data_tier=raw and ub_project.stage=mergebnb and ub_project.version=prod_v04_26_04_05' }

factor_m = { 'extbnb' : 2.4,
             'extunb' : 1.0,
             'bnb'    : 5.3 }

min_run = 5298
max_run = 5779
numsection=5

nruns = [int((max_run-min_run)/5.+0.5)] * (numsection-1)
nruns.append(max_run - min_run - sum(nruns))

run_range_v=[]

prev_run = 5298
for x in nruns:
    run_range_v.append((prev_run,prev_run+x-1))
    prev_run += x

run_cond = {}
for key in cmd_temp_m.keys(): run_cond[key]=[]

for run_range in run_range_v:

    (low,high) = run_range
    nruns = high - low + 1

    for key in run_cond.keys():

        factor = factor_m[key]
        needed = nruns / factor
        chosen = []
        while float(len(chosen)) < needed:
            target = low + int(random() * nruns)
            if not target in chosen:
                chosen.append(target)

        run_cond[key].append(chosen)

cmd_m = {}
for key in cmd_temp_m.keys(): cmd_m[key]=[]
        
for key in run_cond.keys():
    print
    print key, '# runs chosen:'
    for l in run_cond[key]:
        print len(l),
    print

    print '# files match:'
    cmd_temp = cmd_temp_m[key]
    if key == 'extunb':
        for run_range in run_range_v:
            cmd = 'run_number >= %d and run_number < %d and %s' % (run_range[0],run_range[1],cmd_temp)
            cmd_m[key].append(cmd)
    else:
        for l in run_cond[key]:
            cmd = 'run_number='
            for r in l:
                cmd += '%d,' % r
            cmd = cmd.rstrip(',')
            cmd += ' and %s' % cmd_temp
            cmd_m[key].append(cmd)

for key in cmd_m.keys():
    print
    cmd_list = cmd_m[key]
    print key, '# files chosen:'
    for cmd in cmd_list:
        print commands.getoutput('samweb list-files "%s" | wc -l' % cmd),
    print
    

# create definition
for key in cmd_m.keys():
    print
    cmd_list = cmd_m[key]
    for x in xrange(len(cmd_list)):
        cmd = cmd_list[x]
        print commands.getoutput('samweb -e uboone create-definition wireop_supera_larlite_%s_v2_part%02d_input "%s"' % (key,x,cmd))
