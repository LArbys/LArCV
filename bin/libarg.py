
import sys,os,commands

dirs=[x.replace('-lLArCV','') for x in commands.getoutput('larcvapp-config --libs').split() if x.startswith('-lLArCV')]
libs=[x for x in commands.getoutput('larcvapp-config --libs').split() if not x.startswith('-lLArCV')]
libs+= commands.getoutput('root-config --libs').split()
objs_list=[]
dict_list=[]
for l in dirs:
    d='%s/%s' % (os.environ['LARCV_BUILDDIR'],l)
    #print d, os.path.isdir(d)
    objs_list.append(['%s/%s' % (d,x) for x in os.listdir(d) if x.endswith('.o') and not x.endswith('Dict.o')])
    dict_list.append(['%s/%s' % (d,x) for x in os.listdir(d) if x.endswith('Dict.o')])
    #print objs_list[-1]
    #print dict_list[-1]

cmd='-o %s/liblarcv.so ' % os.environ['LARCV_LIBDIR']
for objs in objs_list:

    for obj in objs: cmd += '%s ' % obj

for d in dict_list:

    if d: cmd += '%s ' % d[0]

for l in libs:
    cmd += '%s ' % l

print cmd
