#!/usr/bin/env python


import argparse

parser = argparse.ArgumentParser(description='LArCV Supera Run Script.')

parser.add_argument('-o','--output', 
                    type=str, dest='outfile',required=True,
                    help='string, output file name')

parser.add_argument('-c','--config',  
                    type=str, dest='cfg',required=True
                    help='Config file')

parser.add_argument('-i','--inlist', 
                    type=str, dest='infiles',nargs='+',required=True
                    help='string, expect a text file w/ space or line separated list of input files, override config file')

parser.add_argument('-s','--start',  
                    type=int, dest='start',default=0,
                    help='integer, starting index of storage_manager process, default 0')

parser.add_argument('-n','--numevents',  
                    type=int, dest='nevents',default=0,
                    help='integer, # events to process, default 0 meaning all events')


args = parser.parse_args()

import sys,os

import ROOT
from larlite import larlite as fmwk
fmwk.storage_manager
from larcv import larcv

# Create ana_processor instance
my_proc = fmwk.ana_processor()

# Check if output file
if os.path.exists(args.outfile):
    print "Output file exists. Please remove first."
    print "Specified output file: ",args.outfile
    sys.exit(-1)

   
files = args.infiles
   
if len(args.infiles) == 1:      
    file_=args.infiles[0]

    if os.path.exists(file_):
        with open(file_,'r') as f: 
            files=f.read().split("\n")
              
        if files[-1]=='': files[:-1]
         
for file_ in files:
    my_proc.add_input_file(file_)

# Specify IO mode
my_proc.set_io_mode(fmwk.storage_manager.kREAD)

# Specify output root file name
my_proc.set_ana_output_file("")

# Attach an analysis unit ... here we use a base class which does nothing.
# Replace with your analysis unit if you wish.
unit = fmwk.Supera()

unit.set_config(args.cfg)
unit.supera_fname(args.outfile)

my_proc.add_process(unit)


print
print  "Finished configuring ana_processor. Start event loop!"
print

# Let's run it.
my_proc.run(args.start,args.nevents)

# done!
print
print "Finished running ana_processor event loop!"
print

sys.exit(0)
