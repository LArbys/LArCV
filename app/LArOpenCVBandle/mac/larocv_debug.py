#!/usr/bin/env python
import ROOT, sys
from ROOT import std
from larcv import larcv

def create_process(cfg,input_data):
      
   proc = larcv.ProcessDriver('ProcessDriver')

   proc.configure(cfg)

   flist = ROOT.std.vector('std::string')()
   if type(input_data) == type(str()):
      flist.push_back(input_data)
   else:
      for f in input_data: flist.push_back(f)

   proc.override_input_file(flist)

   proc.initialize()

   return proc

def destroy_process(proc):

   proc.finalize()
   del proc
