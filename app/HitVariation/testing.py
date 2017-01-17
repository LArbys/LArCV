#testing testing 123

import ROOT                                                                                                                                                     
from ROOT import std                                                                                                                                            
from larcv import larcv                                                                                                                                         

file_path = "/Users/mattlindsay/LArCV/app/HitVariation/varyhits.cfg"        
hitvaryalgo = larcv.HitVariation( file_path )                                                                                                             
labels = std.vector("int")() # this is how one makes stdlib vectors in PyROOT                                                                                   
                                                                                                                                                              
                                                                                                                                                
event_img_v = hitvaryalgo.GenerateImages( batchsize, pars, labels, )                                                                                          
for ibatch in batchsize:                                                                                                                                      
    img_v = event_img_v.at(ibatch)                                                                                                                              
    for p in nplanes:                                                                                                                                           
        img = larcv.as_ndarray( img_v.at(p))   

print img.shape
