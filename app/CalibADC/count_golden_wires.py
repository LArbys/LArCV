import os,sys
import ROOT as rt
import numpy as np

chbych = open("chbych_scaling.txt",'r')

goodranges = {0:[0.8,1.4],
              1:[0.5,1.0],
              2:[0.8,1.2]}

ngood = {0:0,
         1:0,
         2:0}

scalehists = {}
for p in range(0,3):
    scalehists[p] = rt.TH1D("hscale_p%d"%(p),"",100,0,2.0)

lchbych = chbych.readlines()
#['2', '863', '368.49', '181.10', '2.03', '1']

badchlist = open('nongolden_channels.txt','w')
planes = {0:"U",
          1:"V",
          2:"Y"}
maxwires = {0:2400,1:2400,2:3456}
for l in lchbych[1:]:
    info =  l.strip().split()
    scale = float( info[4] )
    badmask = int( info[5] )
    plane = int( info[0] )
    wid   = int( info[1] )

    if badmask==1:
        scalehists[plane].Fill( scale )
    else:
        scalehists[plane].Fill( 0.0 )

    if scale>=goodranges[plane][0] and scale<=goodranges[plane][1]:
        ngood[plane] += 1
    else:
        for wire in range( wid*4, (wid+1)*4 ):
            if wire < maxwires[plane]:
                print >>badchlist,wire,'\t',planes[plane]

badchlist.close()

print "NGOOD:"
print "U-plane: ",ngood[0]
print "V-plane: ",ngood[1]
print "Y-plane: ",ngood[2]


c = rt.TCanvas("c","c",800,600)
c.Draw()

scalehists[0].Draw()
scalehists[1].Draw("same")
scalehists[2].Draw("same")

scalehists[0].SetLineColor(rt.kRed)
scalehists[1].SetLineColor(rt.kGreen)
scalehists[2].SetLineColor(rt.kBlue)

c.Update()

raw_input()
    
