import os,sys
import ROOT as rt
import numpy as np

mc_scale = [1.0,1.0,1.0]
global_scales = [1.235,0.703,0.95] # (data/mc)
mc_scale = global_scales # if want to plot aligned
sigmas = [40,30,20]
nentries = 500000

#fdata = rt.TFile("adc_scales_data_cosmics.root")
fdata = rt.TFile("test_out_data_analysis.root")
fmc   = rt.TFile("adc_scales_mc_cosmics.root")

cch = rt.TCanvas("cch","cch", 1200,1200)
cch.Divide(1,3)

graphs = []
planenames = {0:"U",
              1:"V",
              2:"Y"}
chscalings = {}
for p in range(0,3):
    cch.cd(p+1).SetGridx(1)
    cch.cd(p+1).SetGridy(1)
    gdata = fdata.Get("gmean_plane%d"%(p))
    gmc   = fmc.Get("gmean_plane%d"%(p))
    gmc.SetLineColor(rt.kRed)
    gdata.Draw("AL")
    gmc.Draw("L")
    gdata.SetTitle("Plane %d (%s);wire pixel number;peak ADC centroid"%(p,planenames[p]))
    for ch in range(0,900):
        mc = gmc.GetY()[ch]
        data = gdata.GetY()[ch]
        if mc==0 or data==0:
            chscalings[(p,ch)] = 1.0/global_scales[p] # mc/data
        else:
            #if (p==1 and data<100) or (p in [0,2] and data<150):
            #    global_scales[p]
            #chscalings[(p,ch)] = data/mc
            chscalings[(p,ch)] = (mc*mc_scale[p])/data

cch.Draw()
cch.Update()

fsrcdata = rt.TFile("ana_data.root")
fsrcmc   = rt.TFile("mc.root")

srcdata = fsrcdata.Get("adc")
srcmc   = fsrcmc.Get("adc")

cplane = rt.TCanvas("cplane","cplane",800,1200)
cplane.Divide(1,3)
hists = {}

for p in range(0,3):
    cplane.cd(p+1)
    hdata = rt.TH1D("hadc_data_p%d"%(p),"Plane %d;peak pixel ADC value"%(p),500,0,500)
    hmc   = rt.TH1D("hadc_mc_p%d"%(p),"",500,0,500)
    srcmc.Draw("peak*%.3f>>hadc_mc_p%d"%(mc_scale[p],p),"planeid==%d"%(p))
    srcdata.Draw("peak>>hadc_data_p%d"%(p),"planeid==%d"%(p),"same")

    hmc.SetLineColor(rt.kRed)
    m = hmc.Integral()
    d = hdata.Integral()
    hmc.Scale(1.0/m)
    hdata.Scale(1.0/d)
    cplane.Update()
    hists[("mc",p)] = hmc
    hists[("data",p)] = hdata
cplane.Update()

## ========================================================================

print "begin ch by ch correction"
raw_input()
hmc_correct = {}
for p in range(0,3):
    hmc_correct[p] = rt.TH1D("hadc_mccorrect_p%d"%(p),"Plane %d;peak pixel ADC value"%(p),500,0,500)


for entry in range(nentries):
    srcdata.GetEntry(entry)
    hmc_correct[srcdata.planeid].Fill( chscalings[(srcdata.planeid,srcdata.wireid)]*srcdata.peak )
    if entry%25000==0:
        print "correction entry: ",entry
    
for p in range(0,3):
    cplane.cd(p+1)
    m = hmc_correct[p].Integral()
    hmc_correct[p].Scale( 1.0/m )
    hmc_correct[p].SetLineColor(rt.kCyan)

    hdata = hists[("data",p)]
    hmc = hists[("mc",p)]
    hdata.GetXaxis().SetRange(70,400)
    hmc_correct[p].GetXaxis().SetRange(70,400)
    hmc.GetXaxis().SetRange(70,400)
    dmax = hdata.GetMaximum()
    mmax = hmc_correct[p].GetMaximum()
    mcmax = hmc.GetMaximum()
        
    hmc_correct[p].Scale( dmax/mmax )
    hmc.Scale( dmax/mcmax )

    hmc_correct[p].Draw("same")

## ========================================================================

print "to begin smearing"
raw_input()

data_nentries = srcmc.GetEntries()
if data_nentries<nentries:
    nentries = data_nentries

print "MC entries: ",nentries
hmc_smeared = {}
for p in range(0,3):
    hmc_smeared[p] = rt.TH1D("hadc_mcsmeared_p%d"%(p),"Plane %d;peak pixel ADC value"%(p),500,0,500)


for entry in range(nentries):
    srcmc.GetEntry(entry)
    if srcmc.wireid<250 or srcmc.wireid>300:
        continue
    hmc_smeared[srcmc.planeid].Fill( np.random.normal( mc_scale[srcmc.planeid]*srcmc.peak, sigmas[srcmc.planeid] ) )
    if entry%25000==0:
        print "Smearing entry: ",entry
    
for p in range(0,3):
    cplane.cd(p+1)
    m = hmc_smeared[p].Integral()
    hmc_smeared[p].Scale( 1.0/m )
    hmc_smeared[p].SetLineColor(rt.kCyan)

    hdata = hists[("data",p)]
    hmc = hists[("mc",p)]
    hdata.GetXaxis().SetRange(70,400)
    hmc_smeared[p].GetXaxis().SetRange(70,400)
    hmc.GetXaxis().SetRange(70,400)
    dmax = hdata.GetMaximum()
    mmax = hmc_smeared[p].GetMaximum()
    mcmax = hmc.GetMaximum()
    
    
    hmc_smeared[p].Scale( dmax/mmax )
    hmc.Scale( dmax/mcmax )

    hmc_smeared[p].Draw("same")

cplane.Update()
    

raw_input()


