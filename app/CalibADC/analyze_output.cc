#include <iostream>
#include <vector>
#include <cstdlib>
#include "TFile.h"
#include "TTree.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH1D.h"

int main( int nargs, char** argv ) {

  std::string inputfile = "ana_data.root";
  int maxchs   = 900;
  float lowerbounds[3] = { 100, 70, 100};
  float upperbounds[3] = {300,250,300};

  TFile* fin  = new TFile(inputfile.c_str());
  TTree* tree = (TTree*)fin->Get("adc");
  int plane,wireid;
  float peak;
  tree->SetBranchAddress("planeid",&plane);
  tree->SetBranchAddress("wireid", &wireid);
  tree->SetBranchAddress("peak",   &peak);

  TFile* fout = new TFile("test_out_data_analysis.root","recreate");

  
  TH1D** hists[3] = {NULL};
  for (int p=0; p<3; p++) {
    hists[p] = new TH1D*[maxchs];
    for ( int ch=0; ch<maxchs; ch++ ) {
      char histname[500];
      sprintf( histname, "hadc_p%d_ch%d", p, ch );
      char histtitle[500];
      sprintf( histtitle, "Plane %d, Channel %d", p, ch );
      hists[p][ch] = new TH1D(histname, histtitle, 50, 0, 500);
    }
  }

  size_t entry = 0;
  size_t numbytes = tree->GetEntry(entry);

  while ( numbytes>0 ) {
    if ( entry%100000==0 )
      std::cout << "Entry " << entry << " p=" << plane << " ch=" << wireid << " peak=" << peak << std::endl;
    hists[plane][wireid]->Fill( peak );
    entry++;
    numbytes = tree->GetEntry(entry);
  }

  for (int p=0; p<3; p++) {
    for ( int ch=0; ch<maxchs; ch++ ) {
      hists[p][ch]->Write();
    }
  }

  int nbadfits = 0;
  for ( int p=0;p<3;p++ ) {

    TGraph* gmean  = new TGraph( maxchs );
    TGraph* gsigma = new TGraph( maxchs );

    for ( int ch=0; ch<maxchs; ch++ ) {
      float integral = hists[p][ch]->Integral();
      float mean = 0;
      float rms  = 0;
      if ( integral>500 ) {
	mean = hists[p][ch]->GetMean();
	rms  = hists[p][ch]->GetRMS();
	char fitname[500];
	sprintf( fitname, "fit_p%d_ch%d", p, ch );
	TF1* fit    = new TF1(fitname,"gaus");
	fit->SetParameter(0, hists[p][ch]->GetMaximum() );
	fit->SetParameter(1, mean );
	fit->SetParameter(2, rms );
	hists[p][ch]->Fit( fit, "RQ", "", lowerbounds[p], upperbounds[p] );
	mean = fit->GetParameter(1);
	rms  = fit->GetParameter(2);
	if ( mean<0 ) {
	  std::cout << "Bad fit plane=" << p << " ch=" << ch << ": " << mean << std::endl;
	  hists[p][ch]->GetXaxis()->SetRange( lowerbounds[p], upperbounds[p] );
	  mean = hists[p][ch]->GetMean();
	  rms  = hists[p][ch]->GetRMS();
	  fit->SetParameter(0, hists[p][ch]->GetMaximum() );
	  fit->SetParameter(1, mean );
	  fit->SetParameter(2, rms );
	  hists[p][ch]->GetXaxis()->SetRange( 1, 500 );
	}
	else {
	  std::cout << "Fit plane=" << p << " ch=" << ch << " mean=" << mean << std::endl;
	  // really on arithmetic mean instead of poor fits
	  // if ( (p!=1 && mean<150) || (p==1 && mean<80) ) {
	  //   if ( p!=1 )
	  //     hists[p][ch]->GetXaxis()->SetRange( hists[p][ch]->GetXaxis()->FindBin(50), hists[p][ch]->GetXaxis()->FindBin(150) );
	  //   else
	  //     hists[p][ch]->GetXaxis()->SetRange( hists[p][ch]->GetXaxis()->FindBin(10), hists[p][ch]->GetXaxis()->FindBin(80) );
	  //   mean = hists[p][ch]->GetMean();
	  // }
	  // else {
	  //   if ( p!=1 ) {
	  //     hists[p][ch]->GetXaxis()->SetRange(hists[p][ch]->GetXaxis()->FindBin(100), hists[p][ch]->GetXaxis()->FindBin(300) );
	  //     mean = hists[p][ch]->GetMean();
	  //   }
	  //   else {
	  //     hists[p][ch]->GetXaxis()->SetRange(hists[p][ch]->GetXaxis()->FindBin(50), hists[p][ch]->GetXaxis()->FindBin(250) );
	  //     mean = hists[p][ch]->GetMean();
	  //   }
	  // }
	}
	fit->Write(fitname);
	delete fit;
      }
      gmean->SetPoint(ch,ch,mean);
      gsigma->SetPoint(ch,ch,rms);
    }
    char meanname[100];
    sprintf( meanname, "gmean_plane%d", p );
    char sigmaname[100];
    sprintf( sigmaname, "gsigma_plane%d", p );
    gmean->Write( meanname );
    gsigma->Write( sigmaname );
  }
  
  std::cout << "Fin." << std::endl;
}
