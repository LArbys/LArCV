#ifndef __READNUEFILE_CXX__
#define __READNUEFILE_CXX__

#include "ReadNueFile.h"

// larcv
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

// larlite
#include "DataFormat/track.h"
#include "DataFormat/hit.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"

#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/TimeService.h"

// ROOT
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TLine.h"
#include "TF1.h"
#include "TVector3.h"
#include "TRandom3.h"
#include "TSpline.h"

// AStar
#include "AStarUtils.h"
#include "AStar3DAlgoProton.h"

// test
#include <cassert>

namespace larcv {

  static ReadNueFileProcessFactory __global_ReadNueFileProcessFactory__;

  ReadNueFile::ReadNueFile(const std::string name)
    : ProcessBase(name), _spline_file("")
  {}

  void ReadNueFile::configure(const PSet& cfg)
  {}

  void ReadNueFile::initialize()
  {
    LARCV_INFO() << "[ReadNueFile]" << std::endl;
    assert(!_spline_file.empty());
    tracker.SetSplineFile(_spline_file);
    tracker.initialize();
    tracker.SetCompressionFactors(1,6);
    tracker.SetVerbose(0);
    iTrack = 0;
    NvertexSubmitted = 0;

    hEcomp             = new TH2D("hEcomp","hEcomp;E_th;E_reco",100,0,1000,100,0,1000);
    hEcomp1D           = new TH1D("hEcomp1D","hEcomp1D; (E_{reco}-E_{truth})/E_{truth}",500,-2,8);
    hEcomp1D_m         = new TH1D("hEcomp1D_m","hEcomp1D_m; (E_{reco}-E_{truth})/E_{truth}",400,-2,8);
    hEcomp1D_p         = new TH1D("hEcomp1D_p","hEcomp1D_p; (E_{reco}-E_{truth})/E_{truth}",400,-2,8);
    hAverageIonization = new TH1D("hAverageIonization","hAverageIonization",100,0,10000);
    hEnuReco           = new TH1D("hEnuReco","hEnuReco; E_{#nu, reco} (MeV)",200,0,2000);
    hEnuTh             = new TH1D("hEnuTh","hEnuTh;E_{#nu, th} (MeV)",200,0,2000);
    hEcompdQdx         = new TH2D("hEcompdQdx","hEcompdQdx;AverageADCperPix/L;(E_{reco}-E_{truth})/E_{truth}",100,0,1000,100,-2,8);
    hIonvsLength       = new TH2D("hIonvsLength","hIonvsLength;L [cm];AverageADCperPix",100,0,1000,100,0,10000);
    hEnuComp           = new TH2D("hEnuComp","hEnuComp;hEnuTh (MeV);hEnuReco (MeV)",200,0,2000,200,0,2000);
    hEnuComp1D         = new TH1D("hEnuComp1D","hEnuComp1D;(E_{reco}-E_{truth})/E_{truth}",500,-2,10);
    hEnuvsPM_th        = new TH2D("hEnuvsPM_th","hEnuvsPM_th;E_{#nu};E_{p}+E_{m}",200,0,2000,200,0,2000);
    hPM_th_Reco_1D     = new TH1D("hPM_th_Reco_1D","hPM_th_Reco_1D;(E_{p+m reco}-E_{p+m th})/E_{p+m th}",200,-2,10);
    hPM_th_Reco        = new TH2D("hPM_th_Reco","hPM_th_Reco;E_{p+m th};E_{p+m reco}",200,0,2000,200,0,2000);
  }

  bool ReadNueFile::process(IOManager& mgr)
  {
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Entry " << mgr.current_entry() << " / " << mgr.get_n_entries() << std::endl;
    std::cout << "============================================" << std::endl;
    gStyle->SetOptStat(0);

    TVector3 vertex, endPoint[2];

    //
    // Loop per vertex (larcv type is PGraph "Particle Graph")
    //

    auto ev_img_v       = (EventImage2D*) mgr.get_data(kProductImage2D,"wire");
    auto ev_pgraph_v    = (EventPGraph*)  mgr.get_data(kProductPGraph,"test_nue");
    if (ev_pgraph_v->PGraphArray().empty()) return true;
    //auto ev_ctor_v      = (EventPixel2D*) mgr.get_data(kProductPixel2D,"test_nue_ctor");
    auto ev_pix_v       = (EventPixel2D*) mgr.get_data(kProductPixel2D,"test_nue_img");
    auto ev_partroi_v   = (EventROI*)     mgr.get_data(kProductROI,"segment");
    // auto tag_img_thru_v = (EventImage2D*) mgr.get_data(kProductImage2D,"thrumutags");
    // auto tag_img_stop_v = (EventImage2D*) mgr.get_data(kProductImage2D,"stopmutags");

    int run    = ev_pgraph_v->run();
    int subrun = ev_pgraph_v->subrun();
    int event  = ev_pgraph_v->event();

    // get the event clusters and full images
    const auto& full_adc_img_v = ev_img_v->Image2DArray();
    const auto& pix_m      = ev_pix_v->Pixel2DClusterArray();
    const auto& pix_meta_m = ev_pix_v->ClusterMetaArray();

    // const auto& full_tag_img_thru_v = tag_img_thru_v->Image2DArray();
    // const auto& full_tag_img_stop_v = tag_img_stop_v->Image2DArray();

    const auto& mc_roi_v = ev_partroi_v->ROIArray();

    // get MC vertex
    std::vector<TVector3> MuonVertices;
    std::vector<TVector3> ProtonVertices;
    std::vector<TVector3> ElectronVertices;
    std::vector<TVector3> MuonEndPoint;
    std::vector<TVector3> ProtonEndPoint;
    std::vector<TVector3> ElectronEndPoint;
    for(size_t iMC = 0;iMC<mc_roi_v.size();iMC++){
      if(mc_roi_v[iMC].PdgCode() == 13){
	std::cout << "muon.....@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	MuonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
	MuonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
      }
      if(mc_roi_v[iMC].PdgCode() == 2212){
	std::cout << "proton...@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	ProtonVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
	ProtonEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
      }
      if(mc_roi_v[iMC].PdgCode() == 11){
	std::cout << "electron.@" << mc_roi_v[iMC].X() << ", " << mc_roi_v[iMC].Y() << ", " << mc_roi_v[iMC].Z() << " ... " << mc_roi_v[iMC].EnergyDeposit() << " MeV" << std::endl;
	ElectronVertices.push_back(TVector3(mc_roi_v[iMC].X(),mc_roi_v[iMC].Y(),mc_roi_v[iMC].Z()));
	ElectronEndPoint.push_back(TVector3(mc_roi_v[iMC].EndPosition().X(), mc_roi_v[iMC].EndPosition().Y(), mc_roi_v[iMC].EndPosition().Z()));
      }
    }
    std::vector<TVector3> MCVertices;
    std::vector<TVector3> MCEndPoint;
    bool isVertex = false;
    bool isNumu = false;
    bool isNue = false;
    std::vector<int> goodMuon, goodElectron;
    for(size_t iProton = 0;iProton<ProtonVertices.size();iProton++){
      isVertex = false;
      isNumu = false;
      isNue = false;
      for(size_t iMuon = 0;iMuon<MuonVertices.size();iMuon++){
	if(MuonVertices[iMuon] == ProtonVertices[iProton]){isVertex = true;isNumu = true;goodMuon.push_back(iMuon);}
      }
      for(size_t iElectron = 0;iElectron<ElectronVertices.size();iElectron++){
	if(ProtonVertices[iProton] == ElectronVertices[iProton]){isVertex = true;isNue = true;goodElectron.push_back(iElectron);}
      }
      if(isVertex && MCVertices.size()!=0 && ProtonVertices[iProton] == MCVertices[MCVertices.size()-1])continue;
      if(isVertex){
	MCVertices.push_back(ProtonVertices[iProton]);
	MCEndPoint.push_back(ProtonEndPoint[iProton]);
	if(isNumu){
	  for(size_t imu = 0;imu<goodMuon.size();imu++){
	    MCEndPoint.push_back(MuonEndPoint[goodMuon[imu]]);
	  }
	}
	if(isNue){
	  for(size_t ie = 0;ie<goodElectron.size();ie++){
	    MCEndPoint.push_back(ElectronEndPoint[goodElectron[ie]]);
	  }
	}
      }
    }

    //______________
    // End MC
    //--------------


    // loop over found vertices
    std::vector<TVector3> vertex_v;
    std::vector<larcv::ImageMeta> Full_meta_v(3);
    std::vector<larcv::Image2D> Full_image_v(3);
    std::vector<larcv::Image2D> Tagged_Image(3);
    double wireRange = 5000;
    double tickRange = 8502;

    // Create base image2D with the full view, fill it with the input image 2D, we will crop it later
    for(size_t iPlane=0;iPlane<3;iPlane++){
      Full_meta_v[iPlane] = larcv::ImageMeta(wireRange,tickRange,(int)(tickRange)/6,(int)(wireRange),0,tickRange);
      Full_image_v[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
      Tagged_Image[iPlane] = larcv::Image2D(Full_meta_v[iPlane]);
      if(full_adc_img_v.size() == 3) Full_image_v[iPlane].overlay( full_adc_img_v[iPlane] );
      //if(full_tag_img_thru_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_thru_v)[iPlane] );
      //if(full_tag_img_stop_v->size() == 3)Tagged_Image[iPlane].overlay( (*full_tag_img_stop_v)[iPlane] );
    }

    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {

      iTrack++;

      auto const& pgraph        = ev_pgraph_v->PGraphArray().at(pgraph_id);
      auto const& roi_v         = pgraph.ParticleArray();
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();

      size_t nparticles = cluster_idx_v.size();

      //
      // Get Estimated 3D Start and End Points
      std::vector<TVector3> EndPoints;
      TVector3 vertex(pgraph.ParticleArray().front().X(),
		      pgraph.ParticleArray().front().Y(),
		      pgraph.ParticleArray().front().Z());
      EndPoints.push_back(vertex);

      bool WrongEndPoint = false;
      for(size_t iPoint = 0;iPoint<EndPoints.size();iPoint++){
	if(!tracker.CheckEndPointsInVolume(EndPoints[iPoint]) ) {
	  std::cout << "=============> ERROR! End point " << iPoint << " outside of volume" << std::endl; 
	  WrongEndPoint = false;
	}
      }
      if(WrongEndPoint) continue;
      vertex_v.push_back(vertex);
      
      //
      // mask shower particle pixels 
      // method : pixels stored via larbys image
      //
      for(size_t roid=0; roid < roi_v.size(); ++roid) {
	const auto& roi = roi_v[roid];
	auto cidx = cluster_idx_v.at(roid);

	if (roi.Shape() == kShapeTrack) continue;
	
	for(size_t plane=0; plane<3; ++plane) {
	  
	  auto iter_pix = pix_m.find(plane);
	  if(iter_pix == pix_m.end())
	    continue;

	  auto iter_pix_meta = pix_meta_m.find(plane);
	  if(iter_pix_meta == pix_meta_m.end())
	    continue;
	  
	  auto const& pix_v      = (*iter_pix).second;
	  auto const& pix_meta_v = (*iter_pix_meta).second;
	  
	  auto const& pix      = pix_v.at(cidx);
	  auto const& pix_meta = pix_meta_v.at(cidx);
	  
	  auto& plane_img = Full_image_v.at(plane);
	  for(const auto& px : pix) {
	    auto posx = pix_meta.pos_x(px.Y());
	    auto posy = pix_meta.pos_y(px.X());
	    auto row = plane_img.meta().row(posy);
	    auto col = plane_img.meta().col(posx);
	    plane_img.set_pixel(row,col,0);
	  }
	} // end plane
      } // end ROI
    } // end vertex

    NvertexSubmitted+=vertex_v.size();
    tracker.SetOriginalImage(Full_image_v);
    tracker.SetTaggedImage(Tagged_Image);
    tracker.SetTrackInfo(run, subrun, event, 0);
    tracker.SetEventVertices(vertex_v);
    tracker.ReconstructEvent();
    std::cout << std::endl << std::endl;
        
    MCevaluation();
    
    return true;
  }


  void ReadNueFile::MCevaluation(){
    std::cout << "MCevaluation()" << std::endl;

    std::vector< std::vector<double> > Energies_v = tracker.GetEnergies();
    std::vector<double> ionPerTrack = tracker.GetAverageIonization();
    std::vector<double> VertexLengths = tracker.GetVertexLength();

    std::cout << "E" << Energies_v.size() << std::endl;
    std::cout << "i" << ionPerTrack.size() << std::endl;
    std::cout << "v" << VertexLengths.size() << std::endl;

    if(ionPerTrack.size()!=2)   return;
    if(!tracker.IsGoodVertex()) return;

    std::cout << "in store energies : " << std::endl;
    std::cout << Energies_v[0][0] << ":" << Energies_v[0][1] << std::endl;
    std::cout << Energies_v[1][0] << ":" << Energies_v[1][1] << std::endl;

    double Epreco;
    double Emreco;

    if(ionPerTrack[0] > ionPerTrack[1]){
      Epreco = Energies_v[0][0];// first track is proton
      Emreco = Energies_v[1][1];// second track is muon
      hEcomp->Fill(Ep_t,Epreco);
      hEcomp->Fill(Em_t,Emreco);
      hEcomp1D->Fill((Epreco-Ep_t)/Ep_t);
      hEcomp1D->Fill((Emreco-Em_t)/Em_t);
      hEcomp1D_p->Fill((Epreco-Ep_t)/Ep_t);
      hEcomp1D_m->Fill((Emreco-Em_t)/Em_t);
      hEcompdQdx->Fill(ionPerTrack[1]/VertexLengths[1],(Epreco-Ep_t)/Ep_t);
      hEcompdQdx->Fill(ionPerTrack[0]/VertexLengths[0],(Emreco-Em_t)/Em_t);
      if( std::abs((Epreco-Ep_t)/Ep_t) > 0.1 || std::abs((Emreco-Em_t)/Em_t) > 0.1 ){
	checkEvents.push_back(Form("%d_%d_%d",run,subrun,event));
      }
    }
    else{
      Epreco = Energies_v[1][0]; // second track is proton
      Emreco = Energies_v[0][1]; // first track is muon
      hEcomp->Fill(Ep_t,Epreco);
      hEcomp->Fill(Em_t,Emreco);
      hEcomp1D->Fill((Epreco-Ep_t)/Ep_t);
      hEcomp1D->Fill((Emreco-Em_t)/Em_t);
      hEcomp1D_p->Fill((Epreco-Ep_t)/Ep_t);
      hEcomp1D_m->Fill((Emreco-Em_t)/Em_t);
      hEcompdQdx->Fill(ionPerTrack[1]/VertexLengths[1],(Epreco-Ep_t)/Ep_t);
      hEcompdQdx->Fill(ionPerTrack[0]/VertexLengths[0],(Emreco-Em_t)/Em_t);
      if( std::abs((Epreco-Ep_t)/Ep_t) > 0.1 || std::abs((Emreco-Em_t)/Em_t) > 0.1 ){
	checkEvents.push_back(Form("%d_%d_%d",run,subrun,event));
      }
    }

    hEnuvsPM_th->Fill(NeutrinoEnergyTh,Ep_t+Em_t);
    hPM_th_Reco_1D->Fill(((Epreco+Emreco)-(Ep_t+Em_t))/(Ep_t+Em_t));
    hPM_th_Reco->Fill(Ep_t+Em_t,Epreco+Emreco);

    std::cout << "proton : Epreco = " << Epreco << " MeV, Epth : " << Ep_t << " MeV..." << 100*(Epreco-Ep_t)/Ep_t << " %" << std::endl;
    std::cout << "muon   : Emreco = " << Emreco << " MeV, Emth : " << Em_t << " MeV..." << 100*(Emreco-Em_t)/Em_t << " %" << std::endl;

    std::cout << "E nu th   : " << NeutrinoEnergyTh << " MeV" << std::endl;
    std::cout << "E nu Reco : " << Epreco+Emreco << " MeV" << std::endl;
    std::cout << "relative Enu diff: " << 100*((Epreco+Emreco)-NeutrinoEnergyTh)/NeutrinoEnergyTh << " %" << std::endl;
    std::cout << "relative Enu_p+m diff: " << 100*((Epreco+Emreco)-(Ep_t+Em_t))/(Ep_t+Em_t) << " %" << std::endl;
    hEnuReco->Fill(Epreco+Emreco);
    hEnuComp->Fill(NeutrinoEnergyTh,Epreco+Emreco);
    hEnuComp1D->Fill(((Epreco+Emreco)-NeutrinoEnergyTh)/NeutrinoEnergyTh);


    for(size_t itrack = 0;itrack<ionPerTrack.size();itrack++){
      hAverageIonization->Fill(ionPerTrack[itrack]);
      hIonvsLength->Fill(VertexLengths[itrack],ionPerTrack[itrack]);
            
    }
  }
    
  void ReadNueFile::finalize()
  {
    std::cout << NvertexSubmitted << " vertex submitted" << std::endl;
    std::cout << "saving root files?...";
    if(hEcomp->GetEntries() > 1){
      std::cout << "... yes" << std::endl;
      hEcomp->SaveAs(Form("root/hEcomp_%d_%d_%d.root",run,subrun,event));
      hEcompdQdx->SaveAs(Form("root/hEcompdQdx_%d_%d_%d.root",run,subrun,event));
      hEcomp1D->SaveAs(Form("root/hEcomp1D_%d_%d_%d.root",run,subrun,event));
      hEcomp1D_m->SaveAs(Form("root/hEcomp1D_m_%d_%d_%d.root",run,subrun,event));
      hEcomp1D_p->SaveAs(Form("root/hEcomp1D_p_%d_%d_%d.root",run,subrun,event));
      hIonvsLength->SaveAs(Form("root/hIonvsLength_%d_%d_%d.root",run,subrun,event));
      hAverageIonization->SaveAs(Form("root/hAverageIonization_%d_%d_%d.root",run,subrun,event));
      hEnuReco->SaveAs(Form("root/hEnuReco_%d_%d_%d.root",run,subrun,event));
      hEnuTh->SaveAs(Form("root/hEnuTh_%d_%d_%d.root",run,subrun,event));
      hEnuComp->SaveAs(Form("root/hEnuComp_%d_%d_%d.root",run,subrun,event));
      hEnuComp1D->SaveAs(Form("root/hEnuComp1D_%d_%d_%d.root",run,subrun,event));
      hEnuvsPM_th->SaveAs(Form("root/hEnuvsPM_th_%d_%d_%d.root",run,subrun,event));
      hPM_th_Reco_1D->SaveAs(Form("root/hPM_th_Reco_1D_%d_%d_%d.root",run,subrun,event));
      hPM_th_Reco->SaveAs(Form("root/hPM_th_Reco_%d_%d_%d.root",run,subrun,event));
    }
    else{std::cout << "... no" << std::endl;}
    tracker.finalize();

    for(auto picture:checkEvents){
      std::cout << picture << std::endl;
    }

  }    
  void ReadNueFile::SetSplineLocation(const std::string& fpath) {
    LARCV_INFO() << "setting spline loc @ " << fpath << std::endl;
    tracker.SetSplineFile(fpath);
    _spline_file = fpath;
    LARCV_DEBUG() << "end" << std::endl;
  }

}
#endif
