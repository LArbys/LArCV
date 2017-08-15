#ifndef __READRUIFILE_CXX__
#define __READRUIFILE_CXX__

#include "ReadRuiFile.h"

#include "DataFormat/track.h"
#include "DataFormat/hit.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/wire.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/GeometryHelper.h"
#include "LArUtil/TimeService.h"
#include "TFile.h"
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

//#include "LArCV/core/DataFormat/ChStatus.h"
#include "AStarUtils.h"

#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

//#include "SCE/SpaceChargeMicroBooNE.h"
#include "AStarTracker.h"


#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  static ReadRuiFileProcessFactory __global_ReadRuiFileProcessFactory__;

  ReadRuiFile::ReadRuiFile(const std::string name)
    : ProcessBase(name)
  {}
    
  void ReadRuiFile::configure(const PSet& cfg)
  {}

  void ReadRuiFile::initialize()
  {}

  bool ReadRuiFile::process(IOManager& mgr)
  {

    gStyle->SetOptStat(0);
    TFile *fSplines = TFile::Open("Proton_Muon_Range_dEdx_LAr_TSplines.root","READ");
    TSpline3 *sMuonRange2T   = (TSpline3*)fSplines->Get("sMuonRange2T");
    TSpline3 *sMuonT2dEdx    = (TSpline3*)fSplines->Get("sMuonT2dEdx");
    TSpline3 *sProtonRange2T = (TSpline3*)fSplines->Get("sProtonRange2T");
    TSpline3 *sProtonT2dEdx  = (TSpline3*)fSplines->Get("sProtonT2dEdx");


    /*
    TFile *f = new TFile("../data/c_ana_0200_0299.root");
    TTree *T = (TTree*)f->Get("dQdSAnalysis");
    TTree *Tv =(TTree*)f->Get("VertexTree");

    UInt_t run, subrun, event, entry;
    Int_t roid, vtxid;
    Int_t run_v, subrun_v, event_v, entry_v;
    Int_t roid_v, vtxid_v;
    double x,y,z;

    double particle0_end_x;
    double particle0_end_y;
    double particle0_end_z;

    double particle1_end_x;
    double particle1_end_y;
    double particle1_end_z;
    double scedr;

    std::vector< std::pair<int,int> > *particle0_end_point_v = 0;
    std::vector< std::pair<int,int> > *particle1_end_point_v = 0;
    std::vector<TVector3> *image_particle0_plane0_tmp = 0;
    std::vector<TVector3> *image_particle0_plane1_tmp = 0;
    std::vector<TVector3> *image_particle0_plane2_tmp = 0;

    std::vector<TVector3> *image_particle1_plane0_tmp = 0;
    std::vector<TVector3> *image_particle1_plane1_tmp = 0;
    std::vector<TVector3> *image_particle1_plane2_tmp = 0;

    std::vector<larcv::Image2D> *adc_img_v = 0;

    T->SetBranchAddress("run",&run);
    T->SetBranchAddress("subrun",&subrun);
    T->SetBranchAddress("event",&event);
    T->SetBranchAddress("entry",&entry);
    T->SetBranchAddress("roid",&roid);
    T->SetBranchAddress("vtxid",&vtxid);
    T->SetBranchAddress("x",&x);
    T->SetBranchAddress("y",&y);
    T->SetBranchAddress("z",&z);
    T->SetBranchAddress("particle0_end_point_v",&particle0_end_point_v);
    T->SetBranchAddress("particle1_end_point_v",&particle1_end_point_v);

    T->SetBranchAddress("particle0_end_x",&particle0_end_x);
    T->SetBranchAddress("particle0_end_y",&particle0_end_y);
    T->SetBranchAddress("particle0_end_z",&particle0_end_z);
    T->SetBranchAddress("particle1_end_x",&particle1_end_x);
    T->SetBranchAddress("particle1_end_y",&particle1_end_y);
    T->SetBranchAddress("particle1_end_z",&particle1_end_z);

    T->SetBranchAddress("image_particle0_plane0_tmp", &image_particle0_plane0_tmp);
    T->SetBranchAddress("image_particle0_plane1_tmp", &image_particle0_plane1_tmp);
    T->SetBranchAddress("image_particle0_plane2_tmp", &image_particle0_plane2_tmp);
    T->SetBranchAddress("image_particle1_plane0_tmp", &image_particle1_plane0_tmp);
    T->SetBranchAddress("image_particle1_plane1_tmp", &image_particle1_plane1_tmp);
    T->SetBranchAddress("image_particle1_plane2_tmp", &image_particle1_plane2_tmp);

    Tv->SetBranchAddress("run",&run_v);
    Tv->SetBranchAddress("subrun",&subrun_v);
    Tv->SetBranchAddress("event",&event_v);
    Tv->SetBranchAddress("entry",&entry_v);
    Tv->SetBranchAddress("roid",&roid_v);
    Tv->SetBranchAddress("vtxid",&vtxid_v);
    Tv->SetBranchAddress("adc_img_v", &adc_img_v);
    Tv->SetBranchAddress("scedr", &scedr);

    int nentries = T->GetEntries();
    */
    
    TCanvas *c3D = new TCanvas("c3D","c3D",800,800);
    TCanvas *cImage = new TCanvas("cImage","cImage",900,300);
    cImage->Divide(3,1);
    TH2D *hImage[3];
    TGraph *gEndPoint[3];
    TGraph *gROI[3];

    TH1D *hLengths     = new TH1D("hLengths",    "hLengths; L_{3D}[cm]; Entries(/1 cm)",200,0,200);
    TH1D *hMuonLikeE   = new TH1D("hMuonLikeE",  "hMuonLikeE; T[MeV]",200,0,1000);
    TH1D *hProtonLikeE = new TH1D("hProtonLikeE","hProtonLikeE; T[MeV]",200,0,1000);
    TH1D *hAvIon       = new TH1D("hAvIon","hAvIon",100,0,1000);
    TH2D *hdQdx        = new TH2D("hdQdx","hdQdx",200,0,1000,100,0,10);
    TH2D *hdQdx1       = new TH2D("hdQdx1","hdQdx1",200,0,1000,100,0,10);
    TH2D *hdQdx2       = new TH2D("hdQdx2","hdQdx2",200,0,1000,100,0,10);

    std::vector<std::vector<larcv::Image2D> > DataImages;
    std::vector<larcv::Image2D> SingleParticleImages(3);
    larcv::AStarTracker tracker;
    tracker.initialize();
    tracker.SetCompressionFactors(1,6);
    tracker.SetVerbose(0);

    TVector3 vertex, endPoint[2];

    //for(int i = 0;i<nentries;i++){

    //
    // Loop per vertex (larcv type is PGraph "Particle Graph")
    //
    
    // ev_img_v     = (EventImage2D*)mgr.get_data(kProductImage2D,"wire");
    // auto adc_img_v = &(ev_img_v->Image2DArray());

    auto ev_pgraph_v   = (EventPGraph*) mgr.get_data(kProductPGraph,"test");
    auto ev_pcluster_v = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_img");
    auto ev_ctor_v     = (EventPixel2D*)mgr.get_data(kProductPixel2D,"test_ctor");
    
    int run    = ev_pgraph_v->run();
    int subrun = ev_pgraph_v->subrun();
    int event  = ev_pgraph_v->event();
    
    // get the event clusters
    auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();

    // get the pixels corresponding to particle
    auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
    
    for(size_t pgraph_id = 0; pgraph_id < ev_pgraph_v->PGraphArray().size(); ++pgraph_id) {

      int i = pgraph_id;
      
      auto const& pgraph        = ev_pgraph_v->PGraphArray().at(pgraph_id);
      auto const& roi_v         = pgraph.ParticleArray();
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();

      // Get the 3D vertex position
      float x = pgraph.ParticleArray().front().X();
      float y = pgraph.ParticleArray().front().Y();
      float z = pgraph.ParticleArray().front().Z();

      size_t nparticles = cluster_idx_v.size();

      // Get the 3D end point particle 0
      auto particle0_end_x = pgraph.ParticleArray().front().EndPosition().X();
      auto particle0_end_y = pgraph.ParticleArray().front().EndPosition().Y();
      auto particle0_end_z = pgraph.ParticleArray().front().EndPosition().Z();

      // Get the 3D end point particle 1
      auto particle1_end_x = pgraph.ParticleArray().back().EndPosition().X();
      auto particle1_end_y = pgraph.ParticleArray().back().EndPosition().Y();
      auto particle1_end_z = pgraph.ParticleArray().back().EndPosition().Z();

      // Prepare the data image
      std::vector<Image2D> img2d_v;
      img2d_v.resize(3);
      auto adc_img_v = &img2d_v;
      
      // Loop per plane, get the particle pixels for this plane
      for(size_t plane=0; plane<3; ++plane) {
	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) continue;
	
	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) continue;
	
	// Retrieve the particle images and particle contours on this plane
	const auto& pcluster_v = (*iter_pcluster).second;
	const auto& ctor_v = (*iter_ctor).second;

	// Get this planes meta
	auto meta = roi_v.front().BB(plane);

	// Construct this image
	auto& img2d = img2d_v[plane];
	img2d = Image2D(meta);

	// For each particle, get the particle pixels and put them in the image
	for(size_t par_id=0; par_id < cluster_idx_v.size(); ++par_id) {
	  
	  auto cluster_idx = cluster_idx_v[par_id];
	  
	  const auto& pcluster = pcluster_v.at(cluster_idx);
	  
	  // There is no particle cluster on this plane
	  if (pcluster.empty()) continue;

	  for(size_t i=0;i<pcluster.size();++i) {
	    img2d.set_pixel(pcluster[i].X(),
			    pcluster[i].Y(),
			    pcluster[i].Intensity());
	  }
	} // end this particle
      } // end this plane

      
      //if(i > 1000) continue;
      //if(i != 273 && i != 290) continue;
      //Tv->GetEntry(i);
      //if(scedr > 5) continue;
      //T->GetEntry(i);
      std::cout << "Entry " << mgr.current_entry() << std::endl;

      //
      // GetEstimated 3D Start and End Points
      vertex.SetXYZ(x,y,z);
      endPoint[0].SetXYZ(particle0_end_x,particle0_end_y,particle0_end_z);
      endPoint[1].SetXYZ(particle1_end_x,particle1_end_y,particle1_end_z);

      std::vector<TVector3> EndPoints;
      EndPoints.push_back(vertex);
      EndPoints.push_back(endPoint[0]);
      EndPoints.push_back(endPoint[1]);

      bool thisPointOK = false;
      for(int iPoints = 0; iPoints < EndPoints.size(); iPoints++){
	thisPointOK = tracker.CheckEndPointsInVolume(EndPoints[iPoints]);
	if(!thisPointOK){std::cout << "point " << iPoints << " out of range" << std::endl; break;}
      }
      if(!thisPointOK){
	std::cout << "point out of range, do not look at this event anymore" << std::endl;
	continue;
      }
      else{std::cout << "points in range, continue with reco" << std::endl;}

      std::vector<larcv::ImageMeta> meta_v;
      for(int iPlane=0;iPlane<3;iPlane++){
	meta_v.push_back( (*adc_img_v)[iPlane].meta() );
      }
      //std::cout << "about to get the Time and Wire bounds" << std::endl;
      tracker.SetTimeAndWireBounds(EndPoints, meta_v);
      //std::cout << "time and wire bounds set" << std::endl;
      std::vector<std::pair<double,double> > time_bounds = tracker.GetTimeBounds();
      std::vector<std::pair<double,double> > wire_bounds = tracker.GetWireBounds();
      //std::cout << "time and wire bounds OK" << std::endl;
      std::vector<larcv::Image2D> data_images;
      for(int iPlane = 0;iPlane<3;iPlane++){
	//std::cout << "Time Bounds " << iPlane << " : " << time_bounds[iPlane].first << " , " << time_bounds[iPlane].second << std::endl;
	//std::cout << "Wire Bounds " << iPlane << " : " << wire_bounds[iPlane].first << " , " << wire_bounds[iPlane].second << std::endl;

	double image_width  = 1.*(size_t)((wire_bounds[iPlane].second - wire_bounds[iPlane].first)*(*adc_img_v)[iPlane].meta().pixel_width());
	double image_height = 1.*(size_t)((time_bounds[iPlane].second - time_bounds[iPlane].first)*(*adc_img_v)[iPlane].meta().pixel_height());
	size_t col_count    = (size_t)(image_width / (*adc_img_v)[iPlane].meta().pixel_width());
	size_t row_count    = (size_t)(image_height/ (*adc_img_v)[iPlane].meta().pixel_height());
	double origin_x     = (*adc_img_v)[iPlane].meta().tl().x+wire_bounds[iPlane].first*(*adc_img_v)[iPlane].meta().pixel_width();
	double origin_y     = (*adc_img_v)[iPlane].meta().br().y+time_bounds[iPlane].second*(*adc_img_v)[iPlane].meta().pixel_height();


	larcv::ImageMeta newMeta(image_width, image_height,
				 row_count, col_count,
				 origin_x,  // origin x = min wire
				 origin_y, // origin y = max time
				 iPlane);

	//std::cout << "getting ovelapping ROI" << std::endl;
	newMeta = newMeta.overlap((*adc_img_v)[iPlane].meta());
	//std::cout << "cropping meta plane " << iPlane << std::endl;
	larcv::Image2D newImage = (*adc_img_v)[iPlane].crop(newMeta);
	//larcv::Image2D newImage = (*adc_img_v)[iPlane];
	larcv::Image2D invertedImage = newImage;

	for(int icol = 0;icol<newImage.meta().cols();icol++){
	  for(int irow = 0;irow<newImage.meta().rows();irow++){
	    //std::cout << icol << "/" << newImage.meta().cols() << "\t" << newImage.meta().rows()-irow << "/" << newImage.meta().rows() << std::endl;
	    if(newImage.pixel(irow, icol) > 10){
	      invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,newImage.pixel(irow, icol));
	    }
	    else{
	      invertedImage.set_pixel(newImage.meta().rows()-irow-1, icol,0);
	    }
	  }
	}


	hImage[iPlane] = new TH2D(Form("hImage_%d_%d",2*i,iPlane),Form("hImage_%d_%d",2*i,iPlane),invertedImage.meta().cols(),invertedImage.meta().tl().x,invertedImage.meta().tl().x+invertedImage.meta().width(),invertedImage.meta().rows(),invertedImage.meta().br().y,invertedImage.meta().br().y+invertedImage.meta().height());

	for(int icol = 0;icol<invertedImage.meta().cols();icol++){
	  for(int irow = 0;irow<invertedImage.meta().rows();irow++){
	    if(invertedImage.pixel(irow, icol) < 10) continue;
	    hImage[iPlane]->SetBinContent(icol+1,irow+1,invertedImage.pixel(irow, icol));
	  }
	}
	cImage->cd(iPlane+1);
	hImage[iPlane]->Draw("colz");
	gROI[iPlane] = new TGraph();
	gROI[iPlane]->SetPoint(0,newMeta.tl().x,newMeta.tl().y);
	gROI[iPlane]->SetPoint(1,newMeta.tr().x,newMeta.tr().y);
	gROI[iPlane]->SetPoint(2,newMeta.br().x,newMeta.br().y);
	gROI[iPlane]->SetPoint(3,newMeta.bl().x,newMeta.bl().y);
	gROI[iPlane]->SetPoint(4,newMeta.tl().x,newMeta.tl().y);
	gROI[iPlane]->SetLineWidth(3);
	gROI[iPlane]->SetLineColor(2);
	gROI[iPlane]->Draw("same LP");

	gEndPoint[iPlane] = new TGraph();
	gEndPoint[iPlane]->SetMarkerStyle(4);
	gEndPoint[iPlane]->SetMarkerColor(1);

	for(int point = 0;point<EndPoints.size();point++){
	  double _parent_x = EndPoints[point].X();
	  double _parent_y = EndPoints[point].Y();
	  double _parent_z = EndPoints[point].Z();
	  double _parent_t = 0;
	  double x_pixel, y_pixel;
	  ProjectTo3D(newImage.meta(),_parent_x,_parent_y,_parent_z,_parent_t,iPlane,x_pixel,y_pixel); // y_pixel is time
	  gEndPoint[iPlane]->SetPoint(point,(x_pixel+0.5)*newMeta.pixel_width()+newImage.meta().tl().x,
				      (y_pixel+0.5)*newMeta.pixel_height()+newImage.meta().br().y);
	}
	gEndPoint[iPlane]->Draw("same P");

	data_images.push_back(invertedImage);

      }

      std::vector<double> Lengths;
      std::vector< std::vector<TVector3> > lists;
      std::vector< std::vector<std::vector<double> > > dQdx;

      // particle 1
      tracker.SetTrackInfo(run, subrun, event, 2*i);
      tracker.SetImages(data_images);
      tracker.SetEndPoints(EndPoints[0],EndPoints[1]);
      tracker.Reconstruct();
      lists.push_back(tracker.GetTrack());
      tracker.ComputedQdX();
      dQdx.push_back(tracker.GetdQdx());
      Lengths.push_back(tracker.GetLength());

      // particle 2
      tracker.SetTrackInfo(run, subrun, event, 2*i+1);
      tracker.SetEndPoints(EndPoints[0],EndPoints[2]);
      tracker.Reconstruct();
      lists.push_back(tracker.GetTrack());
      tracker.ComputedQdX();
      dQdx.push_back(tracker.GetdQdx());
      Lengths.push_back(tracker.GetLength());
      // done

      for(size_t ilength = 0;ilength<Lengths.size();ilength++){
	std::cout << Form("particle %03zu : %.1f cm",ilength+1,Lengths[ilength]) << std::endl;
	hLengths->Fill(Lengths[ilength]);
	hMuonLikeE->Fill(sMuonRange2T->Eval(Lengths[ilength]));
	hProtonLikeE->Fill(sProtonRange2T->Eval(Lengths[ilength]));
      }

      double calib = 0.01;

      for(int iPart = 0;iPart<dQdx.size();iPart++){
	double AvdQdx = 0;
	for(int iNode = 0;iNode<dQdx[iPart].size(); iNode++){
	  double dqdx = dQdx[iPart][iNode][0]+dQdx[iPart][iNode][1]+dQdx[iPart][iNode][2];
	  AvdQdx+=dqdx/dQdx[iPart].size();
	  double Xremain = 0;
	  for(int j = iNode;j<lists[iPart].size()-1;j++){
	    TVector3 A = lists[iPart][j];
	    TVector3 B = lists[iPart][j+1];
	    Xremain+=(B-A).Mag();
	  }
	  Xremain*=1.5;
	  if((i == 273 || i == 290) && iPart == 1)hdQdx2->Fill(sMuonRange2T->Eval(Xremain),calib*dqdx);// is it muon-like?
	  if((i == 273 || i == 290) && iPart == 0)hdQdx1->Fill(sProtonRange2T->Eval(Xremain),calib*dqdx);// is it proton-like?
	}
	hAvIon->Fill(AvdQdx);
      }

      std::vector<TVector3> list;
      if(lists[0].size() >= 1){
	for(int iPoint = 0 ; iPoint<lists[0].size(); iPoint++){list.push_back(lists[0][iPoint]);}
      }
      std::cout << "list 1 done" << std::endl;
      if(lists[1].size() >= 1){
	for(int iPoint = lists[1].size()-1; iPoint<lists[1].size(); iPoint--){list.push_back(lists[1][iPoint]);}
      }
      std::cout << "list 2 done" << std::endl;
      TGraph *gTrack[3];
      TGraph2D *gTrack2D = new TGraph2D();

      for(int iPlane=0;iPlane<3;iPlane++){
	gTrack[iPlane] = new TGraph();
	for(int ipoint = 0; ipoint<list.size();ipoint++){
	  double x_pixel, y_pixel;
	  ProjectTo3D(data_images[iPlane].meta(),list[ipoint].X(),list[ipoint].Y(),list[ipoint].Z(),0,iPlane,x_pixel,y_pixel); // y_pixel is time
	  double time = y_pixel*data_images[iPlane].meta().pixel_height()+data_images[iPlane].meta().br().y;
	  gTrack[iPlane]->SetPoint(ipoint,x_pixel*data_images[iPlane].meta().pixel_width()+data_images[iPlane].meta().tl().x,time);
	}

	cImage->cd(iPlane+1);
	gTrack[iPlane]->SetLineWidth(1);
	gTrack[iPlane]->SetLineColor(2);
	gTrack[iPlane]->Draw("same LP");
      }

      cImage->Modified();
      cImage->Update();
      cImage->SaveAs(Form("cImage_%04d.pdf",i));

      c3D->cd();
      for(int iPoint = 0;iPoint<list.size();iPoint++){
	gTrack2D->SetPoint(iPoint,list[iPoint].X(),list[iPoint].Y(),list[iPoint].Z());
      }
      gTrack2D->Draw("ALP");
      c3D->Modified();
      c3D->Update();
      c3D->SaveAs(Form("c3D_%04d.root",i));
    }
    tracker.finalize();

    TCanvas *cLengths = new TCanvas("cLengths","cLengths",800,800);
    hLengths->Draw();
    cLengths->SaveAs("cLengths.pdf");
    cLengths->SaveAs("cLengths.root");

    TCanvas *cEnergies = new TCanvas("cEnergies","cEnergies",800,800);
    hMuonLikeE->SetLineColor(1);
    hMuonLikeE->SetLineWidth(2);
    hMuonLikeE->Draw();

    hProtonLikeE->SetLineColor(2);
    hProtonLikeE->SetLineWidth(2);
    hProtonLikeE->Draw("same");
    cEnergies->SaveAs("cEnergies.pdf");
    cEnergies->SaveAs("cEnergies.root");

    TCanvas *cdQdx = new TCanvas("cdQdx","cdQdx",800,800);
    hdQdx1->SetMarkerStyle(7);
    hdQdx1->SetMarkerColor(2);
    hdQdx2->SetMarkerStyle(7);
    hdQdx2->SetMarkerColor(1);
    hdQdx1->Draw();
    hdQdx2->Draw("same");
    sMuonT2dEdx->Draw("same");
    sProtonT2dEdx->Draw("same");
    cdQdx->SaveAs("cdQdx.pdf");
    cdQdx->SaveAs("cdQdx.root");

    TCanvas *cAverageIon = new TCanvas("cAverageIon","cAverageIon",800,800);
    hAvIon->Draw();
    cAverageIon->SaveAs("cAverageIon.pdf");
    cAverageIon->SaveAs("cAverageIon.root");
    
    return true;
  }

  void ReadRuiFile::finalize()
  {}

}
#endif
