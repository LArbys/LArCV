#ifndef __LEE1E1PANA_CXX__
#define __LEE1E1PANA_CXX__

#include "LEE1e1pAna.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventImage2D.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterTypes.h"
#include "opencv2/imgproc.hpp"
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArbysUtils.h"
namespace larcv {

  static LEE1e1pAnaProcessFactory __global_LEE1e1pAnaProcessFactory__;
  
  LEE1e1pAna::LEE1e1pAna(const std::string name)
    : ProcessBase(name)
  {}
    
  void LEE1e1pAna::configure(const PSet& cfg)
  {
    _pgraph_prod        = cfg.get<std::string>("PGraphProducer","test");
    _pcluster_ctor_prod = cfg.get<std::string>("PxContourProducer","test_ctor");
    _pcluster_img_prod  = cfg.get<std::string>("PxImageProducer","test_img");
    _truth_roi_prod     = cfg.get<std::string>("TrueROIProducer","tpc");
    _reco_roi_prod      = cfg.get<std::string>("RecoROIProducer","croi_merge");
  }
    
  
  void LEE1e1pAna::initialize()
  {
    _score0.resize(5,0.0);
    _score1.resize(5,0.0);
    
    _event_tree = new TTree("event_tree","");
    _event_tree->Branch("entry",&_entry,"entry/I");
    _event_tree->Branch("run",&_run,"run/I");
    _event_tree->Branch("subrun",&_subrun,"subrun/I");
    _event_tree->Branch("event",&_event,"event/I");
    _event_tree->Branch("tx",&_tx,"tx/D");
    _event_tree->Branch("ty",&_ty,"ty/D");
    _event_tree->Branch("tz",&_tz,"tz/D");
    _event_tree->Branch("te",&_te,"te/D");
    _event_tree->Branch("tt",&_tt,"tt/D");
    _event_tree->Branch("scex",&_scex,"scex/D");
    _event_tree->Branch("scey",&_scey,"scey/D");
    _event_tree->Branch("scez",&_scez,"scez/D");
    _event_tree->Branch("good_croi0",&_good_croi0,"good_croi0/I");
    _event_tree->Branch("good_croi1",&_good_croi1,"good_croi1/I");
    _event_tree->Branch("good_croi2",&_good_croi2,"good_croi2/I");
    _event_tree->Branch("area_croi0",&_area_croi0,"area_croi0/D");
    _event_tree->Branch("area_croi1",&_area_croi1,"area_croi1/D");
    _event_tree->Branch("area_croi2",&_area_croi2,"area_croi2/D");
    _event_tree->Branch("good_croi_ctr",&_good_croi_ctr,"good_croi_ctr/I");
    _event_tree->Branch("num_croi",&_num_croi,"num_croi/I");
    _event_tree->Branch("min_vtx_dist",&_min_vtx_dist,"min_vtx_dist/D");
    
    _tree = new TTree("tree","");
    _tree->Branch("entry",&_entry,"entry/I");
    _tree->Branch("run",&_run,"run/I");
    _tree->Branch("subrun",&_subrun,"subrun/I");
    _tree->Branch("event",&_event,"event/I");
    _tree->Branch("tx",&_tx,"tx/D");
    _tree->Branch("ty",&_ty,"ty/D");
    _tree->Branch("tz",&_tz,"tz/D");
    _tree->Branch("te",&_te,"te/D");
    _tree->Branch("tt",&_tt,"tt/D");
    _tree->Branch("scex",&_scex,"scex/D");
    _tree->Branch("scey",&_scey,"scey/D");
    _tree->Branch("scez",&_scez,"scez/D");
    
    _tree->Branch("x",&_x,"x/D");
    _tree->Branch("y",&_y,"y/D");
    _tree->Branch("z",&_z,"z/D");
    _tree->Branch("dr",&_dr,"dr/D");
    _tree->Branch("scedr",&_scedr,"scedr/D");

    
    _tree->Branch("shape0",&_shape0,"shape0/I");
    _tree->Branch("shape1",&_shape1,"shape1/I");
    _tree->Branch("score0","std::vector<double>",&_score0);
    _tree->Branch("score1","std::vector<double>",&_score1);

    _tree->Branch("score_shower0",&_score_shower0,"score_shower0/D");
    _tree->Branch("score_shower1",&_score_shower1,"score_shower1/D");
    _tree->Branch("score_track0",&_score_track0,"score_track0/D");
    _tree->Branch("score_track1",&_score_track1,"score_track1/D");


    _tree->Branch("q0",&_q0,"q0/D");
    _tree->Branch("q1",&_q1,"q1/D");
    _tree->Branch("npx0",&_npx0,"npx0/I");
    _tree->Branch("npx1",&_npx1,"npx1/I");
    _tree->Branch("area0",&_area0,"area0/D");
    _tree->Branch("area1",&_area1,"area1/D");

    _tree->Branch("len0",&_len0,"len0/D");
    _tree->Branch("len1",&_len1,"len1/D");
    
    _tree->Branch("score1_e",&_score1_e,"score1_e/D");
    _tree->Branch("score1_g",&_score1_g,"score1_g/D");
    _tree->Branch("score1_pi",&_score1_pi,"score1_pi/D");
    _tree->Branch("score1_mu",&_score1_mu,"score1_mu/D");
    _tree->Branch("score1_p",&_score1_p,"score1_p/D");
    
    _tree->Branch("score0_e",&_score0_e,"score0_e/D");
    _tree->Branch("score0_g",&_score0_g,"score0_g/D");
    _tree->Branch("score0_pi",&_score0_pi,"score0_pi/D");
    _tree->Branch("score0_mu",&_score0_mu,"score0_mu/D");
    _tree->Branch("score0_p",&_score0_p,"score0_p/D");
    
  }

  bool LEE1e1pAna::process(IOManager& mgr)
  {

    auto const ev_pgraph     = (EventPGraph*)(mgr.get_data(kProductPGraph,_pgraph_prod));
    auto const ev_ctor_v     = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_ctor_prod));
    auto const ev_pcluster_v = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_img_prod));
    auto const ev_roi_v      = (EventROI*)(mgr.get_data(kProductROI,_truth_roi_prod));
    auto const ev_croi_v     = (EventROI*)(mgr.get_data(kProductROI,_reco_roi_prod));

    _run    = ev_pgraph->run();
    _subrun = ev_pgraph->subrun();
    _event  = ev_pgraph->event();
    _entry  = mgr.current_entry();

    _tx = _ty = _tz = _tt = _te = -1.;
    _scex = _scey = _scez = -1.;

    for(auto const& roi : ev_roi_v->ROIArray()){
      if(std::abs(roi.PdgCode()) == 12 || std::abs(roi.PdgCode()) == 14) {
	_tx = roi.X();
	_ty = roi.Y();
	_tz = roi.Z();
	_tt = roi.T();
	_te = roi.EnergyInit();
	auto const offset = _sce.GetPosOffsets(_tx,_ty,_tz);
	_scex = _tx - offset[0] + 0.7;
	_scey = _ty + offset[1];
	_scez = _tz + offset[2];
      }
    }
    double xyz[3];
    xyz[0] = _scex;
    xyz[1] = _scey;
    xyz[2] = _scez;
    
    auto geo = larutil::Geometry::GetME();
    auto larp = larutil::LArProperties::GetME();
    double wire_v[3];
    try {
      wire_v[0] = geo->NearestWire(xyz,0);
      wire_v[1] = geo->NearestWire(xyz,1);
      wire_v[2] = geo->NearestWire(xyz,2);
    }catch(const std::exception& e) {
      std::cout<<xyz[0]<<" "<<xyz[1]<<" " <<xyz[2]<<std::endl;
      throw e;
    }
    const double tick = (_scex / larp->DriftVelocity() + 4) * 2. + 3200.;
    _num_croi  = ev_croi_v->ROIArray().size();
    _area_croi0 = 0.;
    _area_croi1 = 0.;
    _area_croi2 = 0.;
    _good_croi0 = 0;
    _good_croi1 = 0;
    _good_croi2 = 0;
    _good_croi_ctr = 0;
    
    for(auto const& croi : ev_croi_v->ROIArray()) {
      
      auto const& bb_v = croi.BB();
      size_t good_croi_ctr = 0;
      for(size_t plane=0; plane<bb_v.size(); ++plane) {
	auto const& croi_meta = bb_v[plane];
	auto const& wire = wire_v[plane];
	if( croi_meta.min_x() <= wire && wire <= croi_meta.max_x() &&
	    croi_meta.min_y() <= tick && tick <= croi_meta.max_y() )
	  ++good_croi_ctr;
      }
      if(good_croi_ctr <= _good_croi_ctr) continue;
      if(good_croi_ctr > 1) _good_croi_ctr+=1;
      
      for(size_t plane=0; plane<bb_v.size(); ++plane) {
	auto const& croi_meta = bb_v[plane];
	auto const& wire = wire_v[plane];
	if( croi_meta.min_x() <= wire && wire <= croi_meta.max_x() &&
	    croi_meta.min_y() <= tick && tick <= croi_meta.max_y() ) {
	  if(plane == 0) _good_croi0 = 1;
	  if(plane == 1) _good_croi1 = 1;
	  if(plane == 2) _good_croi2 = 1;
	}
	if(plane == 0) _area_croi0 += (croi_meta.rows() * croi_meta.cols());
	if(plane == 1) _area_croi1 += (croi_meta.rows() * croi_meta.cols());
	if(plane == 2) _area_croi2 += (croi_meta.rows() * croi_meta.cols());
      }
    }
      
    auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
    
    std::vector<size_t> plane_order_v;
    plane_order_v.push_back(2);
    plane_order_v.push_back(0);
    plane_order_v.push_back(1);
    
    _min_vtx_dist = 1.e9;
    
    for(auto const& pgraph : ev_pgraph->PGraphArray()) {
      auto const& roi_v = pgraph.ParticleArray();
      if(roi_v.size()!=2) continue;
      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      auto const& roi0 = roi_v[0];
      auto const& roi1 = roi_v[1];
      _shape0 = (int)(roi0.Shape());
      _shape1 = (int)(roi1.Shape());
      for(auto& v : _score0) v=0.;
      for(auto& v : _score1) v=0.;
      auto const& score0 = roi0.TypeScore();
      auto const& score1 = roi1.TypeScore();
      
      _x = roi0.X();
      _y = roi0.Y();
      _z = roi0.Z();
      
      _dr = sqrt(pow(_x - _tx,2)+pow(_y - _ty,2)+pow(_z - _tz,2));
      _scedr = sqrt(pow(_x - _scex,2)+pow(_y - _scey,2)+pow(_z - _scez,2));
      if(_scedr < _min_vtx_dist) _min_vtx_dist = _scedr;
      
      _score_shower0 = _score_shower1 = _score_track0 = _score_track1 = 0;
      _score0_e = _score0_g = _score0_mu = _score0_pi = _score0_p = 0;
      _score1_e = _score1_g = _score1_mu = _score1_pi = _score1_p = 0;
      
      for(size_t i=0; i<score0.size() && i<_score0.size(); ++i) {
	_score0[i] = score0[i];
	if(i<2) _score_shower0 += score0[i];
	if(i==2||i==4) _score_track0 += score0[i];
	if(i==0) _score0_e  = score0[i];
	if(i==1) _score0_g  = score0[i];
	if(i==2) _score0_mu = score0[i];
	if(i==3) _score0_pi = score0[i];
	if(i==4) _score0_p  = score0[i];
      }
      for(size_t i=0; i<score1.size() && i<_score1.size(); ++i) {
	_score1[i] = score1[i];
	if(i<2) _score_shower1 += score1[i];
	if(i==2||i==4) _score_track1 += score1[i];
	if(i==0) _score1_e  = score1[i];
	if(i==1) _score1_g  = score1[i];
	if(i==2) _score1_mu = score1[i];
	if(i==3) _score1_pi = score1[i];
	if(i==4) _score1_p  = score1[i];
      }

      auto const& cluster_idx0 = cluster_idx_v[0];
      auto const& cluster_idx1 = cluster_idx_v[1];

      bool done0=false;
      bool done1=false;
      _npx0 = _npx1 = 0;
      _len0 = _len1 = _area0 = _area1 = 0.;
      _q0 = _q1 = 0.;
      for(auto const& plane : plane_order_v) {

	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) continue;
	
	auto iter_ctor = ctor_m.find(plane);
	if(iter_ctor == ctor_m.end()) continue;

	auto const& pcluster_v = (*iter_pcluster).second;
	auto const& ctor_v = (*iter_ctor).second;
	
	auto const& pcluster0 = pcluster_v.at(cluster_idx0);
	auto const& ctor0 = ctor_v.at(cluster_idx0);
	if(!done0 && ctor0.size()>2) {
	  _npx0 = pcluster0.size();
	  for(auto const& pt : pcluster0) _q0 += pt.Intensity();
	  for(size_t i=1; i<ctor0.size(); ++i) {
	    auto const& pt0 = ctor0[i-1];
	    auto const& pt1 = ctor0[i];
	    _len0 += sqrt(pow((float)(pt0.X()) - (float)(pt1.X()),2) + pow((float)(pt0.Y()) - (float)(pt1.Y()),2));
	  }
	  _len0 += sqrt(pow((float)(ctor0.front().X()) - (float)(ctor0.back().X()),2)
			+
			pow((float)(ctor0.front().Y()) - (float)(ctor0.back().Y()),2));
	  ::larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor0.size());
	  for(size_t i=0; i<ctor0.size(); ++i) {
	    ctor[i].x = ctor0[i].X();
	    ctor[i].y = ctor0[i].Y();
	  }
	  _area0 = larocv::ContourArea(ctor);

	  done0 = true;
	}
	auto const& pcluster1 = pcluster_v.at(cluster_idx1);
	auto const& ctor1 = ctor_v.at(cluster_idx1);
	if(!done1 && ctor1.size()>2) {
	  _npx1 = pcluster1.size();
	  for(auto const& pt : pcluster1) _q1 += pt.Intensity();
	  for(size_t i=1; i<ctor1.size(); ++i) {
	    auto const& pt0 = ctor1[i-1];
	    auto const& pt1 = ctor1[i];
	    _len1 += sqrt(pow((float)(pt0.X()) - (float)(pt1.X()),2) + pow((float)(pt0.Y()) - (float)(pt1.Y()),2));
	  }
	  _len1 += sqrt(pow((float)(ctor1.front().X()) - (float)(ctor1.back().X()),2)
			+
			pow((float)(ctor1.front().Y()) - (float)(ctor1.back().Y()),2));
	  _len1 += sqrt(pow(ctor1.front().X()-ctor1.back().X(),2)+pow(ctor1.front().Y()-ctor1.back().Y(),2));
	  ::larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor1.size());
	  for(size_t i=0; i<ctor1.size(); ++i) {
	    ctor[i].x = ctor1[i].X();
	    ctor[i].y = ctor1[i].Y();
	  }
	  _area1 = ::larocv::ContourArea(ctor);
	  
	  done1 = true;
	}
	if(done0 && done1) break;
      }
      _tree->Fill();
    }
    _event_tree->Fill();
    return true;
  }

  void LEE1e1pAna::finalize()
  {
    if(has_ana_file()) {
      _tree->Write();
      _event_tree->Write();
    }
  }

}
#endif
