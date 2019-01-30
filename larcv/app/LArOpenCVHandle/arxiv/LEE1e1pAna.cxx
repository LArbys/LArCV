#ifndef __LEE1E1PANA_CXX__
#define __LEE1E1PANA_CXX__

#include "LEE1e1pAna.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventImage2D.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterTypes.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArbysUtils.h"
#include <cassert>

namespace larcv {

  static LEE1e1pAnaProcessFactory __global_LEE1e1pAnaProcessFactory__;
  
  LEE1e1pAna::LEE1e1pAna(const std::string name)
    : ProcessBase(name)
  {}
    
  void LEE1e1pAna::configure(const PSet& cfg)
  {
    _img2d_prod         = cfg.get<std::string>("Image2DProducer");
    _pgraph_prod        = cfg.get<std::string>("PGraphProducer");
    _pcluster_ctor_prod = cfg.get<std::string>("PxContourProducer");
    _pcluster_img_prod  = cfg.get<std::string>("PxImageProducer");  
    _truth_roi_prod     = cfg.get<std::string>("TrueROIProducer");
    _reco_roi_prod      = cfg.get<std::string>("RecoROIProducer");

    LARCV_DEBUG() << "Image2DProducer:  " << _img2d_prod << std::endl;
    LARCV_DEBUG() << "PGraphProducer:   " << _pgraph_prod << std::endl;
    LARCV_DEBUG() << "PxContourProducer: " << _pcluster_ctor_prod << std::endl;
    LARCV_DEBUG() << "PxImageProducer:  " << _pcluster_img_prod << std::endl;
    LARCV_DEBUG() << "TrueROIProducer:  " << _truth_roi_prod << std::endl;
    LARCV_DEBUG() << "RecoROIProducer:  " << _reco_roi_prod << std::endl;

  }
  
  void LEE1e1pAna::initialize()
  {
    
    _tree = new TTree("LEE1e1pTree","");
    _tree->Branch("run",&_run,"run/I");
    _tree->Branch("subrun",&_subrun,"subrun/I");
    _tree->Branch("event",&_event,"event/I");
    _tree->Branch("entry",&_entry,"entry/I");
    _tree->Branch("roid",&_roid,"roid/I");
    _tree->Branch("vtxid",&_vtxid,"vtxid/I");
    
    _tree->Branch("shape0",&_shape0,"shape0/I");
    _tree->Branch("shape1",&_shape1,"shape1/I");

    _tree->Branch("q0",&_q0,"q0/D");
    _tree->Branch("q1",&_q1,"q1/D");
    
    _tree->Branch("npx0",&_npx0,"npx0/I");
    _tree->Branch("npx1",&_npx1,"npx1/I");

    _tree->Branch("nprotons",&_nprotons,"nprotons/I");
    _tree->Branch("nothers", &_nothers,   "nothers/I");    

    _tree->Branch("area0",&_area0,"area0/D");
    _tree->Branch("area1",&_area1,"area1/D");

    _tree->Branch("len0",&_len0,"len0/D");
    _tree->Branch("len1",&_len1,"len1/D");
  }

  bool LEE1e1pAna::process(IOManager& mgr)
  {
    auto const ev_img2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_img2d_prod));
    if (!ev_img2d) throw larbys("Invalid image producer provided");
    
    auto const ev_pgraph = (EventPGraph*)(mgr.get_data(kProductPGraph,_pgraph_prod));
    if (!ev_pgraph) throw larbys("Invalid pgraph producer provided!");

    auto const ev_ctor_v = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_ctor_prod));
    if (!ev_ctor_v) throw larbys("Invalid Contour Pixel2D producer provided!");

    auto const ev_pcluster_v = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_img_prod));
    if (!ev_pcluster_v) throw larbys("Invalid Particle Pixel2D producer provided!");

    EventROI* ev_roi_v = nullptr;
    if (!_truth_roi_prod.empty()) {
      ev_roi_v = (EventROI*)(mgr.get_data(kProductROI,_truth_roi_prod));
      if (!ev_roi_v) throw larbys("Invalid truth roi producer provided");
      if (ev_roi_v->ROIArray().empty()) throw larbys("Empty truth roi producer provided");
    }
    
    auto const ev_croi_v     = (EventROI*)(mgr.get_data(kProductROI,_reco_roi_prod));
    if (!ev_croi_v) throw larbys("Invalid cROI producer provided");

    ClearEvent();
    
    _run    = ev_img2d->run();
    _subrun = ev_img2d->subrun();
    _event  = ev_img2d->event();
    _entry  = mgr.current_entry();
    
    LARCV_DEBUG() << "Got RSEE=("<<_run<<","<<_subrun<<","<<_event<<","<<_entry<<")"<<std::endl;
    const auto& adc_img_v = ev_img2d->Image2DArray();
    
    std::vector<std::vector<ImageMeta> > roid_v;
    roid_v.reserve(ev_croi_v->ROIArray().size());
    
    for(auto const& croi : ev_croi_v->ROIArray()) {
      auto const& bb_v = croi.BB();
      roid_v.push_back(crop_metas(adc_img_v,bb_v));
    }
    
    _nprotons = 0;
    _nothers  = 0;
    
    auto const& ctor_m = ev_ctor_v->Pixel2DClusterArray();
    auto const& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
    
    std::vector<size_t> plane_order_v = {2,0,1};
    
    _vtxid = -1;
    
    auto vtx_counts = ev_pgraph->PGraphArray().size();
	
    LARCV_DEBUG() << "Got " << vtx_counts << " vertices" << std::endl;
    for (int vtx_idx = 0; vtx_idx < (int)vtx_counts; ++vtx_idx) {

      LARCV_DEBUG() << "@ vtx_idx=" << vtx_idx << std::endl;

      ClearVertex();

      _vtxid += 1;

      assert((size_t)vtx_idx < ev_pgraph->PGraphArray().size());

      auto pgraph = ev_pgraph->PGraphArray().at(vtx_idx);
      auto const& roi_v = pgraph.ParticleArray();
      
      auto const& bb_v = roi_v.front().BB();

      auto iter = std::find(roid_v.begin(),roid_v.end(),bb_v);
      if (iter == roid_v.end()) throw larbys("Unknown image meta");
	
      auto roid = iter - roid_v.begin();

      if (roid!=_roid) _vtxid = 0;
      
      _roid  = roid;

      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      LARCV_DEBUG() << "cluster index array size=" << cluster_idx_v.size() << std::endl;
      if (roi_v.size() < 2) {
	_tree->Fill();
	continue;
      }

      auto const& roi0 = roi_v.at(0);
      LARCV_DEBUG() << "got roi0@" << &roi0 << "..." << std::endl;
      auto const& roi1 = roi_v.at(1);
      LARCV_DEBUG() << "got roi1@" << &roi1 << "..." << std::endl;

      _shape0 = (int)(roi0.Shape());
      _shape1 = (int)(roi1.Shape());

      auto const& cluster_idx0 = cluster_idx_v.at(0);
      auto const& cluster_idx1 = cluster_idx_v.at(1);

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
	
	if (cluster_idx0 > pcluster_v.size()) {
	  LARCV_CRITICAL() << "idx0 " << cluster_idx0 << " vs " << pcluster_v.size() << std::endl;
	  throw larbys("die");
	}

	if (cluster_idx1 > pcluster_v.size()) {
	  LARCV_CRITICAL() << "idx1 " << cluster_idx1 << " vs " << pcluster_v.size() << std::endl;
	  throw larbys("die");
	}

	auto const& pcluster0 = pcluster_v.at(cluster_idx0);
	auto const& ctor0 = ctor_v.at(cluster_idx0);

	if(!done0 && ctor0.size()>2) {
	  _npx0 = pcluster0.size();
	  for(auto const& pt : pcluster0) _q0 += pt.Intensity();
	  for(size_t i=1; i<ctor0.size(); ++i) {
	    auto const& pt0 = ctor0.at(i-1);
	    auto const& pt1 = ctor0.at(i);
	    _len0 += sqrt(pow((float)(pt0.X()) - (float)(pt1.X()),2) + pow((float)(pt0.Y()) - (float)(pt1.Y()),2));
	  }
	  _len0 += sqrt(pow((float)(ctor0.front().X()) - (float)(ctor0.back().X()),2) +
			pow((float)(ctor0.front().Y()) - (float)(ctor0.back().Y()),2));
	  larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor0.size());
	  for(size_t i=0; i<ctor0.size(); ++i) {
	    ctor.at(i).x = ctor0.at(i).X();
	    ctor.at(i).y = ctor0.at(i).Y();
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
	    auto const& pt1 = ctor1.at(i);
	    _len1 += sqrt(pow((float)(pt0.X()) - (float)(pt1.X()),2) + pow((float)(pt0.Y()) - (float)(pt1.Y()),2));
	  }
	  _len1 += sqrt(pow((float)(ctor1.front().X()) - (float)(ctor1.back().X()),2) +
			pow((float)(ctor1.front().Y()) - (float)(ctor1.back().Y()),2));
	  larocv::GEO2D_Contour_t ctor;
	  ctor.resize(ctor1.size());
	  for(size_t i=0; i<ctor1.size(); ++i) {
	    ctor.at(i).x = ctor1.at(i).X();
	    ctor.at(i).y = ctor1.at(i).Y();
	  }
	  _area1 = larocv::ContourArea(ctor);
	  done1 = true;
	}
	if(done0 && done1) break;
      }
      _tree->Fill();
    }
    return true;
  }

  void LEE1e1pAna::ClearEvent() {
    _vtxid = kINVALID_INT;

    _nprotons = kINVALID_INT;
    _nothers = kINVALID_INT;
    
    ClearVertex();
  }
  
  void LEE1e1pAna::ClearVertex() {

    _shape0 = kINVALID_INT;
    _shape1 = kINVALID_INT;

    _npx0 = kINVALID_INT;
    _npx1 = kINVALID_INT;
    
    _q0 = kINVALID_DOUBLE;
    _q1 = kINVALID_DOUBLE;

    _area0 = kINVALID_DOUBLE;
    _area1 = kINVALID_DOUBLE;

    _len0 = kINVALID_DOUBLE;
    _len1 = kINVALID_DOUBLE;
  }
  
  void LEE1e1pAna::finalize()
  {
    if(has_ana_file()) {
      _tree->Write();
    }
  }

}
#endif
