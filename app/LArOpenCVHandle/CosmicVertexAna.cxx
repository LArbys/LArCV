#ifndef __COSMICVERTEXANA_CXX__
#define __COSMICVERTEXANA_CXX__

#include "CosmicVertexAna.h"
#include "LArbysImageMaker.h"
#include "VertexAna.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventImage2D.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterTypes.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArbysUtils.h"
#include "LArbysImageMaker.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include "LArOpenCV/Core/larbys.h"
#include "Geo2D/Core/VectorArray.h"
#include "CVUtil/CVUtil.h"

namespace larcv {

  static CosmicVertexAnaProcessFactory __global_CosmicVertexAnaProcessFactory__;

  CosmicVertexAna::CosmicVertexAna(const std::string name) :
    ProcessBase(name),
    _LArbysImageMaker(),
    _tree(nullptr)
  {}
    
  void CosmicVertexAna::configure(const PSet& cfg)
  {
    _img2d_prod         = cfg.get<std::string>("Image2DProducer");
    _pgraph_prod        = cfg.get<std::string>("PGraphProducer");
    _pcluster_ctor_prod = cfg.get<std::string>("PxContourProducer");
    _pcluster_img_prod  = cfg.get<std::string>("PxImageProducer");
    _thrumu_img_prod = cfg.get<std::string>("ThruMuProducer");
    _stopmu_img_prod = cfg.get<std::string>("StopMuProducer");
    _reco_roi_prod      = cfg.get<std::string>("RecoROIProducer");
    auto tags_datatype  = cfg.get<size_t>("CosmicTagDataType");
    _tags_datatype = (ProductType_t) tags_datatype;

    _LArbysImageMaker.Configure(cfg.get<PSet>("LArbysImageMaker"));

    LARCV_DEBUG() << "Image2DProducer:  " << _img2d_prod << std::endl;
    LARCV_DEBUG() << "PGraphProducer:   " << _pgraph_prod << std::endl;
    LARCV_DEBUG() << "PxContouProducer: " << _pcluster_ctor_prod << std::endl;
    LARCV_DEBUG() << "PxImageProducer:  " << _pcluster_img_prod << std::endl;
    LARCV_DEBUG() << "RecoROIProducer:  " << _reco_roi_prod << std::endl;
  }

  void CosmicVertexAna::initialize()
  {
    _tree = new TTree("CosmicVertexTree","");

    _tree->Branch("run"         , &_run       , "run/I"       );
    _tree->Branch("subrun"      , &_subrun    , "subrun/I"    );
    _tree->Branch("roid"        , &_roid      , "roid/I"      );
    _tree->Branch("vtxid"       , &_vtxid     , "vtxid/I"     );
    _tree->Branch("event"       , &_event     , "event/I"     );
    _tree->Branch("entry"       , &_entry     , "entry/I"     );
 
    _tree->Branch("num_nottag_v"        , &_num_nottag_v        );
    _tree->Branch("num_stoptag_v"       , &_num_stoptag_v       );
    _tree->Branch("num_thrutag_v"       , &_num_thrutag_v       );
    _tree->Branch("num_allpix_v"        , &_num_allpix_v        );
    _tree->Branch("pts_in_raw_clus_v"   , &_pts_in_raw_clus_v   );
    _tree->Branch("pts_stopmu_ovrlap_v" , &_pts_stopmu_ovrlap_v );
    _tree->Branch("pts_thrumu_ovrlap_v" , &_pts_thrumu_ovrlap_v );

    _tree->Branch("endCos_v"       , &_endCos_v       );

  }

  
  bool CosmicVertexAna::process(IOManager& mgr)
  {
    bool has_reco_vtx = false;

    // Get Image2D vector
    auto const ev_img2d = (EventImage2D*)(mgr.get_data(kProductImage2D,_img2d_prod));
    if (!ev_img2d) throw larbys("Invalid image producer provided");

    //Get cROI vector
    auto const ev_croi_v     = (EventROI*)(mgr.get_data(kProductROI,_reco_roi_prod));
    if (!ev_croi_v) throw larbys("Invalid cROI producer provided");

    //Get pgraph vector
    EventPGraph* ev_pgraph = nullptr;
    if (!_pgraph_prod.empty()) {
      ev_pgraph = (EventPGraph*)(mgr.get_data(kProductPGraph,_pgraph_prod));
      if (!ev_pgraph) throw larbys("Invalid pgraph producer provided!");
      has_reco_vtx = true;
    }

    //Get countour vector
    EventPixel2D* ev_ctor_v = nullptr;
    if (!_pcluster_ctor_prod.empty()) {
      ev_ctor_v = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_ctor_prod));
      if (!ev_ctor_v) throw larbys("Invalid Contour Pixel2D producer provided!");
      if (!has_reco_vtx) throw larbys("Gave PGraph producer but no particle cluster?");
    }

    //Get cluster vector
    EventPixel2D* ev_pcluster_v = nullptr;
    if (!_pcluster_img_prod.empty()) {
      ev_pcluster_v = (EventPixel2D*)(mgr.get_data(kProductPixel2D,_pcluster_img_prod));
      if (!ev_pcluster_v) throw larbys("Invalid Particle Pixel2D producer provided!");
      if (!has_reco_vtx) throw larbys("Gave PGraph producer but no image cluster?");
    }

    //Convert pcluster vector to pcluster array
    const auto& pcluster_m = ev_pcluster_v->Pixel2DClusterArray();
    
    //Convert raw Image2D vector to Image2D Array
    const auto& adc_img2d_v = ev_img2d->Image2DArray();

    //Contruct thrumu and stopmu images
    std::vector<larcv::Image2D> thrumu_img2d_v;
    std::vector<larcv::Image2D> stopmu_img2d_v;
    _LArbysImageMaker.ConstructCosmicImage(mgr,_thrumu_img_prod,_tags_datatype,adc_img2d_v,thrumu_img2d_v);
    _LArbysImageMaker.ConstructCosmicImage(mgr,_stopmu_img_prod,_tags_datatype,adc_img2d_v,stopmu_img2d_v);

    //Create empty ROI vector with expected size of the cROI array, proceed to crop the
    //raw adc images to the ROIs
    std::vector<std::vector<ImageMeta> > roid_v;
    roid_v.reserve(ev_croi_v->ROIArray().size());

    for(auto const& croi : ev_croi_v->ROIArray()) {
      auto const& bb_v = croi.BB();
      roid_v.push_back(crop_metas(adc_img2d_v,bb_v));
    }
    
    auto num_verts = ev_pgraph->PGraphArray().size();
    _vtxid = -1;
    std::vector<size_t> plane_order_v = {2,0,1};
    for (int vtxID = 0; vtxID < num_verts; ++vtxID) {

      
      _num_allpix_v.clear();
      _num_allpix_v.resize(3,-1);
      _num_nottag_v.clear();
      _num_nottag_v.resize(3,-1);
      _num_thrutag_v.clear();
      _num_thrutag_v.resize(3,-1);
      _num_stoptag_v.clear();
      _num_stoptag_v.resize(3,-1);
      _pts_in_raw_clus_v.clear();
      _pts_in_raw_clus_v.resize(3,-1);
      _pts_stopmu_ovrlap_v.clear();
      _pts_stopmu_ovrlap_v.resize(3,-1);
      _pts_thrumu_ovrlap_v.clear();
      _pts_thrumu_ovrlap_v.resize(3,-1);
      _endCos_v.clear();
      _endCos_v.resize(3,-2);
      
      auto pgraph = ev_pgraph->PGraphArray().at(vtxID);
      auto const& roi_v = pgraph.ParticleArray();
      auto const& bb_v  = roi_v.front().BB();

      auto iter = std::find(roid_v.begin(),roid_v.end(),bb_v);
      if (iter == roid_v.end()) throw larbys("Unknown image meta");

      auto roid = iter - roid_v.begin();

      if (roid != _roid) _vtxid = 0;

      _roid  = roid;
      _vtxid += 1;
      _run    = ev_img2d->run();
      _subrun = ev_img2d->subrun();
      _event  = ev_img2d->event();
      _entry  = mgr.current_entry();


      auto const& cluster_idx_v = pgraph.ClusterIndexArray();
      if (cluster_idx_v.size() < 2) {
	_tree->Fill();
	continue;
      }
      auto const& cluster_idx0  = cluster_idx_v[0];
      auto const& cluster_idx1  = cluster_idx_v[1];
      
      
      for(auto const& plane : plane_order_v){
	auto crop_img      =  pgraph.ParticleArray().back().BB(plane);
	auto crop_mat_raw  = _LArbysImageMaker.ExtractMat(adc_img2d_v.at(plane).crop(crop_img));
	auto crop_mat_stop = _LArbysImageMaker.ExtractMat(stopmu_img2d_v.at(plane).crop(crop_img));
	auto crop_mat_thru = _LArbysImageMaker.ExtractMat(thrumu_img2d_v.at(plane).crop(crop_img));
	
	auto raw_non_zero    = cv::countNonZero(crop_mat_raw);
	auto stopmu_non_zero = cv::countNonZero(crop_mat_stop);
	auto thrumu_non_zero = cv::countNonZero(crop_mat_thru);

	_num_allpix_v.at(plane)  = raw_non_zero;
	_num_nottag_v.at(plane)  = raw_non_zero - thrumu_non_zero - stopmu_non_zero;//thrumu_non_zero + stopmu_non_zero - raw_non_zero;
	_num_stoptag_v.at(plane) = stopmu_non_zero;//raw_non_zero - stopmu_non_zero;
	_num_thrutag_v.at(plane) = thrumu_non_zero;//raw_non_zero - thrumu_non_zero;

	auto iter_pcluster = pcluster_m.find(plane);
	if(iter_pcluster == pcluster_m.end()) continue;
	auto const& pcluster_v = (*iter_pcluster).second;
	auto const& pcluster0  = pcluster_v.at(cluster_idx0);
	auto const& pcluster1  = pcluster_v.at(cluster_idx1);

	int nInClusterRaw   = pcluster0.size() + pcluster1.size();
	std::vector<cv::Point> pointsInRaw;

	larcv::Pixel2D startPt;
	if (pcluster0.size() > 0) {
	  startPt = pcluster0.at(0);
	}
	larcv::Pixel2D maxPt0;
	larcv::Pixel2D maxPt1;
	float maxDist0 = -1;
	float maxDist1 = -1;
	for (auto const& pt : pcluster0) {
	  pointsInRaw.push_back(cv::Point(pt.X(),pt.Y()));
	  float dist0 = std::sqrt( ((float)startPt.X() - (float)pt.X())*((float)startPt.X() - (float)pt.X()) + ((float)startPt.Y() - (float)pt.Y())*((float)startPt.Y() - (float)pt.Y())) ;
	  if (dist0 > maxDist0 ) {
	    maxDist0 = dist0;
	    maxPt0 = pt;
	  }
	}
	for (auto const& pt : pcluster1) {
	  pointsInRaw.push_back(cv::Point(pt.X(),pt.Y()));
	  float dist1 = std::sqrt( ((float)startPt.X() - (float)pt.X())*((float)startPt.X() - (float)pt.X()) + ((float)startPt.Y() - (float)pt.Y())*((float)startPt.Y() - (float)pt.Y()));
	  if ( dist1 > maxDist1 ) {
	    maxDist1 = dist1;
	    maxPt1 = pt;
	  }
	}

	if (pcluster0.size() > 0) {
	  float dot  = ((float)maxPt0.X()-(float)startPt.X())*((float)maxPt1.X()-(float)startPt.X()) + ((float)maxPt0.Y()-(float)startPt.Y())*((float)maxPt1.Y()-(float)startPt.Y());
	  float mag0 = std::sqrt(std::pow((float)maxPt0.X()-(float)startPt.X(),2)+std::pow((float)maxPt0.Y()-(float)startPt.Y(),2));
	  float mag1 = std::sqrt(std::pow((float)maxPt1.X()-(float)startPt.X(),2)+std::pow((float)maxPt1.Y()-(float)startPt.Y(),2));
	  _endCos_v.at(plane) = dot/(mag0*mag1);	
	}
	
	auto pointsInThru = larocv::FindNonZero(crop_mat_thru);
	auto pointsInStop = larocv::FindNonZero(crop_mat_stop);
	
	int nInThru = 0;
	int nInStop = 0;
	for (auto const& pt : pointsInRaw) {
	  if (std::find(pointsInThru.begin(),pointsInThru.end(),pt) != pointsInThru.end()) nInThru++;
	  if (std::find(pointsInStop.begin(),pointsInStop.end(),pt) != pointsInStop.end()) nInStop++;
	}

	_pts_thrumu_ovrlap_v.at(plane) = nInThru;
	_pts_stopmu_ovrlap_v.at(plane) = nInStop;
	_pts_in_raw_clus_v.at(plane)   = nInClusterRaw;

      }
      
      _tree->Fill();
    }

    return true;
  }

  void CosmicVertexAna::finalize()
  {
    if(has_ana_file()) {
      _tree->Write();
    }  
  }

}
#endif
