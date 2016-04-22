#ifndef __IMAGEMERGER_CXX__
#define __IMAGEMERGER_CXX__

#include "ImageMerger.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventChStatus.h"

namespace larcv {

  static ImageMergerProcessFactory __global_ImageMergerProcessFactory__;

  ImageMerger::ImageMerger(const std::string name)
    : ProcessBase(name)
    , _nu_proc       (nullptr)
    , _cosmic_proc   (nullptr)
    , _min_ch_status (0)
  {}
    
  void ImageMerger::configure(const PSet& cfg)
  {
    _pmt_pedestal         = cfg.get<float>       ("PMTPedestal"      );
    _min_ch_status        = cfg.get<short>       ("MinChannelStatus" );
    _out_tpc_producer     = cfg.get<std::string> ("OutNameTPCImage"  );
    _out_pmt_producer     = cfg.get<std::string> ("OutNamePMTImage"  );
    _out_roi_producer     = cfg.get<std::string> ("OutNameROI"       );
    _out_status_producer  = cfg.get<std::string> ("OutNameChStatus"  );
    _out_segment_producer = cfg.get<std::string> ("OutNameSegment"   );
  }

  void ImageMerger::initialize()
  {
  }

  void ImageMerger::NeutrinoImageHolder(ImageHolder* ptr)
  { _nu_proc = ptr; }

  void ImageMerger::CosmicImageHolder(ImageHolder* ptr)
  { _cosmic_proc = ptr; }

  bool ImageMerger::process(IOManager& mgr)
  {
    LARCV_INFO() << "Start merging"<<std::endl;
    if(!_nu_proc) {
      LARCV_CRITICAL() << "NeutrinoImageHolder() must be called to set ImageHolder pointer!" << std::endl;
      throw larbys();
    }

    if(!_cosmic_proc) {
      LARCV_CRITICAL() << "CosmicImageHolder() must be called to set ImageHolder pointer!" << std::endl;
      throw larbys();
    }

    mgr.set_id(_cosmic_proc->run(), _cosmic_proc->subrun(), _cosmic_proc->event());

    //
    // Retrieve data from data/mc image holders
    //
    std::vector<larcv::Image2D>  data_tpc_image_v;
    std::vector<larcv::Image2D>  data_tpc_segment_v;
    larcv::Image2D  data_pmt_image;
    std::map<larcv::PlaneID_t,larcv::ChStatus> data_status_m;

    std::vector<larcv::Image2D>  mc_tpc_image_v;
    std::vector<larcv::Image2D>  mc_tpc_segment_v;
    larcv::Image2D  mc_pmt_image;
    std::map<larcv::PlaneID_t,larcv::ChStatus> mc_status_m;
    std::vector<larcv::ROI> mc_roi_v;

    LARCV_INFO() << "Moving Cosmic images..." << std::endl;
    _cosmic_proc->move_tpc_image   ( data_tpc_image_v   );
    _cosmic_proc->move_tpc_segment ( data_tpc_segment_v );
    _cosmic_proc->move_pmt_image   ( data_pmt_image     );
    _cosmic_proc->move_ch_status   ( data_status_m      );

    LARCV_INFO() << "Moving Neutrino images..." << std::endl;
    _nu_proc->move_tpc_image   ( mc_tpc_image_v   );
    _nu_proc->move_tpc_segment ( mc_tpc_segment_v );
    _nu_proc->move_pmt_image   ( mc_pmt_image     );
    _nu_proc->move_ch_status   ( mc_status_m      );
    _nu_proc->move_roi         ( mc_roi_v         );

    //
    // Sanity checks
    //
    // Check size
    if(data_tpc_image_v.size() != mc_tpc_image_v.size()) {
      LARCV_ERROR() << "# of Data stream image do not match w/ MC stream! Skipping this entry..." << std::endl;
      return false;
    }
    if(mc_roi_v.empty()) {
      LARCV_ERROR() << "No MC Interaction ROI found. skipping..." << std::endl;
      return false;
    }
    /*
    if(mc_roi_v.size() != data_tpc_image_v.size()) {
      LARCV_ERROR() << "# of image do not match w/ # of ROI! Skipping this entry..." << std::endl;
      return true;
    }
    */
    
    // Check PlaneID
    for(size_t i=0; i<data_tpc_image_v.size(); ++i) {
      auto const& image1 = data_tpc_image_v[i];
      auto const& image2 = mc_tpc_image_v[i];
      if(image1.meta().plane() != image2.meta().plane()) {
	LARCV_ERROR() << "Plane ID mismatch! skipping..." << std::endl;
	return false;
      }
      if( (!data_status_m.empty() && (data_status_m.find(image1.meta().plane()) == data_status_m.end())) ||
	  (!mc_status_m.empty() && (mc_status_m.find(image1.meta().plane()) == mc_status_m.end())) ) {
	LARCV_ERROR() << "Plane ID " << image1.meta().plane() << " not found for ch status!" << std::endl;
	return false;
      }
    }
    // All check done

    //
    // Merge them
    //
    // Merge status
    std::set<larcv::PlaneID_t> status_plane_s;
    for( auto const& plane_status : data_status_m ) status_plane_s.insert(plane_status.first);
    for( auto const& plane_status : mc_status_m   ) status_plane_s.insert(plane_status.first);
    for( auto const& plane : status_plane_s) {

      auto data_iter = data_status_m.find(plane);
      auto mc_iter   = mc_status_m.find(plane);

      if(mc_iter == mc_status_m.end()) continue;
      if(data_iter == data_status_m.end()) {
	data_status_m[plane] = (*mc_iter).second;
	continue;
      }

      auto const& mc_status_v = (*mc_iter).second.as_vector();
      auto const& data_status_v = (*data_iter).second.as_vector();

      std::vector<short> status_v(std::max(mc_status_v.size(),data_status_v.size()),0);
      const size_t min_entry = std::min(mc_status_v.size(),data_status_v.size());

      for(size_t i=0; i<min_entry; ++i)
	status_v[i] = std::min(mc_status_v[i],data_status_v[i]);

      if(data_status_v.size() > min_entry)
	for(size_t i=min_entry; i<data_status_v.size(); ++i) status_v[i] = data_status_v[i];

      if(mc_status_v.size() > min_entry)
	for(size_t i=min_entry; i<mc_status_v.size(); ++i) status_v[i] = mc_status_v[i];
	
      (*data_iter).second = std::move(ChStatus(plane,std::move(status_v)));
      
    }

    // Merge tpc image
    for(size_t i=0; i<data_tpc_image_v.size(); ++i) {

      // retrieve & sum
      auto& mc_image   = mc_tpc_image_v[i];
      auto& data_image = data_tpc_image_v[i];
      data_image.overlay(mc_image,Image2D::kSum);
      

      auto const& meta = data_image.meta();
      std::vector<float> null_col(meta.rows(),0);
      // Impose ChStatus
      auto data_status_iter = data_status_m.find(meta.plane());
      if(data_status_iter != data_status_m.end()) {
	auto const& stat_v = (*data_status_iter).second.as_vector();
	for(size_t wire_num=0; wire_num < stat_v.size(); ++wire_num) {
	  if(wire_num < data_image.meta().min_x() || wire_num>= data_image.meta().max_x()) continue;
	  auto const& stat = stat_v[wire_num];
	  if(stat < _min_ch_status) {
	    auto col = meta.col((double)wire_num);
	    data_image.copy(0,col,null_col);
	  }
	}
      }
    }
    mc_tpc_image_v.clear();

    // Merge tpc segment
    std::vector<larcv::Image2D> out_segment_v;
    if(data_tpc_segment_v.empty()) {
      out_segment_v = std::move(mc_tpc_segment_v);
    }
    else {
      out_segment_v = std::move(data_tpc_segment_v);
      
      for(size_t i=0; i<out_segment_v.size(); ++i) {
	
	// retrieve & sum
	auto& data_segment = out_segment_v[i];
	
	if(mc_tpc_segment_v.size()>i) {
	  auto const& mc_segment   = mc_tpc_segment_v[i];
	  LARCV_WARNING() << data_segment.meta().dump() << std::endl << mc_segment.meta().dump() << std::endl;
	  data_segment.overlay(mc_segment,Image2D::kMaxPool);
	}
	
	auto const& meta = data_segment.meta();
	std::vector<float> null_col(meta.rows(),(float)kROIUnknown);
	// Impose ChStatus
	auto data_status_iter = data_status_m.find(meta.plane());
	if(data_status_iter != data_status_m.end()) {
	  auto const& stat_v = (*data_status_iter).second.as_vector();
	  for(size_t wire_num=0; wire_num < stat_v.size(); ++wire_num) {
	    
	    if(wire_num < data_segment.meta().min_x() || wire_num>= data_segment.meta().max_x()) continue;
	    
	    auto const& stat = stat_v[wire_num];
	    if(stat < _min_ch_status) {
	      auto col = meta.col((double)wire_num);
	      data_segment.copy(0,col,null_col);
	    }
	  }
	}
      }
    }

    // Merge pmt image
    data_pmt_image.overlay(mc_pmt_image);
    data_pmt_image -= _pmt_pedestal;

    // Store
    auto out_tpc_image   = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_tpc_producer));
    out_tpc_image->Emplace(std::move(data_tpc_image_v));
    
    auto out_pmt_image   = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_pmt_producer));
    out_pmt_image->Emplace(std::move(data_pmt_image));

    auto out_tpc_segment = (EventImage2D*)(mgr.get_data(kProductImage2D,_out_segment_producer));
    out_tpc_segment->Emplace(std::move(out_segment_v));

    auto out_status = (EventChStatus*)(mgr.get_data(kProductChStatus,_out_status_producer));
    for(auto& plane_status : data_status_m) out_status->Emplace(std::move(plane_status.second));
    
    auto out_roi = (EventROI*)(mgr.get_data(kProductROI,_out_roi_producer));
    out_roi->Emplace(std::move(mc_roi_v));

    LARCV_INFO() << "End merging"<<std::endl;
    return true;
  }

  void ImageMerger::finalize(TFile* ana_file)
  {}

}
#endif
