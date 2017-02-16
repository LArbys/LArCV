#ifndef __CROPROI_CXX__
#define __CROPROI_CXX__

#include "CropROI.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventImage2D.h"
namespace larcv {

  static CropROIProcessFactory __global_CropROIProcessFactory__;

  CropROI::CropROI(const std::string name)
    : ProcessBase(name)
  {}
    
  void CropROI::configure(const PSet& cfg)
  {
    _roi_producer = cfg.get<std::string>("ROIProducer");
    _input_producer = cfg.get<std::string>("InputProducer");
    _output_producer = cfg.get<std::string>("OutputProducer");
    _image_idx = cfg.get<std::vector<size_t> >("ImageIndex");
  }

  void CropROI::initialize()
  {}

  bool CropROI::process(IOManager& mgr)
  {
    auto input_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_input_producer));
    if(!input_image) {
      LARCV_CRITICAL() << "No Image2D found with a name: " << _input_producer << std::endl;
      throw larbys();
    }

    auto output_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_output_producer));
    if(!output_image) {
      LARCV_CRITICAL() << "No Image2D found with a name: " << _output_producer << std::endl;
      throw larbys();
    }

    auto event_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));
    if(!event_roi) {
      LARCV_CRITICAL() << "No ROI found with a name: " << _roi_producer << std::endl;
      throw larbys();
    }

    for(auto const& idx : _image_idx) {
      if(idx >= input_image->Image2DArray().size()) {
	LARCV_CRITICAL() << "ImageIndex array contains index " << idx
			 << " not available in Image2DArray (" << input_image->Image2DArray().size()
			 << ")!" << std::endl;
	throw larbys();
      }
    }
    
    auto const& roi_v = event_roi->ROIArray();

    if(roi_v.size() != 1) {
      LARCV_CRITICAL() << "More than 1 ROI (not supported)!" << std::endl;
      throw larbys();
    }

    std::vector<larcv::Image2D> image_v;
    if(_input_producer == _output_producer) {
      std::vector<larcv::Image2D> tmp_v;
      input_image->Move(tmp_v);
      for(auto const& idx : _image_idx)
	image_v.emplace_back(std::move(tmp_v[idx]));
    }else{
      auto const& tmp_v = input_image->Image2DArray();
      for(auto const& idx : _image_idx)
	image_v.push_back(tmp_v[idx]);
    }
    
    auto const& bb_v = roi_v[0].BB();
    if(bb_v.size() < _image_idx.size()) {
      LARCV_CRITICAL() << "Not enough bounding box!" << std::endl;
      throw larbys();
    }

    // Now process
    for(size_t idx=0; idx<_image_idx.size(); ++idx) {
      auto const& bb = bb_v[idx];
      image_v[idx] = image_v[idx].crop(bb);
    }
    output_image->Emplace(std::move(image_v));
    return true;
  }

  void CropROI::finalize()
  {}

}
#endif
