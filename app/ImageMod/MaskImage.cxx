#ifndef __MASKIMAGE_CXX__
#define __MASKIMAGE_CXX__

#include "MaskImage.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  static MaskImageProcessFactory __global_MaskImageProcessFactory__;

  MaskImage::MaskImage(const std::string name)
    : ProcessBase(name)
  {}
    
  void MaskImage::configure(const PSet& cfg)
  {

    _pi_thresh_min = cfg.get<float>("MinPIThreshold");
    _mask_value = cfg.get<float>("MaskValue");

    _reference_image_producer = cfg.get<std::string>("ReferenceProducer");
    _target_image_producer = cfg.get<std::string>("TargetProducer");

  }

  void MaskImage::initialize()
  {}

  bool MaskImage::process(IOManager& mgr)
  {
    auto ref_event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_reference_image_producer));
    if(!ref_event_image) {
      LARCV_CRITICAL() << "Reference EventImage2D not found: " << _reference_image_producer << std::endl;
      throw larbys();
    }
	
    auto tar_event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_target_image_producer));
    if(!tar_event_image) {
      LARCV_CRITICAL() << "Target EventImage2D not found: " << _target_image_producer << std::endl;
      throw larbys();
    }

    auto const& ref_image_v = ref_event_image->Image2DArray();

    std::vector<larcv::Image2D> tar_image_v;
    tar_event_image->Move(tar_image_v);

    // Check # planes
    if(ref_image_v.size() != tar_image_v.size()) {
      LARCV_CRITICAL() << "# planes in target (" << tar_image_v.size()
		       << ") and reference (" << ref_image_v.size()
		       << ") are not same!" << std::endl;
      throw larbys();
    }

    for(size_t pid=0; pid<tar_image_v.size(); ++pid) {

      auto& tar_image = tar_image_v[pid];

      auto const& ref_image = ref_image_v[pid].as_vector();

      if(tar_image.as_vector().size() != ref_image.size()) {
	LARCV_CRITICAL() << "Different size among the target (" << tar_image.as_vector().size()
			 << ") and reference (" << ref_image.size()
			 << ")!" << std::endl;
	throw larbys();
      }

      for(size_t px_idx = 0; px_idx < ref_image.size(); ++px_idx)

	if(ref_image[px_idx] < _pi_thresh_min) tar_image.set_pixel(px_idx,_mask_value);

    }

    tar_event_image->Emplace(std::move(tar_image_v));
    return true;
  }

  void MaskImage::finalize()
  {}

}
#endif
