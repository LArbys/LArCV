#ifndef __BNBNUSTREAM_CXX__
#define __BNBNUSTREAM_CXX__

#include "BNBNuStream.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
namespace larcv {

  static BNBNuStreamProcessFactory __global_BNBNuStreamProcessFactory__;

  BNBNuStream::BNBNuStream(const std::string name)
    : ImageHolder(name)
  {}
    
  void BNBNuStream::configure(const PSet& cfg)
  {
    _image_producer     = cfg.get<std::string> ( "ImageProducer"    );
    _roi_producer       = cfg.get<std::string> ( "ROIProducer"      );
    _min_energy_deposit = cfg.get<double>      ( "MinEnergyDeposit" );

    _min_energy_init = cfg.get<double>( "MinEnergyInit" );
    _min_width       = cfg.get<double>( "MinWidth"      );
    _min_height      = cfg.get<double>( "MinHeight"     );
  }

  void BNBNuStream::initialize()
  {}

  bool BNBNuStream::process(IOManager& mgr)
  {
    _image_v.clear();
    _roi = ROI();

    auto event_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));

    bool found=false;
    for(auto const& roi : event_roi->ROIArray()) {

      if(roi.Type() != kROIBNB) continue;

      if( roi.EnergyDeposit() < _min_energy_deposit ) return false;
      if( roi.EnergyInit()    < _min_energy_init    ) return false;
      double min_width  = 1e12;
      double min_height = 1e12;
      for(auto const& bb : roi.BB()) {
	if( bb.width()  < min_width  ) min_width  = bb.width();
	if( bb.height() < min_height ) min_height = bb.height();
      }
      if( min_width  < _min_width  ) return false;
      if( min_height < _min_height ) return false;

      found=true;
      _roi = roi;
    }
    if(!found) return false;

    auto event_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_image_producer));

    if(!event_image || event_image->Image2DArray().empty()) return false;

    for(auto const& img : event_image->Image2DArray())
      
      _image_v.push_back(img);

    retrieve_id(event_image);

    return true;
    
  }

  void BNBNuStream::finalize(TFile* ana_file)
  {}

}
#endif
