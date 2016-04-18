#ifndef __MCNUSTREAM_CXX__
#define __MCNUSTREAM_CXX__

#include "MCNuStream.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
namespace larcv {

  static MCNuStreamProcessFactory __global_MCNuStreamProcessFactory__;

  MCNuStream::MCNuStream(const std::string name)
    : ImageHolder(name)
  {}
    
  void MCNuStream::configure(const PSet& cfg)
  {
    _tpc_image_producer = cfg.get<std::string> ( "TPCImageProducer" );
    _pmt_image_producer = cfg.get<std::string> ( "PMTImageProducer" );
    _segment_producer   = cfg.get<std::string> ( "SegmentProducer"  );
    _roi_producer       = cfg.get<std::string> ( "ROIProducer"      );

    _min_energy_deposit = cfg.get<double>      ( "MinEnergyDeposit" );
    _min_energy_init = cfg.get<double>( "MinEnergyInit" );
    _min_width       = cfg.get<double>( "MinWidth"      );
    _min_height      = cfg.get<double>( "MinHeight"     );
  }

  void MCNuStream::initialize()
  {}

  bool MCNuStream::process(IOManager& mgr)
  {
    _tpc_image_v.clear();
    _tpc_segment_v.clear();
    _roi_v.clear();
    
    // Retrieve ROI that match our condition
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
      break;
    }
    if(!found) return false;

    _roi_v = event_roi->ROIArray();

    // Retrieve TPC image
    auto event_tpc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_tpc_image_producer));

    if(!event_tpc_image || event_tpc_image->Image2DArray().empty()) return false;

    _tpc_image_v = event_tpc_image->Image2DArray();

    // Retrieve PMT image
    auto event_pmt_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_pmt_image_producer));

    if(!event_pmt_image || event_pmt_image->Image2DArray().empty()) return false;

    _pmt_image = event_pmt_image->Image2DArray().front();

    // Retrieve TPC segment
    auto event_tpc_segment = (EventImage2D*)(mgr.get_data(kProductImage2D,_segment_producer));

    _tpc_segment_v = event_tpc_segment->Image2DArray();
    
    retrieve_id(event_tpc_image);

    return true;
    
  }

  void MCNuStream::finalize(TFile* ana_file)
  {}

}
#endif
