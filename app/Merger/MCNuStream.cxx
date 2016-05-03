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

    LARCV_INFO() << "Configured..." << std::endl;
  }

  void MCNuStream::initialize()
  {
    LARCV_INFO() << "Initialized..." << std::endl;
  }

  bool MCNuStream::process(IOManager& mgr)
  {
    LARCV_INFO() << "Clearing attributes..." << std::endl;
    _tpc_image_v.clear();
    _tpc_segment_v.clear();
    _roi_v.clear();

    // Retrieve ROI that match our condition
    LARCV_INFO() << "Reading in ROI " << _roi_producer << std::endl;
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

    LARCV_INFO() << "Copying ROIs.." <<std::endl;
    _roi_v = event_roi->ROIArray();

    // Retrieve TPC image
    LARCV_INFO() << "Reading in TPC Image2D " << _tpc_image_producer << std::endl;
    auto event_tpc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_tpc_image_producer));
    
    if(!event_tpc_image || event_tpc_image->Image2DArray().empty()) return false;

    LARCV_INFO() << "Copying TPC Image2D " << _tpc_image_producer << std::endl;
    event_tpc_image->Move(_tpc_image_v);

    // Retrieve PMT image
    LARCV_INFO() << "Reading in PMT Image2D " << _pmt_image_producer << std::endl;
    auto event_pmt_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_pmt_image_producer));
    if(!event_pmt_image || event_pmt_image->Image2DArray().empty()) return false;

    LARCV_INFO() << "Copying in PMT Image2D " << _pmt_image_producer << std::endl;
    std::vector<larcv::Image2D> tmp_v;
    event_pmt_image->Move(tmp_v);
    if(tmp_v.size())
      _pmt_image = std::move(tmp_v[0]);
    
    // Retrieve TPC segment
    LARCV_INFO() << "Reading in Segment Image2D " << _segment_producer << std::endl;
    auto event_tpc_segment = (EventImage2D*)(mgr.get_data(kProductImage2D,_segment_producer));

    LARCV_INFO() << "Copying in Segment Image2D " << _segment_producer << std::endl;    
    event_tpc_segment->Move(_tpc_segment_v);

    retrieve_id(event_tpc_image);

    return true;
    
  }

  void MCNuStream::finalize()
  {}

}
#endif
