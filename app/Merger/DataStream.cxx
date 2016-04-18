#ifndef __DATASTREAM_CXX__
#define __DATASTREAM_CXX__

#include "DataStream.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventChStatus.h"

namespace larcv {

  static DataStreamProcessFactory __global_DataStreamProcessFactory__;

  DataStream::DataStream(const std::string name)
    : ImageHolder(name)
  {}
    
  void DataStream::configure(const PSet& cfg)
  {
    _tpc_image_producer = cfg.get<std::string>("TPCImageProducer");
    _pmt_image_producer = cfg.get<std::string>("PMTImageProducer");
    _ch_status_producer = cfg.get<std::string>("ChStatusProducer");
    _adc_threshold = cfg.get<float>("ADCThreshold");
  }

  void DataStream::initialize()
  {}

  bool DataStream::process(IOManager& mgr)
  {
    _ch_status_m.clear();
    _tpc_image_v.clear();
    _tpc_segment_v.clear();
    _pmt_image = Image2D();
    
    // Retrieve ChStatus    
    auto event_chstatus = (EventChStatus*)(mgr.get_data(kProductChStatus,_ch_status_producer));
    
    if(!event_chstatus || event_chstatus->ChStatusMap().empty()) return false;

    _ch_status_m = event_chstatus->ChStatusMap();

    // Retrieve TPC Image
    auto event_tpc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_tpc_image_producer));

    if(!event_tpc_image || event_tpc_image->Image2DArray().empty()) return false;

    for(auto const& img : event_tpc_image->Image2DArray())
      
      _tpc_image_v.push_back(img);

    // Retrieve PMT Image
    auto event_pmt_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_pmt_image_producer));

    if(!event_pmt_image || event_pmt_image->Image2DArray().empty()) return false;

    _pmt_image = event_pmt_image->Image2DArray().front();

    // Create segmentation map
    for(auto const& img : _tpc_image_v) {

      auto copy_img  = img;

      copy_img.binary_threshold(_adc_threshold,(float)kROIUnknown,(float)kROICosmic);

      _tpc_segment_v.emplace_back(img);

    }

    // Retrieve event id
    retrieve_id(event_tpc_image);

    return true;
  }

  void DataStream::finalize(TFile* ana_file)
  {}

}
#endif
