#ifndef __MCINFORETRIEVER_CXX__
#define __MCINFORETRIEVER_CXX__

#include "MCinfoRetriever.h"

namespace larcv {

  static MCinfoRetrieverProcessFactory __global_MCinfoRetrieverProcessFactory__;

  MCinfoRetriever::MCinfoRetriever(const std::string name)
    : ProcessBase(name)
  {}
    
  void MCinfoRetriever::configure(const PSet& cfg)
  {}

  void MCinfoRetriever::initialize()
  {}

  bool MCinfoRetriever::process(IOManager& mgr)
  {

    // get the ROI data that has the MC information
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/EventROI.h
    //https://github.com/LArbys/LArCV/blob/develop/core/DataFormat/ROI.h
    auto ev_roi = mgr.get_data(kProductROI,"tpc");
    std::cout << "This event is: " << ev_roi->event() << std::endl;

    return true;
    
  }

  void MCinfoRetriever::finalize()
  {}

}
#endif
