#ifndef __ROIMASK_CXX__
#define __ROIMASK_CXX__

#include "ROIMask.h"

namespace larcv {

  static ROIMaskProcessFactory __global_ROIMaskProcessFactory__;

  ROIMask::ROIMask(const std::string name)
    : ProcessBase(name)
  {}
    
  void ROIMask::configure(const PSet& cfg)
  {
    fInputImageProducer  = cfg.get<std::string>( "InputImageProducer" );
    fOutputImageProducer = cfg.get<std::string>( "OutputImageProducer" );
    fInputROIProducer    = cfg.get<std::string>( "InputROIProducer" );
    fMaskOutsideROI      = cfg.get<bool>( "MaskOutsideROI" );

    if (fInputImageProducer==fOutputImageProducer) inplace = true;
    else inplace = false;
      
  }

  void ROIMask::initialize()
  {}

  bool ROIMask::process(IOManager& mgr)
  {}

  void ROIMask::finalize()
  {}

}
#endif
