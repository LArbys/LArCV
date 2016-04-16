#ifndef __HIRESIMAGEDIVIDER_CXX__
#define __HIRESIMAGEDIVIDER_CXX__

#include "HiResImageDivider.h"

namespace larcv {
  namespace hires {
    static HiResImageDividerProcessFactory __global_HiResImageDividerProcessFactory__;
    
    HiResImageDivider::HiResImageDivider(const std::string name)
      : ProcessBase(name)
    {}
    
    void HiResImageDivider::configure(const PSet& cfg)
    {
      fGeoFile = cfg.get<std::string>("GeoFile");
    }
    
    void HiResImageDivider::initialize()
    {
      m_WireInfo = new ::larcv::pmtweights::PMTWireWeights( fGeoFile, 3456 );
      
    }
    
    bool HiResImageDivider::process(IOManager& mgr)
    {
      
      
    }
    
    void HiResImageDivider::finalize(TFile* ana_file)
    {}
    
    float HiResImageDivider::cross2D( float a[], float b[] ) {
      return a[0]*b[1] - a[1]*b[0];
    }
  }
}
#endif
