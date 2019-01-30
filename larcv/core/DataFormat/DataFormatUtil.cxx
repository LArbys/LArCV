#ifndef __DATAFORMAT_DATAFORMATUTIL_CXX__
#define __DATAFORMAT_DATAFORMATUTIL_CXX__

#include "DataFormatUtil.h"

namespace larcv {

  ROIType_t PdgCode2ROIType(int pdgcode) 
  {

    if(pdgcode == 11 || pdgcode == -11) return kROIEminus;
    if(pdgcode == 321 || pdgcode == -321) return kROIKminus;
    if(pdgcode == 2212) return kROIProton;
    if(pdgcode == 13 || pdgcode == -13) return kROIMuminus;
    if(pdgcode == 211 || pdgcode == -211) return kROIPiminus;
    if(pdgcode == 22) return kROIGamma;
    if(pdgcode == 111) return kROIPizero;
    return kROIUnknown;

  }
}

#endif
