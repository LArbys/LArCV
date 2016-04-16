#ifndef __DATAFORMAT_UTILFUNC_CXX__
#define __DATAFORMAT_UTILFUNC_CXX__

#include "UtilFunc.h"
namespace larcv {

  ROIType_t PDG2ROIType(const int pdgcode)
  {
    switch(pdgcode) {
      
      // electron
    case  11:
    case -11:
      return kROIEminus;
      // muon
    case  13:
    case -13:
      return kROIMuminus;
      // gamma
    case 22:
      return kROIGamma;
      // neutrinos
    case 12:
    case 14:
      return kROIBNB;
    // proton
    case  2212:
    case -2212:
      return kROIProton;
      // pi0
    case  111:
      return kROIPizero;
	// pi +/-
    case  211:
    case -211:
      return kROIPiminus;
      // K +/-
    case  321:
    case -321:
      return kROIKminus;
    default:
      return kROIUnknown;
    }
    return kROIUnknown;
  }
}

#endif
