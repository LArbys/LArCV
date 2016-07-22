#ifndef __DATAFORMAT_UTILFUNC_CXX__
#define __DATAFORMAT_UTILFUNC_CXX__

#include "UtilFunc.h"
#include "Base/larbys.h"
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

  std::string ROIType2String(const ROIType_t type) 
  {
    switch(type) {
    case kROIUnknown: return "Unknown";
    case kROICosmic:  return "Cosmic";
    case kROIBNB:     return "BNB";
    case kROIEminus:  return "Eminus";
    case kROIGamma:   return "Gamms";
    case kROIPizero:  return "Pizero";
    case kROIMuminus: return "Muminus";
    case kROIKminus:  return "Kminus";
    case kROIPiminus: return "Piminus";
    case kROIProton:  return "Proton";
    default:
      std::stringstream ss;
      ss << "Unsupported type: " << type << std::endl;
      throw larbys(ss.str());
    }
    return "";
  }

  ROIType_t String2ROIType(const std::string& name)
  {
    if(name == "Unknown") return kROIUnknown;
    if(name == "Cosmic" ) return kROICosmic;
    if(name == "BNB"    ) return kROIBNB;
    if(name == "Eminus" ) return kROIEminus;
    if(name == "Gamma"  ) return kROIGamma;
    if(name == "Pizero" ) return kROIPizero;
    if(name == "Muminus") return kROIMuminus;
    if(name == "Kminus" ) return kROIKminus;
    if(name == "Piminus") return kROIPiminus;
    if(name == "Proton" ) return kROIProton;
    
    std::stringstream ss;
    ss << "Unsupported name: " << name << std::endl;
    throw larbys(ss.str());

    return kROIUnknown;
  }
}

#endif
