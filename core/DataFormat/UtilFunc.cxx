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
    case kROIUnknown: return "Unknown";//0
    case kROICosmic:  return "Cosmic"; //1
    case kROIBNB:     return "BNB";    //2
    case kROIEminus:  return "Eminus"; //3
    case kROIGamma:   return "Gamms";  //4
    case kROIPizero:  return "Pizero"; //5
    case kROIMuminus: return "Muminus";//6
    case kROIKminus:  return "Kminus"; //7
    case kROIPiminus: return "Piminus";//8
    case kROIProton:  return "Proton"; //9
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
