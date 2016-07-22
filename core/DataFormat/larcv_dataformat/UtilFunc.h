#ifndef __DATAFORMAT_UTILFUNC_H__
#define __DATAFORMAT_UTILFUNC_H__

#include "DataFormatTypes.h"
#include <sstream>
namespace larcv {

  ROIType_t PDG2ROIType(const int pdgcode);

  std::string ROIType2String(const ROIType_t type);

  ROIType_t String2ROIType(const std::string& name);
  
}
#endif
