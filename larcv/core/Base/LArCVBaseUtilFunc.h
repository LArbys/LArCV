#ifndef __LARCVBASE_UTILFUNC_H__
#define __LARCVBASE_UTILFUNC_H__

#include "PSet.h"

namespace larcv {

  /// Given a configuration string, format to create larcv::PSet
  //std::string FormatPSetString(std::string fname);
  /// Given a configuration file (full path), read & parse contents to create larcv::PSet
  std::string ConfigFile2String(std::string fname);
  /// Given a configuration file (full path), create and return larcv::PSet
  PSet CreatePSetFromFile(std::string fname,std::string cfg_name="cfg");

}

#endif
