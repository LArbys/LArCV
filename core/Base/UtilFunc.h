#ifndef CVFHICL_UTILFUNC_H
#define CVFHICL_UTILFUNC_H

#include "PSet.h"

namespace larcv {

  std::string ConfigFile2String(std::string fname);

  PSet CreatePSetFromFile(std::string fname,std::string cfg_name="cfg");

}

#endif
