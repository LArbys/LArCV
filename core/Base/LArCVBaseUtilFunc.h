#ifndef __LARCVBASE_UTILFUNC_H__
#define __LARCVBASE_UTILFUNC_H__

#include "PSet.h"

namespace larcv {

  std::string ConfigFile2String(std::string fname);

  PSet CreatePSetFromFile(std::string fname,std::string cfg_name="cfg");

}

#endif
