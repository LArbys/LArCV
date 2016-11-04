#ifndef __LARBYSIMAGEANA_CXX__
#define __LARBYSIMAGEANA_CXX__

#include "LArbysImageAna.h"

namespace larcv {

  static LArbysImageAnaProcessFactory __global_LArbysImageAnaProcessFactory__;

  LArbysImageAna::LArbysImageAna(const std::string name)
    : ProcessBase(name)
  {}
    
  void LArbysImageAna::configure(const PSet& cfg)
  {}

  void LArbysImageAna::initialize()
  {}

  bool LArbysImageAna::process(IOManager& mgr)
  {}

  void LArbysImageAna::finalize()
  {}

}
#endif
