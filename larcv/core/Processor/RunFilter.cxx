#ifndef __RUNFILTER_CXX__
#define __RUNFILTER_CXX__

#include "RunFilter.h"

namespace larcv {

  static RunFilterProcessFactory __global_RunFilterProcessFactory__;

  RunFilter::RunFilter(const std::string name)
    : ProcessBase(name)
  {}
    
  void RunFilter::configure(const PSet& cfg)
  {
    _producer   = cfg.get<std::string>("Producer");
    _type       = (ProductType_t)(cfg.get<unsigned short>("ProductType"));
    auto run_v = cfg.get<std::vector<size_t> >("Exclude");
    for(auto const& v : run_v) {
      if(_run_s.find(v) != _run_s.end()) {
	LARCV_WARNING() << "Run " << v << " appears more than once in the list..." << std::endl;
	continue;
      }
      _run_s.insert(v);
    }
  }

  void RunFilter::initialize()
  {}

  bool RunFilter::process(IOManager& mgr)
  {
    auto ptr = mgr.get_data(_type,_producer);
    return (_run_s.find(ptr->run()) == _run_s.end());
  }

  void RunFilter::finalize()
  { return; }

}
#endif
