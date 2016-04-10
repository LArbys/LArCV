#ifndef CHSTATUS_CXX
#define CHSTATUS_CXX

#include "ChStatus.h"
#include "Base/larbys.h"

namespace larcv {

  void  ChStatus::Reset(size_t nwires, short init_status)
  {
    _status_v.resize(nwires);
    for(auto& s : _status_v) s = init_status;
  }
  
  void  ChStatus::Status(size_t wire, short status)
  {
    if(_status_v.size() >= wire) throw larbys("Invalid wire requested!");
    _status_v[wire] = status;
  }
  
  short ChStatus::Status(size_t wire) const
  {
    if(_status_v.size() >= wire) throw larbys("Invalid wire requested!");
    return _status_v[wire];
  }


}

#endif
