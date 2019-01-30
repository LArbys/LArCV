#ifndef THREADFILLERFACTORY_H
#define THREADFILLERFACTORY_H
#include "ThreadDatumFiller.h"

namespace larcv {
  class ThreadFillerFactory {
    
  private:
    ThreadFillerFactory() {}

  public:
    ~ThreadFillerFactory(){ for(auto& name_ptr : _filler_m) delete name_ptr.second; _filler_m.clear(); }

    static bool exist_filler(const std::string& name) {
      auto iter = _filler_m.find(name);
      return (iter != _filler_m.end() && (*iter).second);
    }

    static void destroy_filler(const std::string& name) {
      if(!exist_filler(name)) return;
      if(_filler_m[name]) {
	delete _filler_m[name];
	_filler_m[name] = nullptr;
      }
    }
    
    static ThreadDatumFiller& get_filler(const std::string& name) {
      auto iter = _filler_m.find(name);
      if(iter != _filler_m.end() && (*iter).second) return (*((*iter).second));
      auto ptr = new ::larcv::ThreadDatumFiller(name);
      _filler_m[name]=ptr;
      return (*ptr);
    }

  private:
    
    static std::map<std::string,larcv::ThreadDatumFiller*> _filler_m;
  };
}

#endif
