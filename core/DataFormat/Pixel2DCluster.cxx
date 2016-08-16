#ifndef PIXEL2DCLUSTER_CXX
#define PIXEL2DCLUSTER_CXX

#include "Pixel2DCluster.h"
#include <map>

namespace larcv {

  void Pixel2DCluster::Pool(const PoolType_t type)
  {
    std::vector<larcv::Pixel2D> res;
    res.reserve(this->size());

    std::vector<float> unique_count;
    unique_count.reserve(this->size());
    
    std::map<larcv::Pixel2D,size_t> unique_map;
    for(auto const& px : *this) {
      auto iter = unique_map.find(px);
      if(iter == unique_map.end()) {
	unique_map[px] = res.size();
	unique_count.push_back(1.);
	res.push_back(px);
      }else{
	switch(type) {
	case kPoolMax:
	  res[(*iter).second].Intensity(std::max( res[(*iter).second].Intensity() , px.Intensity() ));
	  break;
	case kPoolSum:
	case kPoolAverage:
	  res[(*iter).second].Intensity( res[(*iter).second].Intensity() + px.Intensity() );
	  unique_count[(*iter).second] += 1.0;
	  break;
	}
      }
    }
    if(type == kPoolAverage) {

      for(size_t i=0; i<res.size(); ++i)

	res[i].Intensity( res[i].Intensity() / unique_count[i] );
    }

    (*this) = Pixel2DCluster(std::move(res));
  }


}

#endif
