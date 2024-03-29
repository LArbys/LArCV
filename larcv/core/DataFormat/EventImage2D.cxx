#ifndef EVENTIMAGE2D_CXX
#define EVENTIMAGE2D_CXX

#include <sstream>
#include "EventImage2D.h"
#include "larcv/core/Base/larbys.h"

#include <sstream>

namespace larcv {

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventImage2DFactory __global_EventImage2DFactory__;

  void EventImage2D::clear()
  {
    EventBase::clear();
    _image_v.clear();
  }

  const Image2D& EventImage2D::at(ImageIndex_t id) const
  {
    if( id >= _image_v.size() ) {
      std::stringstream msg;
      msg << "EventImage2D.cxx" << ".L" << __LINE__ << ":"
	  << "Invalid request (ImageIndex_t[" << id << "] out-o-range [>=" << _image_v.size() << "] )!"
	  << std::endl;
      throw larbys(msg.str());
    }
    return _image_v[id];
  }

  Image2D& EventImage2D::modimgat(ImageIndex_t id)
  {
    if( id >= _image_v.size() ) {
      std::stringstream ss;
      ss << __FILE__"." << __LINE__ << "Invalid request [" << id << "] (ImageIndex_t out-o-range, max=" << _image_v.size() << ")!";
      throw larbys(ss.str());
    }
    return _image_v[id];
  }
  
  void EventImage2D::Append(const Image2D& img)
  {
    _image_v.push_back(img);
    _image_v.back().index((ImageIndex_t)(_image_v.size()-1));
  }

  void EventImage2D::Emplace(Image2D&& img)
  {
    _image_v.emplace_back(img);
    _image_v.back().index((ImageIndex_t)(_image_v.size()-1));
  }

  void EventImage2D::Emplace(std::vector<larcv::Image2D>&& image_v)
  {
    _image_v = std::move(image_v);
    for(size_t i=0; i<_image_v.size(); ++i) _image_v[i].index((ImageIndex_t)i);
  }
}

#endif
