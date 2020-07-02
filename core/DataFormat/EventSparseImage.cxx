#ifndef EVENT_SPARSE_IMAGE_CXX
#define EVENT_SPARSE_IMAGE_CXX

#include "EventSparseImage.h"
#include "Base/larbys.h"

namespace larcv {

  /// Global larcv::SBClusterFactory to register ClusterAlgoFactory
  static EventSparseImageFactory __global_EventSparseImageFactory__;

  void EventSparseImage::clear()
  {
    EventBase::clear();
    _image_v.clear();
  }

  const SparseImage& EventSparseImage::at(ImageIndex_t id) const
  {
    if( id >= _image_v.size() ) throw larbys("Invalid request (ImageIndex_t out-o-range)!");
    return _image_v[id];
  }

  void EventSparseImage::Append(const SparseImage& img)
  {
    _image_v.push_back(img);
    _image_v.back().index((ImageIndex_t)(_image_v.size()-1));
  }

  void EventSparseImage::Emplace(SparseImage&& img)
  {
    _image_v.emplace_back(img);
    _image_v.back().index((ImageIndex_t)(_image_v.size()-1));
  }

  void EventSparseImage::Emplace(std::vector<larcv::SparseImage>&& image_v)
  {
    _image_v = std::move(image_v);
    for(size_t i=0; i<_image_v.size(); ++i) _image_v[i].index((ImageIndex_t)i);
  }
}

#endif
