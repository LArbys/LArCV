#ifndef EVENTPIXEL2D_CXX
#define EVENTPIXEL2D_CXX

#include "EventPixel2D.h"
#include "Base/larcv_logger.h"
#include "Base/larbys.h"

namespace larcv {

  void EventPixel2D::clear()
  {
    _pixel_m.clear(); _cluster_m.clear();
  }

  const std::vector<larcv::Pixel2D>& EventPixel2D::Pixel2DArray(const ::larcv::PlaneID_t plane)
  {
    auto iter = _pixel_m.find(plane);
    if(iter == _pixel_m.end()) {
      logger::get("EventPixel2D").send(msg::kCRITICAL, __FUNCTION__, __LINE__)
	<< "No pixel for plane " << plane << std::endl;
      throw larbys();
    }

    return (*iter).second;
  }

  const std::vector<larcv::Pixel2DCluster>& EventPixel2D::Pixel2DClusterArray(const ::larcv::PlaneID_t plane)
  {
    auto iter = _cluster_m.find(plane);
    if(iter == _cluster_m.end()) {
      logger::get("EventPixel2DCluster").send(msg::kCRITICAL, __FUNCTION__, __LINE__)
	<< "No cluster for plane " << plane << std::endl;
      throw larbys();
    }

    return (*iter).second;
  }

  void EventPixel2D::Append(const larcv::PlaneID_t plane, const Pixel2D& pixel)
  {
    _pixel_m[plane].push_back(pixel);
  }

  void EventPixel2D::Append(const larcv::PlaneID_t plane, const Pixel2DCluster& cluster)
  {
    auto& col = _cluster_m[plane];
    col.push_back(cluster);
    col.back()._id = col.size() - 1;
  }
    
  void EventPixel2D::Emplace(const larcv::PlaneID_t plane, Pixel2D&& pixel)
  {
    auto& col = _pixel_m[plane];
    col.emplace_back(std::move(pixel));
  }

  void EventPixel2D::Emplace(const larcv::PlaneID_t plane, Pixel2DCluster&& cluster)
  {
    auto& col = _cluster_m[plane];
    col.emplace_back(std::move(cluster));
    col.back()._id = col.size() - 1;
  }

}

#endif