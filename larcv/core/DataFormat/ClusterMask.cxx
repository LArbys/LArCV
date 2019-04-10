#ifndef LARCV_CLUSTERMASK_CXX
#define LARCV_CLUSTERMASK_CXX

#include "ClusterMask.h"

namespace larcv {

  ClusterMask::ClusterMask()
  : box(0,0,0,0,kINVALID_PROJECTIONID) , meta(ImageMeta()), points_v(0,Point2D(0,0))
  {
    probability_of_class = -1;
    type = 0;
    // _box = {(float) meta.col(box.min_x()), (float) meta.row(box.min_y()), (float) (meta.col(box.max_x())-meta.col(box.min_x())), (float) (meta.row(box.max_y())- meta.row(box.min_y())), (float) type};
    // _box.clear();
  }


  ClusterMask::ClusterMask(BBox2D box_in, ImageMeta meta_in, std::vector<Point2D> pts_in, InteractionID_t type_in)
  :  box(box_in), meta(meta_in), points_v(pts_in), type(type_in)
  {
    probability_of_class = -1;
    // _box = {(float) meta_in.col(box.min_x()), (float) meta_in.row(box.min_y()), (float) (meta_in.col(box.max_x())-meta_in.col(box.min_x())), (float) (meta_in.row(box.max_y())-meta_in.row(box.min_y())), (float) type};
    // _mask = std::vector<float>( (box.height()/meta_in.pixel_height()+1) * (box.width()/meta_in.pixel_width()+1), 0.0);
    // for (Point2D pt : points_v){
    //   // _mask[(int)(pt.x - meta_in.col(box.min_x())) * box.height()/meta_in.pixel_height()+1 + pt.y-meta_in.row(box.min_y())] = 1.0;
    //   _mask[(int)(pt.x - meta_in.col(box.min_x())) * ((box.height()/meta_in.pixel_height())+1) + (pt.y - meta_in.row(box.min_y())) ] = 1.0;
    // }
  }


  bool ClusterMask::check_containment() {
    for (Point2D pt : points_v) {
      if ((pt.x >= 0 && pt.y >= 0 && pt.x <= box.width() && pt.y <= box.height())==false) {return false;}
    }
    return true;
  }

  const std::vector<float> ClusterMask::as_vector_mask() const {
    std::vector<float>_mask = std::vector<float>( (box.height()/meta.pixel_height()+1) * (box.width()/meta.pixel_width()+1), 0.0);
    for (Point2D pt : points_v){
      // _mask[(int)(pt.x - meta.col(box.min_x())) * box.height()/meta.pixel_height()+1 + pt.y-meta.row(box.min_y())] = 1.0;
      _mask[(int)(pt.x - meta.col(box.min_x())) * ((box.height()/meta.pixel_height())+1) + (pt.y - meta.row(box.min_y())) ] = 1.0;
    }

    return _mask;
  }
  const std::vector<float> ClusterMask::as_vector_box() const {
    std::vector<float> _box = {(float) meta.col(box.min_x()), (float) meta.row(box.min_y()), (float) (meta.col(box.max_x())-meta.col(box.min_x())), (float) (meta.row(box.max_y())-meta.row(box.min_y())), (float) type};
    return _box;
  }

  const std::vector<float> ClusterMask::as_vector_mask_no_convert() const {
    std::vector<float>_mask = std::vector<float>( points_v.size()*2, 0.0);
    int idx =0;
    for (Point2D pt : points_v){
      _mask[idx*2] = pt.x;
      _mask[idx*2+1] = pt.y;
      idx++;
    }
    return _mask;
  }
  const std::vector<float> ClusterMask::as_vector_box_no_convert() const {
    std::vector<float> _box = {(float)(box.min_x()), (float) (box.min_y()), (float) (box.max_x()), (float) (box.max_y()), (float) type};
    return _box;
  }

}
#endif
