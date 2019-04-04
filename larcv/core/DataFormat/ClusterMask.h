/**
 * \file ClusterMask.h
 *
 * \ingroup DataFormat
 *
 * \brief Class def header for a class ClusterMask
 *
 * @author JoshuaMills
 */

/** \addtogroup DataFormat

    @{*/
#ifndef LARCV_CLUSTERMASK_H
#define LARCV_CLUSTERMASK_H

#include <iostream>
#include <cmath>
//For LArCV 2
#include "Point.h"
// #include "../Base/LArCVTypes.h"
#include "BBox.h"
#include "ImageMeta.h"
namespace larcv {

  /**
     \class ClusterMask
     Simple 2D point struct (unit of "x" and "y" are not defined here and app specific)
  */
  typedef int InteractionID_t;


  class ClusterMask {
  public:
    ClusterMask();
    ClusterMask(BBox2D box_in, ImageMeta meta, std::vector<Point2D> pts_in, InteractionID_t type_in);

    ~ClusterMask() {}


    BBox2D box; //Placeholder for bbox
    ImageMeta meta;
    std::vector<Point2D> points_v;
    InteractionID_t type;
    float probability_of_class;
    // std::vector<float> _mask;
    // std::vector<float> _box;


    double get_fraction_clustered() {return points_v.size() / (box.area());}
    bool check_containment() ;
    const std::vector<float> as_vector_mask() const ; // Return wrapped sparse array
    const std::vector<float> as_vector_box() const  ; // (x1,y1,width,height,type)
    const std::vector<float> as_vector_mask_no_convert() const ; // Return vector of points(x1,y1,x2,y2,...xn,yn)
    const std::vector<float> as_vector_box_no_convert() const  ; // Return vector of box corners without converting via meta
                                                                 // (whatever coord bbox is in) (x1,y1,x2,y2,type)


  };
  //End of Class

}
#endif
/** @} */ // end of doxygen group
