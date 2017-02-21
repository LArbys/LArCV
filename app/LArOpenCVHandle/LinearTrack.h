#ifndef LINEARTRACK_H
#define LINEARTRACK_H
#include "LArOpenCV/Core/LArOCVTypes.h"

namespace larcv {

  enum class Type_t { kUnknown, kTrack, kShower };
  
  struct LinearTrack {
    LinearTrack() :
      track_frac(0),
      shower_frac(0),
      type(Type_t::kUnknown),
      ignore(false),
      straight(false)
    {}
    
    ~LinearTrack() {}
    larocv::GEO2D_Contour_t ctor;
    geo2d::Vector<float> edge1;
    geo2d::Vector<float> edge2;
    float length;
    float width;
    float perimeter;
    float area;
    uint npixel;
    geo2d::Line<float> overallPCA;
    geo2d::Line<float> edge1PCA;
    geo2d::Line<float> edge2PCA;
    float track_frac;
    float shower_frac;
    Type_t type;
    bool ignore;
    double mean_pixel_dist;
    double sigma_pixel_dist;
    bool straight;
  };

}
#endif
