#ifndef LARBYSIMAGEMAKER_H
#define LARBYSIMAGEMAKER_H

#include "Base/larcv_base.h"
#include "Base/PSet.h"
#include "LArOpenCV/Core/ImageManager.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include <opencv2/opencv.hpp>
#include <tuple>


namespace larcv {
  class LArbysImageMaker : public larcv_base {

  public:
    LArbysImageMaker() :
      _charge_max(500),
      _charge_min(0.1),
      _charge_to_gray_scale(2)
    {}
    
    ~LArbysImageMaker(){}

    void
    Configure(const PSet& pset);

    std::vector<cv::Mat>
    ExtractMat(const std::vector<larcv::Image2D>& image_v);
    
    std::vector<std::tuple<cv::Mat,larocv::ImageMeta> >
    ExtractImage(const std::vector<larcv::Image2D>& image_v);

  private:
    float _charge_max;
    float _charge_min;
    float _charge_to_gray_scale;
    
  };
}
#endif
/** @} */ // end of doxygen group 

