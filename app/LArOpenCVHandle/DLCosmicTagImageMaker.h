#ifndef DLCOSMICTAG_IMAGEMAKER_H
#define DLCOSMICTAG_IMAGEMAKER_H

/* *******************************************************
 * DLCosmicTagImageMaker
 * 
 * Prepares inputs from upstream DLCosmicTag output
 *  for use in LArOpenCV ImageCluster framework
 *  for vertex reconstruction.
 *
 * ******************************************************* */

#include <tuple>

// larlite

// larcv
#include "Base/larcv_base.h"
#include "Base/PSet.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"

// laropencv
#include "LArOpenCV/Core/ImageManager.h"


#ifndef __CLING__
#ifndef __CINT__
#include <opencv2/opencv.hpp>
#endif
#endif

namespace larcv {
  
  class DLCosmicTagImageMaker : public larcv_base {

  public:
    
    DLCosmicTagImageMaker() {};
    ~DLCosmicTagImageMaker(){};

    void
      Configure(const PSet& pset);

    /* std::vector<cv::Mat> */
    /*   ExtractMat(const std::vector<Image2D>& image_v); */

    /* cv::Mat */
    /*   ExtractMat(const Image2D& image); */
    
    /* std::tuple<cv::Mat,larocv::ImageMeta> */
    /*   ExtractImage(const Image2D& image, size_t plane=0); */
    
    /* std::vector<std::tuple<cv::Mat,larocv::ImageMeta> > */
    /*   ExtractImage(const std::vector<Image2D>& image_v); */
    

    /* Image2D ConstructCosmicImage(const EventPixel2D& ev_pixel2d, */
    /* 				 const Image2D& adc_image, */
    /* 				 const size_t plane, */
    /* 				 float value=100); */

    /* Image2D ConstructCosmicImage(const EventPixel2D* ev_pixel2d, */
    /* 				 const Image2D& adc_image, */
    /* 				 const size_t plane, */
    /* 				 float value); */
    
    /* void ConstructCosmicImage(IOManager& mgr, */
    /* 			      std::string producer, */
    /* 			      ProductType_t datatype,  */
    /* 			      const std::vector<larcv::Image2D>& adc_image_v, */
    /* 			      std::vector<larcv::Image2D>& mu_image_v); */
    
  public:

    /* float _charge_max; */
    /* float _charge_min; */
    /* float _charge_to_gray_scale; */
    
  };
}
#endif
/** @} */ // end of doxygen group 

