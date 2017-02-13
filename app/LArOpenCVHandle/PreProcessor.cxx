#ifndef PREPROCESSOR_CXX
#define PREPROCESSOR_CXX

#include "PreProcessor.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"

namespace larcv {

  PreProcessor::PreProcessor()
  {
    _pi_threshold=10;  
  }
  
  bool
  PreProcessor::PreProcess(cv::Mat& adc_img, cv::Mat& track_img, cv::Mat& shower_img)
  {
    
    //threshold the images to pi_threshold
    cv::Mat adc_img_t, track_img_t, shower_img_t;
    cv::threshold(adc_img   , adc_img_t   , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(track_img , track_img_t , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(shower_img, shower_img_t, _pi_threshold, 255, CV_THRESH_BINARY);

    auto adc_ctor_v = larocv::FindContours(adc_img_t);
    auto track_ctor_v = larocv::FindContours(track_img_t);
    auto shower_ctor_v = larocv::FindContours(shower_img_t);
    
    return true;
  }

}

#endif
