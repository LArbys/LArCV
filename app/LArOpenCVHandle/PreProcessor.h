#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H
#include "Base/larcv_base.h"
#include <opencv2/opencv.hpp>
#include "LArOpenCV/ImageCluster/AlgoClass/SingleLinearTrack.h"

namespace larcv {
  class PreProcessor : public larcv_base{

  public:
    
    PreProcessor();
    ~PreProcessor(){}
    
    bool
    PreProcess(cv::Mat& adc_img, cv::Mat& track_img, cv::Mat& shower_img);

  private:
    uint _pi_threshold;
    uint _min_ctor_size;
    float _allowed_shower_track_distance;
    
    larocv::SingleLinearTrack _SingleLinearTrack;
  };
}
#endif


