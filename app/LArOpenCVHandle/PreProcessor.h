#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H
#include "Base/larcv_base.h"
#include <opencv2/opencv.hpp>
#include "LArOpenCV/ImageCluster/AlgoClass/SingleLinearTrack.h"

namespace larcv {

  struct LinearTrack {
    LinearTrack() {}
    ~LinearTrack() {}

    LinearTrack(const larocv::GEO2D_Contour_t& ctor_,
		const geo2d::Vector<float>& edge1_,
		const geo2d::Vector<float>& edge2_,
		const float length_,
		const float width_,
		const float perimeter_,
		const float area_,
		const uint npixel_,
		geo2d::Line<float> overallPCA_,
		geo2d::Line<float> edge1PCA_,
		geo2d::Line<float> edge2PCA_) :
      ctor(ctor_),
      edge1(edge1_),
      edge2(edge2_),
      length(length_),
      width(width_),
      perimeter(perimeter_),
      area(area_),
      npixel(npixel_),
      overallPCA(overallPCA_),
      edge1PCA(edge1PCA_),
      edge2PCA(edge2PCA_)
    {}

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
  };


  class PreProcessor : public larcv_base{

  public:
    
    PreProcessor();
    ~PreProcessor(){}
    
    bool
    PreProcess(cv::Mat& adc_img,
	       cv::Mat& track_img,
	       cv::Mat& shower_img);

  private:
    void
    FilterContours(larocv::GEO2D_ContourArray_t& ctor_v,
		   const cv::Mat& img);
    void
    Configure(const fcllite::PSet& pset);

    std::vector<LinearTrack>
    MakeLinearTracks(const larocv::GEO2D_ContourArray_t& ctor_v,
		     const cv::Mat& img);

    bool
    IsSandwich(const LinearTrack& shower,
	       const LinearTrack& track1,
	       const LinearTrack& track2);
    
  private:
    uint _pi_threshold;
    uint _min_ctor_size;
    uint _blur;
    float _allowed_shower_track_distance;
    uint _pca_box_size;
    
    larocv::SingleLinearTrack _SingleLinearTrack;

    
  };
}
#endif


