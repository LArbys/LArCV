#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H
#include "Base/larcv_base.h"
#include <opencv2/opencv.hpp>
#include "LArOpenCV/ImageCluster/AlgoClass/SingleLinearTrack.h"

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
    bool straight;
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


    bool
    IsStraight(const LinearTrack& track,
	       const cv::Mat& img);
    
    void
    FilterContours(larocv::GEO2D_ContourArray_t& ctor_v);
		   
    void
    Configure(const fcllite::PSet& pset);

    std::vector<LinearTrack>
    MakeLinearTracks(const larocv::GEO2D_ContourArray_t& ctor_v,
		     const cv::Mat& img,
		     Type_t type,
		     bool calc_params=true);
    bool
    EdgeConnected(const LinearTrack& track1,
		  const LinearTrack& track2);

    cv::Mat
    PrepareImage(const cv::Mat& img);

    float
    GetClosestEdge(const LinearTrack& track1, const LinearTrack& track2,
		   geo2d::Vector<float>& edge1, geo2d::Vector<float>& edge2);
    float
    GetClosestEdge(const LinearTrack& track1, const LinearTrack& track2);
    
  private:
    uint _pi_threshold;
    uint _min_ctor_size;
    uint _blur;
    float _allowed_neighbor_dist;
    uint _pca_box_size;
    float _min_overall_angle;
    float _min_pca_angle;
    float _min_track_size;
    bool _merge_pixel_frac;
    bool _claim_showers;
    float _min_track_frac;
    float _min_shower_frac;
    float _max_track_in_shower_frac;
    float _save_straight_tracks_frac;
    double _mean_distance_pca;
    larocv::SingleLinearTrack _SingleLinearTrack;

    
  };
}
#endif


