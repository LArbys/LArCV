#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "Base/larcv_base.h"
#include <opencv2/opencv.hpp>
#include "LArOpenCV/ImageCluster/AlgoClass/SingleLinearTrack.h"
#include "LinearTrack.h"
#include "Base/PSet.h"

namespace larcv {
  
  class PreProcessor : public larcv_base {
    
  public:
    
    PreProcessor();
    ~PreProcessor(){}
    
    bool
    PreProcess(cv::Mat& adc_img,
	       cv::Mat& track_img,
	       cv::Mat& shower_img);

    void
    Configure(const PSet& pset);

    
  private:

    bool
    IsStraight(const LinearTrack& track,
	       const cv::Mat& img);
    
    void
    FilterContours(larocv::GEO2D_ContourArray_t& ctor_v);
		   
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


