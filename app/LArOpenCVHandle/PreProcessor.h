#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "Base/larcv_base.h"
#include <opencv2/opencv.hpp>
#include "PixelChunk.h"
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

    void
    MergeTracklets(cv::Mat& track_img, cv::Mat& shower_img);

    void
    ClaimShowers(cv::Mat& adc_img,
		 cv::Mat& track_img,
		 cv::Mat& shower_img);

    void
    MergePixelByFraction(cv::Mat& adc_img,
			 cv::Mat& track_img,
			 cv::Mat& shower_img);

    void
    MergeStraightShowers(cv::Mat& adc_img,
			 cv::Mat& track_img,
			 cv::Mat& shower_img);
    
  private:
    bool
    OverallStraightCompatible(const PixelChunk& pchunk1,
			      const PixelChunk& pchunk2);
    
    bool
    IsStraight(const PixelChunk& track,
	       const cv::Mat& img);
    
    void
    FilterContours(larocv::GEO2D_ContourArray_t& ctor_v);
		   
    std::vector<PixelChunk>
    MakePixelChunks(const cv::Mat& img,
		    Type_t type,
		    bool calc_params=true,
		    size_t min_ctor_size=2,
		    size_t min_track_size=0);
    bool
    EdgeConnected(const PixelChunk& track1,
		  const PixelChunk& track2);

    cv::Mat
    PrepareImage(const cv::Mat& img);

    float
    GetClosestEdge(const PixelChunk& track1, const PixelChunk& track2,
		   geo2d::Vector<float>& edge1, geo2d::Vector<float>& edge2);
    float
    GetClosestEdge(const PixelChunk& track1, const PixelChunk& track2);

    
  private:
    
    uint _pi_threshold;
    uint _min_ctor_size;
    uint _blur;
    float _allowed_neighbor_dist;
    uint _pca_box_size;
    float _min_overall_angle;
    float _min_pca_angle;
    float _min_track_size;
    bool _merge_tracklets;
    bool _merge_pixel_frac;
    bool _claim_showers;
    float _min_track_frac;
    float _min_shower_frac;
    float _max_track_in_shower_frac;
    double _mean_distance_pca;
    uint _min_merge_track_size;
    bool _merge_straight_showers;
    uint _min_merge_track_shower_dist;
    double _allowed_merge_neighbor_dist;
    double _allowed_edge_overlap;
    
  };
}
#endif


