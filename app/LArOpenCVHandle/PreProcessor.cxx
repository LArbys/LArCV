#ifndef PREPROCESSOR_CXX
#define PREPROCESSOR_CXX

#include "PreProcessor.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include <array>
#include "Geo2D/Core/Line.h"

namespace larcv {

  PreProcessor::PreProcessor()
  {
    this->set_verbosity((msg::Level_t)0);
    LARCV_DEBUG() << "start" << std::endl;
    _pi_threshold=1;
    _min_ctor_size=10;
    _allowed_shower_track_distance=10;//10;
    _blur = 4;//0
    _pca_box_size=5;
    _min_overall_angle=10;
    LARCV_DEBUG() << "end" << std::endl;
  }

  void
  PreProcessor::Configure(const fcllite::PSet& pset) {
    LARCV_DEBUG() << "start" << std::endl;
    /// configure...
    LARCV_DEBUG() << "end" << std::endl;
    return;
  }
  
  void
  PreProcessor::FilterContours(larocv::GEO2D_ContourArray_t& ctor_v,const cv::Mat& img) {
    larocv::GEO2D_ContourArray_t ctor_tmp_v;
    ctor_tmp_v.reserve(ctor_v.size());

    for(auto& ctor : ctor_v)
      if(ctor.size()>_min_ctor_size)
	ctor_tmp_v.emplace_back(std::move(ctor));
    
    std::swap(ctor_tmp_v,ctor_v);
    return;
  }

  std::vector<LinearTrack>
  PreProcessor::MakeLinearTracks(const larocv::GEO2D_ContourArray_t& ctor_v,
				 const cv::Mat& img) {
    
    std::vector<LinearTrack> track_v;
    track_v.reserve(ctor_v.size());
    
    for(const auto& ctor : ctor_v) {

      // make the edge
      geo2d::Vector<float> edge1,edge2;
      _SingleLinearTrack.EdgesFromMeanValue(ctor,edge1,edge2);

      //rotated rect coordinates      
      auto min_rect  = cv::minAreaRect(ctor);
      cv::Point2f vertices[4];
      min_rect.points(vertices);

      //set parameters from rotated rect
      auto rect = min_rect.size;
      auto length     = rect.height > rect.width ? rect.height : rect.width;
      auto width      = rect.height > rect.width ? rect.width  : rect.height;
      auto perimeter  = cv::arcLength(ctor,1);
      auto area       = cv::contourArea(ctor);
      auto npixel     = cv::countNonZero(larocv::MaskImage(img,ctor,0,false));
      auto overallPCA = larocv::CalcPCA(ctor);
      auto edge1PCA   = larocv::SquarePCA(img,edge1,_pca_box_size,_pca_box_size);
      auto edge2PCA   = larocv::SquarePCA(img,edge2,_pca_box_size,_pca_box_size);
      
      // axis aligned
      //auto bounding_rect = cv::boundingRect(ctor);
      //auto min_bounding_rect = {bounding_rect.br(),bounding_rect.tl()};
      
      //get the length, area, npixels
      track_v.emplace_back(ctor,edge1,edge2,length,width,perimeter,area,npixel,overallPCA,edge1PCA,edge2PCA);
    }

    return track_v;
  }


  bool PreProcessor::EdgeConnected(const LinearTrack& track1,
				   const LinearTrack& track2) {
    const auto& t1edge1 = track1.edge1;
    const auto& t1edge2 = track1.edge2;

    const auto& t2edge1 = track2.edge1;
    const auto& t2edge2 = track2.edge2;

    bool bt11t21,bt11t22,bt12t21,bt12t22;
    bt11t21=bt11t22=bt12t21=bt12t22=false;

    auto ft11t21 = geo2d::dist(t1edge1,t2edge1);
    auto ft11t22 = geo2d::dist(t1edge1,t2edge2);
    auto ft12t21 = geo2d::dist(t1edge2,t2edge1);
    auto ft12t22 = geo2d::dist(t1edge2,t2edge2);
    
    if(ft11t21 < _allowed_shower_track_distance) bt11t21=true;
    if(ft11t22 < _allowed_shower_track_distance) bt11t22=true;
    if(ft12t21 < _allowed_shower_track_distance) bt12t21=true;
    if(ft12t22 < _allowed_shower_track_distance) bt12t22=true;

    if (bt11t21)
      { LARCV_DEBUG() << "Track 1 edge1 & Track 2 edge1 neaby" << std::endl; return true; }
    if (bt11t22)
      { LARCV_DEBUG() << "Track 1 edge1 & Track 2 edge2 nearby" << std::endl; return true; }
    if (bt12t21)
      { LARCV_DEBUG() << "Track 1 edge2 & Track 2 edge1 nearby" << std::endl; return true; }
    if (bt12t22)
      { LARCV_DEBUG() << "Track 1 edge2 & Track 2 edge2 nearby" << std::endl; return true; }
    
    return false;
  }
  
  
  bool
  PreProcessor::PreProcess(cv::Mat& adc_img, cv::Mat& track_img, cv::Mat& shower_img)
  {
    LARCV_DEBUG() << "start" << std::endl;

    // Copy input
    cv::Mat adc_img_t = adc_img.clone();
    cv::Mat track_img_t = track_img.clone();
    cv::Mat shower_img_t = shower_img.clone();
    
    if(this->logger().level() == msg::kDEBUG) {
      cv::imwrite("adc_img.png",adc_img);
      cv::imwrite("track_img.png",track_img);
      cv::imwrite("shower_img.png",shower_img);
    }
    
    // Blur
    if (_blur) {
      cv::blur(adc_img_t   ,adc_img_t,::cv::Size(_blur,_blur) );
      cv::blur(track_img_t ,track_img_t,::cv::Size(_blur,_blur) );
      cv::blur(shower_img_t,shower_img_t,::cv::Size(_blur,_blur) );
      if(this->logger().level() == msg::kDEBUG) {
	cv::imwrite("adc_img_t0.png",adc_img_t);
	cv::imwrite("track_img_t0.png",track_img_t);
	cv::imwrite("shower_img_t0.png",shower_img_t);
      }
    }
    
    // Threshold
    cv::threshold(adc_img_t   , adc_img_t   , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(track_img_t , track_img_t , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(shower_img_t, shower_img_t, _pi_threshold, 255, CV_THRESH_BINARY);

    if(this->logger().level() == msg::kDEBUG) {
      cv::imwrite("adc_img_t1.png",adc_img_t);
      cv::imwrite("track_img_t1.png",track_img_t);
      cv::imwrite("shower_img_t1.png",shower_img_t);
    }
    
    // FindContours
    auto adc_ctor_v = larocv::FindContours(adc_img_t);
    auto track_ctor_v = larocv::FindContours(track_img_t);
    auto shower_ctor_v = larocv::FindContours(shower_img_t);

    // Filter them by size (number of contour points)
    FilterContours(adc_ctor_v   ,adc_img);
    FilterContours(track_ctor_v ,track_img);
    FilterContours(shower_ctor_v,shower_img);

    // Make linear track
    auto track_lintrk_v = MakeLinearTracks(track_ctor_v,track_img_t);
    auto shower_lintrk_v = MakeLinearTracks(shower_ctor_v,shower_img_t);

    // Shower contours that are probably track..
    std::vector<size_t> cidx_v;
    
    // Find "sandwich" showers -- showers between two tracks    
    for(size_t shower_id=0;shower_id<shower_lintrk_v.size();++shower_id) {
      const auto& shower = shower_lintrk_v[shower_id];
      for(size_t track1_id=0;track1_id<track_lintrk_v.size();++track1_id) {
	const auto& track1 = track_lintrk_v[track1_id];
	if (!EdgeConnected(shower,track1))
	  continue;
	for(size_t track2_id=track1_id+1;track2_id<track_lintrk_v.size();++track2_id) {
	  const auto& track2 = track_lintrk_v[track2_id];
	  if (!EdgeConnected(shower,track2))
	    continue;
	  
	  LARCV_DEBUG() << "Shower " << shower_id
			<< " sandwich btw"
			<< " track " << track1_id
			<< " & track " << track2_id
			<< std::endl;

	  cidx_v.push_back(shower_id);
	} // end track2

	auto angle = geo2d::angle(shower.overallPCA,track1.overallPCA);
	if (angle < _min_overall_angle) 
	  cidx_v.push_back(shower_id);

      } // end track1
    } // end this shower

    
    
    
    LARCV_INFO() << "Found " << cidx_v.size() << " compatible tracks and showers" << std::endl;
    //lets use the shower to mask the ADC image, and append to the track image
    for(const auto& cidx : cidx_v) {
      const auto& shower_ctor = shower_lintrk_v[cidx].ctor;
      //get a mask of the ADC image
      auto mask_adc = larocv::MaskImage(adc_img,shower_ctor,0,false);
      //add it to the track image
      track_img += mask_adc;
      //mask it OUT of the shower iamge
      shower_img = larocv::MaskImage(shower_img,shower_ctor,0,true);
    }
    
    LARCV_DEBUG() << "end" << std::endl;    
    return true;
  }
}

#endif
