#ifndef PREPROCESSOR_CXX
#define PREPROCESSOR_CXX

#include "PreProcessor.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include <array>

namespace larcv {

  PreProcessor::PreProcessor()
  {
    this->set_verbosity((msg::Level_t)0);
    LARCV_DEBUG() << "start" << std::endl;
    _pi_threshold=10;
    _min_ctor_size=0;
    _allowed_shower_track_distance=5;
    LARCV_DEBUG() << "end" << std::endl;
  }
  
  bool
  PreProcessor::PreProcess(cv::Mat& adc_img, cv::Mat& track_img, cv::Mat& shower_img)
  {
    LARCV_DEBUG() << "start" << std::endl;
    /// Threshold the images to pi_threshold
    cv::Mat adc_img_t, track_img_t, shower_img_t;
    cv::threshold(adc_img   , adc_img_t   , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(track_img , track_img_t , _pi_threshold, 255, CV_THRESH_BINARY);
    cv::threshold(shower_img, shower_img_t, _pi_threshold, 255, CV_THRESH_BINARY);

    /// FindContours
    auto adc_ctor_tmp_v = larocv::FindContours(adc_img_t);
    auto track_ctor_tmp_v = larocv::FindContours(track_img_t);
    auto shower_ctor_tmp_v = larocv::FindContours(shower_img_t);

    /// Remove small contours
    larocv::GEO2D_ContourArray_t adc_ctor_v,track_ctor_v,shower_ctor_v;
    adc_ctor_v.reserve(adc_ctor_tmp_v.size());
    track_ctor_v.reserve(track_ctor_tmp_v.size());
    shower_ctor_v.reserve(shower_ctor_tmp_v.size());
    /// ... remove small adc ctors
    for(auto& ctor:adc_ctor_tmp_v)
      if(ctor.size()>_min_ctor_size)
	adc_ctor_v.emplace_back(std::move(ctor));
    /// ... remove small track ctors
    for(auto& ctor:track_ctor_tmp_v)
      if(ctor.size()>_min_ctor_size)
	track_ctor_v.emplace_back(std::move(ctor));
    /// ... remove small shower ctors
    for(auto& ctor:shower_ctor_tmp_v)
      if(ctor.size()>_min_ctor_size)
	shower_ctor_v.emplace_back(std::move(ctor));
    
    std::vector<larocv::data::LinearTrack2D> track_track2d_v,shower_track2d_v;
    track_track2d_v.reserve(track_ctor_v.size());
    shower_track2d_v.reserve(shower_ctor_v.size());

    // make the track linear tracks
    for(const auto& track_ctor : track_ctor_v) {
      geo2d::Vector<float> edge1,edge2;
      _SingleLinearTrack.EdgesFromMeanValue(track_ctor,edge1,edge2);
      track_track2d_v.emplace_back(track_ctor,edge1,edge2);
    }

    // make the shower linear tracks
    for(const auto& shower_ctor : shower_ctor_v) {
      geo2d::Vector<float> edge1,edge2;
      _SingleLinearTrack.EdgesFromMeanValue(shower_ctor,edge1,edge2);
      shower_track2d_v.emplace_back(shower_ctor,edge1,edge2);
    }

    // find shower contours that and "sandwiched" between linear tracks
    std::vector<std::array<size_t,3> > cidx_v;
    for(size_t shower_id=0;shower_id<shower_track2d_v.size();++shower_id) {
      const auto& shower = shower_track2d_v[shower_id];
      const auto& sedge1 = shower.edge1;
      const auto& sedge2 = shower.edge2;
      LARCV_DEBUG() << "<======== shower " << shower_id << " ========>" << std::endl;
      LARCV_DEBUG() << "sedge1 " << sedge1 << std::endl;
      LARCV_DEBUG() << "sedge2 " << sedge2 << std::endl;
      
      bool bs1t11,bs1t12,bs1t21,bs1t22,bs2t11,bs2t12,bs2t21,bs2t22;
      bool bs1,bs2;
      for(size_t track1_id=0;track1_id<track_track2d_v.size();++track1_id) {
	const auto& track1 = track_track2d_v[track1_id];
	const auto& t1edge1 = track1.edge1;
	const auto& t1edge2 = track1.edge2;
	LARCV_DEBUG() << "t1edge1 " << t1edge1 << std::endl;
	LARCV_DEBUG() << "t1edge2 " << t1edge2 << std::endl;
	for(size_t track2_id=track1_id+1;track2_id<track_track2d_v.size();++track2_id) {
	  LARCV_DEBUG() << "Examining... (s,t1,t2) ("
			<< shower_id << ","
			<< track1_id << ","
			<< track2_id << ")" << std::endl;
	  
	  bs1t11=bs1t12=bs1t21=bs1t22=bs2t11=bs2t12=bs2t21=bs2t22=false;
	  bs1=bs2=false;
	  
	  const auto& track2 = track_track2d_v[track2_id];
	  const auto& t2edge1 = track2.edge1;
	  const auto& t2edge2 = track2.edge2;
	  LARCV_DEBUG() << "t2edge1 " << t2edge1 << std::endl;
	  LARCV_DEBUG() << "t2edge2 " << t2edge2 << std::endl;

	  //have to tracks, determine if edge is compatible
	  auto fs1t11 = geo2d::dist(sedge1,t1edge1);
	  auto fs1t12 = geo2d::dist(sedge1,t1edge2);
	  auto fs1t21 = geo2d::dist(sedge1,t2edge1);
	  auto fs1t22 = geo2d::dist(sedge1,t2edge2);
	  auto fs2t11 = geo2d::dist(sedge2,t1edge1);
	  auto fs2t12 = geo2d::dist(sedge2,t1edge2);
	  auto fs2t21 = geo2d::dist(sedge2,t2edge1);
	  auto fs2t22 = geo2d::dist(sedge2,t2edge2);
	  LARCV_DEBUG() << fs1t11 << "," << fs1t12 << "," << fs1t21 << "," << fs1t22 << "," << fs2t11 << "," << fs2t12 << "," << fs2t21 << "," << fs2t22 << std::endl;
	  if(fs1t11<_allowed_shower_track_distance)bs1t11=true;
	  if(fs1t12<_allowed_shower_track_distance)bs1t12=true;
	  if(fs1t21<_allowed_shower_track_distance)bs1t21=true;
	  if(fs1t22<_allowed_shower_track_distance)bs1t22=true;
	  if(fs2t11<_allowed_shower_track_distance)bs2t11=true;
	  if(fs2t12<_allowed_shower_track_distance)bs2t12=true;
	  if(fs2t21<_allowed_shower_track_distance)bs2t21=true;
	  if(fs2t22<_allowed_shower_track_distance)bs2t22=true;

	  bs1 = ( (bs1t11 or bs1t12) or (bs1t21 or bs1t22) );
	  bs2 = ( (bs2t11 or bs2t12) or (bs2t21 or bs2t22) );

	  if (bs1 && bs2) {
	    LARCV_INFO() << "Compatible match between tracks (" << track1_id << "," << track2_id << ") w/ shower " << shower_id << std::endl;
	    std::array<size_t,3> arr{{shower_id,track1_id,track2_id}};
	    cidx_v.emplace_back(std::move(arr));
	  }
	}
      }
      LARCV_DEBUG() << "<=================" << "========>" << std::endl;
    }

    LARCV_INFO() << "Found " << cidx_v.size() << " compatible tracks and showers" << std::endl;
    //NOTE: there may be duplicate shower index, will remove

    //lets use the shower to mask the ADC image, and append to the track image
    for(const auto& cidx : cidx_v) {
      const auto& shower_ctor = shower_track2d_v[cidx[0]].ctor;
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
