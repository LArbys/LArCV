#ifndef PREPROCESSOR_CXX
#define PREPROCESSOR_CXX

#include "PreProcessor.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/Contour2DAnalysis.h"
#include "LArOpenCV/ImageCluster/AlgoFunction/ImagePatchAnalysis.h"
#include "Geo2D/Core/Line.h"
#include <array>
#include <limits>

namespace larcv {

  PreProcessor::PreProcessor()
  {
    this->set_verbosity((msg::Level_t)0);
    LARCV_DEBUG() << "start" << std::endl;
    _pi_threshold=1;
    _min_ctor_size=4;
    _min_track_size=3;
    _allowed_neighbor_dist=10; //10;
    _blur = 4; //0
    _pca_box_size=5;
    _min_overall_angle=10;
    _min_pca_angle=10;
    _merge_pixel_frac=true;
    _claim_showers=true;
    _min_track_frac = 0.8;
    _min_shower_frac = 0.8;
    _save_straight_tracks_frac = 0.1;
    _max_track_in_shower_frac = 0.1;
    _mean_distance_pca = 3;
    LARCV_DEBUG() << "end" << std::endl;
  }

  void
  PreProcessor::Configure(const PSet& pset) {
    LARCV_DEBUG() << "start" << std::endl;

    _pi_threshold = pset.get<uint>("PiThreshold",1);
    _min_ctor_size = pset.get<uint>("MinContourSize",4);
    _min_track_size = pset.get<uint>("MinTrackSize",3);
    _allowed_neighbor_dist = pset.get<float>("AllowedNeighborSeparation",10);
    _blur = pset.get<uint>("BlurSize",4);
    _pca_box_size = pset.get<uint>("EdgePCABoxSize",5);
    _min_overall_angle = pset.get<float>("MinPCAOverallAngle",10);
    
    LARCV_DEBUG() << "end" << std::endl;
    return;
  }

  bool
  PreProcessor::IsStraight(const LinearTrack& track,const cv::Mat& img) {
    
    auto e1e2_a = std::abs(geo2d::angle(track.edge1PCA,track.edge2PCA));
    auto e1oA_a = std::abs(geo2d::angle(track.edge1PCA,track.overallPCA));
    auto e2oA_a = std::abs(geo2d::angle(track.edge2PCA,track.overallPCA));
    
    // auto oe1PCA_a  = std::abs(geo2d::angle(track.overallPCA,track.edge1PCA));
    bool e1e2_b = false;
    bool e1oA_b = false;
    bool e2oA_b = false;
    bool pca_b = false;
    
    if ((e1e2_a < _min_pca_angle) or (e1e2_a > 180-_min_pca_angle)) // probably straight
      e1e2_b=true;
    if ((e1oA_a < _min_pca_angle) or (e1oA_a > 180-_min_pca_angle)) // probably straight
      e1oA_b=true;
    if ((e2oA_a < _min_pca_angle) or (e2oA_a > 180-_min_pca_angle)) // probably straight
      e2oA_b=true;
    
    if ( track.mean_pixel_dist < _mean_distance_pca)
      pca_b = true;
    
    return (pca_b and (e1e2_b or e1oA_b or e2oA_b));
  }
  
  float
  PreProcessor::GetClosestEdge(const LinearTrack& track1, const LinearTrack& track2,
			       geo2d::Vector<float>& edge1, geo2d::Vector<float>& edge2) {
    

    geo2d::Vector<float> e1,e2;
    float dist=std::numeric_limits<float>::max();
    
    auto dt1e1t2e1 = geo2d::dist(track1.edge1,track2.edge1);
    auto dt1e1t2e2 = geo2d::dist(track1.edge1,track2.edge2);
    auto dt1e2t2e1 = geo2d::dist(track1.edge2,track2.edge1);
    auto dt1e2t2e2 = geo2d::dist(track1.edge2,track2.edge2);

    if (dt1e1t2e1 < dist)
      { dist = dt1e1t2e1; edge1=track1.edge1; edge2=track2.edge1; }
    if (dt1e1t2e2 < dist)
      { dist = dt1e1t2e2; edge1=track1.edge1; edge2=track2.edge2; }
    if (dt1e2t2e1 < dist)
      { dist = dt1e2t2e1; edge1=track1.edge2; edge2=track2.edge1; }
    if (dt1e2t2e2 < dist)
      { dist = dt1e2t2e1; edge1=track1.edge2; edge2=track2.edge2; }
    
    return dist;
  }

  float
  PreProcessor::GetClosestEdge(const LinearTrack& track1, const LinearTrack& track2) {
    geo2d::Vector<float> temp1,temp2;
    return GetClosestEdge(track1,track2,temp1,temp2);
  }

  
  void
  PreProcessor::FilterContours(larocv::GEO2D_ContourArray_t& ctor_v) {
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
				 const cv::Mat& img,
				 Type_t type,
				 bool calc_params) {
    
    std::vector<LinearTrack> track_v;
    track_v.reserve(ctor_v.size());
    
    for(const auto& ctor : ctor_v) {
      LinearTrack lintrack;
      lintrack.npixel     = cv::countNonZero(larocv::MaskImage(img,ctor,0,false));
      
      if (lintrack.npixel<_min_track_size) continue;

      lintrack.ctor = ctor;

      if (calc_params) {
	// make the edge
	geo2d::Vector<float> edge1,edge2;
	_SingleLinearTrack.EdgesFromMeanValue(ctor,edge1,edge2);

	//rotated rect coordinates      
	auto min_rect  = cv::minAreaRect(ctor);
	cv::Point2f vertices[4];
	min_rect.points(vertices);

	lintrack.edge1 = edge1;
	lintrack.edge2 = edge2;

	//set parameters from rotated rect
	auto rect = min_rect.size;
	lintrack.length = rect.height > rect.width ? rect.height : rect.width;
	lintrack.width = rect.height > rect.width ? rect.width  : rect.height;
	lintrack.perimeter = cv::arcLength(ctor,1);
	lintrack.area = cv::contourArea(ctor);
	lintrack.overallPCA = larocv::CalcPCA(ctor);
	lintrack.edge1PCA = larocv::SquarePCA(img,edge1,_pca_box_size,_pca_box_size);
	lintrack.edge2PCA = larocv::SquarePCA(img,edge2,_pca_box_size,_pca_box_size);
	lintrack.type = type;
	lintrack.track_frac = type==Type_t::kTrack  ? 1 : 0;
	lintrack.shower_frac = type==Type_t::kShower ? 1 : 0;
	auto masked_pts=larocv::MaskImage(img,ctor,0,false);
	lintrack.mean_pixel_dist = larocv::MeanDistanceToLine(masked_pts,lintrack.overallPCA);
	lintrack.sigma_pixel_dist = larocv::SigmaDistanceToLine(masked_pts,lintrack.overallPCA);
	lintrack.straight = IsStraight(lintrack,img);
      }
      // axis aligned
      //auto bounding_rect = cv::boundingRect(ctor);
      //auto min_bounding_rect = {bounding_rect.br(),bounding_rect.tl()};
      
      //get the length, area, npixels
      track_v.emplace_back(std::move(lintrack));
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
    
    if(ft11t21 < _allowed_neighbor_dist) bt11t21=true;
    if(ft11t22 < _allowed_neighbor_dist) bt11t22=true;
    if(ft12t21 < _allowed_neighbor_dist) bt12t21=true;
    if(ft12t22 < _allowed_neighbor_dist) bt12t22=true;

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


  cv::Mat PreProcessor::PrepareImage(const cv::Mat& img) {
    // Copy input
    cv::Mat img_t = img.clone();

    if(this->logger().level() == msg::kDEBUG) {
      static uint call0=0;
      std::stringstream ss;
      ss << "img_" << call0 << ".png";
      cv::imwrite(ss.str(),img_t);
      call0+=1;
    }
    
    // Blur
    if (_blur) {
      cv::blur(img_t,img_t,::cv::Size(_blur,_blur) );
      if(this->logger().level() == msg::kDEBUG) {
	static uint call1=0;
	std::stringstream ss;
	ss << "img_t0_" << call1 << ".png";
	cv::imwrite(ss.str(),img_t);
	call1+=1;
      }
    }
    
    // Threshold
    img_t = larocv::Threshold(img_t,_pi_threshold,255);
    
    if(this->logger().level() == msg::kDEBUG) {
      static uint call2=0;
      std::stringstream ss;
      ss << "img_t1_" << call2 << ".png";
      cv::imwrite(ss.str(),img_t);
      call2+=1;
    }

    return img_t;
  }
  
  bool
  PreProcessor::PreProcess(cv::Mat& adc_img, cv::Mat& track_img, cv::Mat& shower_img)
  {
    LARCV_DEBUG() << "start" << std::endl;

    cv::Mat adc_img_t,track_img_t,shower_img_t;

    // prepare the image via threshold and blur
    adc_img_t = PrepareImage(adc_img);
    track_img_t = PrepareImage(track_img);
    shower_img_t = PrepareImage(shower_img);
    
    // FindContours
    larocv::GEO2D_ContourArray_t adc_raw_ctor_v,adc_ctor_v,track_ctor_v,shower_ctor_v;
    adc_raw_ctor_v = larocv::FindContours(adc_img.clone());
    adc_ctor_v  = larocv::FindContours(adc_img_t);
    track_ctor_v = larocv::FindContours(track_img_t);
    shower_ctor_v = larocv::FindContours(shower_img_t);
    
    // Filter them by size (number of contour points)
    FilterContours(adc_ctor_v);
    FilterContours(track_ctor_v);
    FilterContours(shower_ctor_v);
    LARCV_DEBUG() << "Found " << adc_ctor_v.size() << " adc contours" << std::endl;
    LARCV_DEBUG() << "Found " << track_ctor_v.size() << " track contours" << std::endl;
    LARCV_DEBUG() << "Found " << shower_ctor_v.size() << " shower contours" << std::endl;

    std::vector<LinearTrack> adc_lintrk_v,track_lintrk_v,shower_lintrk_v;

    if (_claim_showers) {
      LARCV_DEBUG() << "Cleaning up showers first..." << std::endl;
      // Make simple linear tracks
      adc_lintrk_v = MakeLinearTracks(adc_ctor_v,adc_img_t,Type_t::kUnknown,false);
      track_lintrk_v = MakeLinearTracks(track_ctor_v,track_img_t,Type_t::kTrack,false);
      shower_lintrk_v = MakeLinearTracks(shower_ctor_v,shower_img_t,Type_t::kShower,false);

      for(auto& shower_lintrk : shower_lintrk_v) {
	auto& shower_ctor = shower_lintrk.ctor;
	auto& track_frac = shower_lintrk.track_frac;
	for(const auto& track_lintrk : track_lintrk_v) {
	  auto& track_ctor = track_lintrk.ctor;
	  track_frac += larocv::PixelFraction(adc_img_t,shower_ctor,track_ctor);
	}

	if (track_frac < _max_track_in_shower_frac) {
	  //mask these pixels out of the track image
	  auto mask_shower = larocv::MaskImage(track_img,shower_ctor,0,false);
	  //mask away these pixels in the track image
	  track_img = larocv::MaskImage(track_img,shower_ctor,0,true);
	  //update the shower
	  shower_img = shower_img + mask_shower;
	}
      }

      //re-prepare the track and shower images
      track_img_t = PrepareImage(track_img);
      shower_img_t = PrepareImage(shower_img);

      //re-prepare the track and shower contours
      track_ctor_v = larocv::FindContours(track_img_t);
      shower_ctor_v = larocv::FindContours(shower_img_t);

      // Filter them by by number of contour points
      FilterContours(track_ctor_v);
      FilterContours(shower_ctor_v);
      LARCV_DEBUG() << "Re-Found " << track_ctor_v.size() << " track contours" << std::endl;
      LARCV_DEBUG() << "Re-Found " << shower_ctor_v.size() << " shower contours" << std::endl;      
      
    }
    
    if (_merge_pixel_frac) {

      adc_lintrk_v = MakeLinearTracks(adc_ctor_v,adc_img_t,Type_t::kUnknown,false);
      track_lintrk_v = MakeLinearTracks(track_ctor_v,track_img_t,Type_t::kTrack,true);
      shower_lintrk_v = MakeLinearTracks(shower_ctor_v,shower_img_t,Type_t::kShower,true);
     
      LARCV_DEBUG() << "Found " << adc_lintrk_v.size() << " adc & " << track_lintrk_v.size() << " tracks & " << shower_lintrk_v.size() << " shower linear track" << std::endl;

      /// determine track/shower fraction of ADC contours
      for(auto& adc_lintrk : adc_lintrk_v) {
	auto& adc_ctor = adc_lintrk.ctor;
	auto& track_frac = adc_lintrk.track_frac;
	auto& shower_frac = adc_lintrk.shower_frac;

	LARCV_DEBUG() << "ADC track @ " << &adc_lintrk << " contour size " << adc_ctor.size() << std::endl;

	const LinearTrack* track = nullptr;
	const LinearTrack* shower = nullptr;
	float largest_track_frac(std::numeric_limits<float>::min());
	float largest_shower_frac(std::numeric_limits<float>::min());
	
	for(const auto& track_lintrk : track_lintrk_v) {
	  auto frac = larocv::PixelFraction(adc_img_t,adc_ctor,track_lintrk.ctor);

	  if (frac > largest_track_frac)
	    {largest_track_frac = frac; track = &track_lintrk;}
	  
	  track_frac += frac;
	}
	  
	for(const auto& shower_lintrk : shower_lintrk_v) { 
	  auto frac = larocv::PixelFraction(adc_img_t,adc_ctor,shower_lintrk.ctor);

	  if (frac>largest_shower_frac)
	    {largest_shower_frac = frac; shower = &shower_lintrk;}

	  shower_frac += frac;
	}
	
	if (!track or !shower) continue;
	
	if(largest_track_frac > _save_straight_tracks_frac)
	  if (GetClosestEdge(*track,*shower) < _allowed_neighbor_dist)
	    adc_lintrk.ignore=true;
	
	LARCV_DEBUG() << " ... track frac " << track_frac << " & shower_frac " << shower_frac << std::endl;
      }
    
      // loop over the ADC contours, claim contours above threshold as track and shower
      for(auto& adc_lintrk : adc_lintrk_v) {

	auto& adc_ctor = adc_lintrk.ctor;
	// auto& adc_small_ctor = adc_raw_ctor_v.at(larocv::FindContainingContour(adc_raw_ctor_v,adc_ctor));
	auto& track_frac = adc_lintrk.track_frac;
	auto& shower_frac = adc_lintrk.shower_frac;    

	if (track_frac > _min_track_frac) {
	  LARCV_INFO() << "Claiming a mostly track ADC contour as track" << std::endl;
	  //mask these pixels out of the ADC image
	  auto mask_adc = larocv::MaskImage(adc_img,adc_ctor,0,false);
	  //mask away these pixels in the track image
	  auto mask_track = larocv::MaskImage(track_img,adc_ctor,0,true);
	  //update the track image
	  track_img = mask_adc + mask_track;
	  //remove from shower image
	  shower_img = larocv::MaskImage(shower_img,adc_ctor,0,true);
	}

	if (shower_frac > _min_shower_frac and !adc_lintrk.ignore) {
	  LARCV_INFO() << "Claiming a mostly shower ADC contour as shower" << std::endl;
	  //mask these pixels out of the ADC image
	  auto mask_adc = larocv::MaskImage(adc_img,adc_ctor,0,false);
	  //mask away these pixels in the show image
	  auto mask_shower = larocv::MaskImage(shower_img,adc_ctor,0,true);
	  //reset the shower image
	  shower_img = mask_adc + mask_shower;
	  //remove from track
	  track_img = larocv::MaskImage(track_img,adc_ctor,0,true);
	}
      }

      //re-prepare the track and shower images
      track_img_t = PrepareImage(track_img);
      shower_img_t = PrepareImage(shower_img);

      //re-prepare the track and shower contours
      track_ctor_v = larocv::FindContours(track_img_t);
      shower_ctor_v = larocv::FindContours(shower_img_t);

      // Filter them by by number of contour points
      FilterContours(track_ctor_v);
      FilterContours(shower_ctor_v);
      LARCV_DEBUG() << "Re-Found " << track_ctor_v.size() << " track contours" << std::endl;
      LARCV_DEBUG() << "Re-Found " << shower_ctor_v.size() << " shower contours" << std::endl;
    }
	
    // Make extended linear tracks
    LARCV_DEBUG() << "Generating extended ADC linear tracks..." << std::endl;
    adc_lintrk_v = MakeLinearTracks(adc_ctor_v,adc_img_t,Type_t::kUnknown);
    LARCV_DEBUG() << "... made " << adc_lintrk_v.size() << " adc tracks " << std::endl;

    LARCV_DEBUG() << "Generating extended Track linear tracks..." << std::endl;
    track_lintrk_v = MakeLinearTracks(track_ctor_v,track_img_t,Type_t::kTrack);
    LARCV_DEBUG() << "... made " << track_lintrk_v.size() << " track tracks " << std::endl;

    LARCV_DEBUG() << "Generating extended Shower linear tracks..." << std::endl;
    shower_lintrk_v = MakeLinearTracks(shower_ctor_v,shower_img_t,Type_t::kShower);
    LARCV_DEBUG() << "... made " << shower_lintrk_v.size() << " shower tracks " << std::endl;
    
    std::vector<size_t> cidx_v;

    //Loop to determine track and shower compatibility
    for(size_t shower_id=0;shower_id<shower_lintrk_v.size();++shower_id) {
      LARCV_DEBUG() << "On shower " << shower_id << std::endl;
      const auto& shower = shower_lintrk_v[shower_id];

      for(size_t track1_id=0;track1_id<track_lintrk_v.size();++track1_id) {
	const auto& track1 = track_lintrk_v[track1_id];

	//Determine the straightness of this track and this shower
	if (track1.straight and shower.straight) {
	  LARCV_DEBUG() << "Straight shower and straight track identified" << std::endl;
	  auto angle = std::abs(geo2d::angle(shower.overallPCA,track1.overallPCA));
	  if ((angle < _min_overall_angle) or (angle > 180 - _min_overall_angle)) {
	    cidx_v.push_back(shower_id);
	    continue;
	  }
	}

	if (!EdgeConnected(shower,track1))
	  continue;

	// Find "sandwich" showers -- showers between two tracks    
	for(size_t track2_id=track1_id+1;track2_id<track_lintrk_v.size();++track2_id) {
	  const auto& track2 = track_lintrk_v[track2_id];

	  if (!EdgeConnected(shower,track2))
	    continue;
	  
	  LARCV_CRITICAL() << "Shower " << shower_id
			<< " sandwich btw"
			<< " track " << track1_id
			<< " & track " << track2_id
			<< std::endl;
	  
	  LARCV_CRITICAL() << "This shower of size " << shower.npixel
			   << " e1 " << shower.edge1 << " e2 " << shower.edge2
			   << "mean dist " << shower.mean_pixel_dist << std::endl;
	  LARCV_CRITICAL() << "Straight " << shower.straight << std::endl;

	  if (shower.straight)
	    cidx_v.push_back(shower_id);
	  
	} // end track2
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

    // Lets make sure the track and shower images do not gain
    // extra pixel value from masking
    adc_img.copyTo(track_img,larocv::Threshold(track_img,1,1));
    adc_img.copyTo(shower_img,larocv::Threshold(shower_img,1,1));
    
    LARCV_DEBUG() << "end" << std::endl;    
    return true;
  }
}

#endif
