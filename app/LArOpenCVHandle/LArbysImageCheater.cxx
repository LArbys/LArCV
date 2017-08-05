#ifndef __LARBYSIMAGECHEATER_CXX__
#define __LARBYSIMAGECHEATER_CXX__

#include "LArbysImageCheater.h"
#include "LArOpenCV/ImageCluster/AlgoModule/VertexCheater.h"
#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"

namespace larcv {

  static LArbysImageCheaterProcessFactory __global_LArbysImageCheaterProcessFactory__;
  
  LArbysImageCheater::LArbysImageCheater(const std::string name) :
    LArbysImage(),
    _sce(),
    _mgr(nullptr)
  {}

  void LArbysImageCheater::SetIOManager(IOManager* mgr) {
    _mgr = mgr;
    LARCV_INFO() << "Set IOManager pointer as " << _mgr << std::endl;
  }
  
  bool LArbysImageCheater::Reconstruct(const std::vector<larcv::Image2D>& adc_image_v,
				       const std::vector<larcv::Image2D>& track_image_v,
				       const std::vector<larcv::Image2D>& shower_image_v,
				       const std::vector<larcv::Image2D>& thrumu_image_v,
				       const std::vector<larcv::Image2D>& stopmu_image_v,
				       const std::vector<larcv::Image2D>& chstat_image_v) {
    
    _adc_img_mgr.clear();
    _track_img_mgr.clear();
    _shower_img_mgr.clear();
    _thrumu_img_mgr.clear();
    _stopmu_img_mgr.clear();
    _chstat_img_mgr.clear();

    _alg_mgr.ClearData();
    
    larocv::Watch watch_all, watch_one;
    watch_all.Start();
    watch_one.Start();
    
    for(auto& img_data : _LArbysImageMaker.ExtractImage(adc_image_v)) {
      _adc_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				std::move(std::get<1>(img_data)));
    }

    if(!_track_producer.empty()) {
      for(auto& img_data : _LArbysImageMaker.ExtractImage(track_image_v))  { 
	_track_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				    std::move(std::get<1>(img_data)));
      }
    }
    
    if(!_shower_producer.empty()) {
      for(auto& img_data : _LArbysImageMaker.ExtractImage(shower_image_v)) {
	_shower_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				     std::move(std::get<1>(img_data)));
      }
    }

    if(!_stopmu_producer.empty()) { 
      for(auto& img_data : _LArbysImageMaker.ExtractImage(thrumu_image_v)) {
	_thrumu_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				     std::move(std::get<1>(img_data)));
      }
    }
    if(!_thrumu_producer.empty()) {
      for(auto& img_data : _LArbysImageMaker.ExtractImage(stopmu_image_v)) {
	_stopmu_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				     std::move(std::get<1>(img_data)));
      }
    }
    if(!_channel_producer.empty()) {
      for(auto& img_data : _LArbysImageMaker.ExtractImage(chstat_image_v)) {
	_chstat_img_mgr.emplace_back(std::move(std::get<0>(img_data)),
				     std::move(std::get<1>(img_data)));
      }
    }
    
    _process_time_image_extraction += watch_one.WallTime();

    for (size_t plane = 0; plane < _adc_img_mgr.size(); ++plane) {
      auto       & img  = _adc_img_mgr.img_at(plane);
      const auto & meta = _adc_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetWire);
    }

    for (size_t plane = 0; plane < _track_img_mgr.size(); ++plane) {
      auto       & img  = _track_img_mgr.img_at(plane);
      const auto & meta = _track_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetTrack);
    }
    
    for (size_t plane = 0; plane < _shower_img_mgr.size(); ++plane) {
      auto       & img  = _shower_img_mgr.img_at(plane);
      const auto & meta = _shower_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetShower);
    }

    for (size_t plane = 0; plane < _thrumu_img_mgr.size(); ++plane) {
      auto       & img  = _thrumu_img_mgr.img_at(plane);
      const auto & meta = _thrumu_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetThruMu);
    }

    for (size_t plane = 0; plane < _stopmu_img_mgr.size(); ++plane) {
      auto       & img  = _stopmu_img_mgr.img_at(plane);
      const auto & meta = _stopmu_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetStopMu);
    }

    for (size_t plane = 0; plane < _chstat_img_mgr.size(); ++plane) {
      auto       & img  = _chstat_img_mgr.img_at(plane);
      const auto & meta = _chstat_img_mgr.meta_at(plane);
      if (!meta.num_pixel_row() || !meta.num_pixel_column()) continue;
      _alg_mgr.Add(img, meta, larocv::ImageSetID_t::kImageSetChStatus);
    }

    if (_preprocess) {
      // give a single plane @ a time to pre processor
      auto& adc_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetWire);
      auto& trk_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetTrack);
      auto& shr_img_v= _alg_mgr.InputImages(larocv::ImageSetID_t::kImageSetShower);
      auto nplanes = adc_img_v.size();
      for(size_t plane_id=0;plane_id<nplanes;++plane_id) {
	LARCV_DEBUG() << "Preprocess image set @ "<< " plane " << plane_id << std::endl;
	if (!_PreProcessor.PreProcess(adc_img_v[plane_id],trk_img_v[plane_id],shr_img_v[plane_id])) {
	  LARCV_CRITICAL() << "... could not be preprocessed, abort!" << std::endl;
	  throw larbys();
	}
      }
    }

    // Cheat the first algo by extracting the truth vertex position
    std::string cheater_alg_name = "vertexcheater";
    LARCV_DEBUG() << "Set cheater algo name: " << cheater_alg_name << std::endl;

    auto cheater_id = _alg_mgr.GetClusterAlgID(cheater_alg_name);
    auto cheater_alg = (larocv::VertexCheater*)_alg_mgr.GetClusterAlgRW(cheater_id);

    if (!cheater_alg) throw larbys("Could not find RW cheater algo");
    if (!_mgr) throw larbys("No IOManager pointer specified");
    
    auto true_roi_producer = _true_prod;
    LARCV_DEBUG() << "Set true ROI producer name: " << true_roi_producer << std::endl;
    
    auto ev_roi = (EventROI*)_mgr->get_data(kProductROI,true_roi_producer);
    
    if(!ev_roi) throw larbys("Could not read given true ROI producer");
    if(ev_roi->ROIArray().empty()) throw larbys("Could not find given true ROI producer");

    const auto& roi = ev_roi->ROIArray().front();
    
    auto tx = roi.X();
    auto ty = roi.Y();
    auto tz = roi.Z();

    LARCV_DEBUG() << "Read (x,y,z)=("<<tx<<","<<ty<<","<<tz<<")"<<std::endl;
    
    const auto offset = _sce.GetPosOffsets(tx,ty,tz);
    
    tx = tx - offset[0] + 0.7;
    ty = ty + offset[1];
    tz = tz + offset[2];

    LARCV_DEBUG() << "SCE (x,y,z)=("<<tx<<","<<ty<<","<<tz<<")"<<std::endl;
    
    larocv::data::Vertex3D true_vertex;
    true_vertex.x = tx;
    true_vertex.y = ty;
    true_vertex.z = tz;
    
    cheater_alg->SetTrueVertex(true_vertex);

    _alg_mgr.Process();

    watch_one.Start();
    
    _process_time_cluster_storage += watch_one.WallTime();

    _process_time_analyze += watch_all.WallTime();
    
    ++_process_count;

    return true;
  }

  void LArbysImageCheater::SetTrueROIProducer(const std::string& true_prod) {
    _true_prod = true_prod;
  }
  
}

#endif
