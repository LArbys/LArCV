#ifndef __SEGWEIGHTINSTANCE_CXX__
#define __SEGWEIGHTINSTANCE_CXX__

#include "SegWeightInstance.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventPixel2D.h"

namespace larcv {

  static SegWeightInstanceProcessFactory __global_SegWeightInstanceProcessFactory__;

  SegWeightInstance::SegWeightInstance(const std::string name)
    : ProcessBase(name)
  {}
    
  void SegWeightInstance::configure(const PSet& cfg)
  {

    _instance_image_producer = cfg.get<std::string>("InstanceProducer");

    _keypt_pixel2d_producer = cfg.get<std::string>("KeyPointProducer","");
    
    _weight_producer = cfg.get<std::string>("WeightProducer");

    _plane_v = cfg.get<std::vector<size_t> >("PlaneID");

    _weight_max = cfg.get<unsigned int>("WeightMax");

    if(_instance_image_producer.empty()) {
      LARCV_CRITICAL() << "Label producer empty!" << std::endl;
      throw larbys();
    }

    if(_weight_producer.empty()) {
      LARCV_CRITICAL() << "Weight producer empty!" << std::endl;
      throw larbys();
    }

    if(_plane_v.empty()) {
      LARCV_CRITICAL() << "No plane ID requested for process" << std::endl;
      throw larbys();
    }

    _pool_type = (PoolType_t)(cfg.get<unsigned short>("PoolType",(unsigned short)kSumPool));

    _weight_surrounding = cfg.get<bool>("WeightSurrounding",true);

    _weight_vertex = cfg.get<bool>("WeightVertex",false);

    _dist_surrounding = cfg.get<size_t>("DistSurrounding",1);

    if(_weight_surrounding && !_dist_surrounding) {
      LARCV_CRITICAL() << "You want to weight surrounding pixels but did not provide DistSurrounding!" << std::endl;
      throw larbys();
    }

    if(_weight_vertex) 
      LARCV_WARNING() << "Vertex weighting is not supported yet." << std::endl;

  }

  void SegWeightInstance::initialize()
  {}

  bool SegWeightInstance::process(IOManager& mgr)
  {
    auto const& instance_image_v = ((EventImage2D*)(mgr.get_data(kProductImage2D,_instance_image_producer)))->Image2DArray();

    auto event_weight_v = ((EventImage2D*)(mgr.get_data(kProductImage2D,_weight_producer)));

    std::vector<larcv::Image2D> weight_image_v;
    event_weight_v->Move(weight_image_v);

    // make sure plane id is available
    for(auto const& plane : _plane_v) {
      if(plane < instance_image_v.size()) continue;
      LARCV_CRITICAL() << "Plane ID " << plane << " not found!" << std::endl;
      throw larbys();
    }

    std::vector<float> temp_weight_data;
    std::vector<float> boundary_data;
    for(auto const& plane : _plane_v) {
      
      weight_image_v.resize(plane+1);
      auto& weight_image = weight_image_v[plane];
      auto const& instance_image = instance_image_v[plane];
      auto const& label_data  = instance_image.as_vector();

      if(!weight_image.as_vector().empty()) {
	// image already exists. check the dimension
	if( weight_image.meta().min_x() != instance_image.meta().min_x() ||
	    weight_image.meta().max_y() != instance_image.meta().max_y() ||
	    weight_image.meta().cols()  != instance_image.meta().cols()  ||
	    weight_image.meta().rows()  != instance_image.meta().rows() ) {
	  LARCV_CRITICAL() << "Plane " << plane << "has incompatible label/weight image meta" << std::endl;
	  throw larbys();
	}
      }else{
	// create weight image
	weight_image = larcv::Image2D(instance_image.meta());
	weight_image.paint(1.);
      }

      //
      // compute # instance pixels
      //
      std::vector<float> weight_v;
      for(auto const& v : instance_image.as_vector()) {
	size_t instance = (size_t)(v);
	if(instance<1) continue;
	if(weight_v.size()<instance) weight_v.resize(instance,0.);
	weight_v[instance-1] += 1.;
      }

      for(size_t idx=0; idx<weight_v.size(); ++idx) {
	auto& value = weight_v[idx];
	if(value<1)
	  value = 1.;
	else
	  value = (float)(instance_image.as_vector().size()) / value;
	LARCV_INFO() << "Instance " << idx << " ...weight " << value << std::endl;
      }

      //
      // Initialize 
      //
      auto const& weight_data = weight_image.as_vector();
      temp_weight_data.resize(weight_data.size());
      boundary_data.resize(weight_data.size());
      
      for(size_t idx=0; idx<weight_data.size(); ++idx)
	temp_weight_data[idx] = boundary_data[idx] = 0.;

      //
      // Compute surrounding pixels
      //
      for(size_t idx=0; idx<weight_data.size(); ++idx) {
	size_t instance = (size_t)(label_data[idx]);
	if(instance<1) continue;
	temp_weight_data[idx] = weight_v[instance-1];
      }
      
      float surrounding_weight=0;
      if(_weight_surrounding && _dist_surrounding) {
	int target_row, target_col;
	int nrows = weight_image.meta().rows();
	int ncols = weight_image.meta().cols();
	size_t n_surrounding_pixel = 0;
	bool flag = false;
	auto const& label_meta = instance_image.meta();
	for(int row=0; row<nrows; ++row) {
	  for(int col=0; col<ncols; ++col) {
	    size_t instance = (size_t)(instance_image.pixel(row,col));
	    if(instance)
	      continue;
	    flag = false;
	    for(target_row = (int)(row+_dist_surrounding); (int)(target_row + _dist_surrounding) >= row; --target_row) {
	      if(target_row >= nrows) continue;
	      if(target_row < 0) break;
	      for(target_col = (int)(col+_dist_surrounding); (int)(target_col + _dist_surrounding) >= col; --target_col) {
		if(target_col >= ncols) continue;
		if(target_col < 0) break;
		instance = (size_t)(instance_image.pixel(target_row,target_col));
		if(instance) {
		  flag=true;
		  break;
		}
	      }
	      if(flag) break;
	    }
	    boundary_data[label_meta.index(row,col)] = (flag ? 1. : 0.);
	    if(flag) ++n_surrounding_pixel;
	  }
	}

	if(n_surrounding_pixel)
	  surrounding_weight = (int)((float)(label_data.size()) / (float)(n_surrounding_pixel));

	LARCV_INFO() << "Found " << n_surrounding_pixel << " pixels around interesting ones..." << std::endl;
	LARCV_INFO() << "Weight: " << surrounding_weight << std::endl;

	for(size_t idx=0; idx < weight_data.size(); ++idx) 
	  temp_weight_data[idx] += (boundary_data[idx] * surrounding_weight);
      }

      //
      // Compute keypoint weghts
      //
      if(!_keypt_pixel2d_producer.empty()) {
	auto ev_pixel = ((EventPixel2D*)(mgr.get_data(kProductPixel2D,_keypt_pixel2d_producer)));
	auto const& pcluster_m = ev_pixel->Pixel2DClusterArray();
	auto const& meta_m     = ev_pixel->ClusterMetaArray();

	auto meta_iter = meta_m.find(plane);
	if(meta_iter == meta_m.end()) {
	  LARCV_CRITICAL() << "Plane " << plane << " not found in ClusterMetaArray()!" << std::endl;
	  throw larbys();
	}
	
	auto clus_iter = pcluster_m.find(plane);
	if(clus_iter==pcluster_m.end()) {
	  LARCV_CRITICAL() << "Plane " << plane << " not found in Pixel2DClusterArray()!" << std::endl;
	  throw larbys();
	}

	auto const& ref_meta   = instance_image.meta();
	auto const& ref_data   = instance_image.as_vector();
	auto const& pcluster_v = (*clus_iter).second;
	auto const& meta_v     = (*meta_iter).second;

	// sanity check
	if(meta_v.size() != pcluster_v.size() ) {
	  LARCV_CRITICAL() << "# cluster meta and # cluster does not match!" << std::endl;
	  throw larbys();
	}
	for(auto const& meta : meta_v) {
	  if(meta == ref_meta) continue;
	  LARCV_CRITICAL() << "Found non-compatible image meta!" << std::endl
			   << "      Cluster: " << meta.dump()
			   << "      Ref    : " << ref_meta.dump();
	  throw larbys();
	}
      
	for(size_t cluster_idx=0; cluster_idx<pcluster_v.size(); ++cluster_idx){
	  auto const& pcluster = pcluster_v[cluster_idx];
	  auto const& meta     = meta_v[cluster_idx];
	  for(auto const& pixel2d : pcluster) {
	    size_t index = (pixel2d.X() * ref_meta.rows() + pixel2d.Y());
	    if(ref_data[index]>0) temp_weight_data[index] += _weight_max;
	  }
	}
      }

      //
      // Combine weights
      //
      float v=0;
      switch(_pool_type) {
      case kSumPool:
	for(size_t idx=0; idx<weight_data.size(); ++idx) {
	  v = weight_data[idx] + temp_weight_data[idx];
	  if(v > (float)(_weight_max)) v = _weight_max;
	  weight_image.set_pixel(idx,v);
	}
	break;
      case kMaxPool:
	for(size_t idx=0; idx<weight_data.size(); ++idx) {
	  v = std::max(weight_data[idx],temp_weight_data[idx]);
	  if(v > (float)(_weight_max)) v = _weight_max;
	  weight_image.set_pixel(idx,v);
	}
	break;
      case kAveragePool:
	for(size_t idx=0; idx<weight_data.size(); ++idx) {
	  v = (weight_data[idx] + temp_weight_data[idx])/2.;
	  if(v > (float)(_weight_max)) v = _weight_max;
	  weight_image.set_pixel(idx,v);
	}
	break;
      case kOverwrite:
	for(size_t idx=0; idx<weight_data.size(); ++idx) {
	  v = temp_weight_data[idx];
	  if(v > (float)(_weight_max)) v = _weight_max;
	  weight_image.set_pixel(idx,v);
	}
	break;
      }

    }
    
    event_weight_v->Emplace(std::move(weight_image_v));
    return true;
  }

  void SegWeightInstance::finalize()
  {}

}
#endif
