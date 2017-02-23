#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImage.h"

namespace larcv {

  class LArbysImageAna : public ProcessBase {

  public:
    
    LArbysImageAna(const std::string name="LArbysImageAna");
    ~LArbysImageAna(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

    void SetManager(const::larocv::ImageClusterManager* icm) { _mgr_ptr = icm; }
    
  private:
    
    TTree* _event_tree;
    TTree* _vtx3d_tree;

    std::string _track_vertex_estimate_algo_name;

    const ::larocv::ImageClusterManager* _mgr_ptr;

    /// Clear vertex
    void ClearVertex();

    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;
    uint _entry;
    
    /// Vtx3d data
    uint _n_vtx3d;
    uint _vtx3d_n_planes;
    uint _vtx3d_type;
    
    double _vtx3d_x, _vtx3d_y, _vtx3d_z;

    std::vector<double> _vtx2d_x_v, _vtx2d_y_v;
    
    uint _vtx3d_id;

    std::vector<double> _circle_x_v,_circle_y_v;
    std::vector<uint>   _circle_xs_v;
    
  };

  class LArbysImageAnaProcessFactory : public ProcessFactoryBase {
  public:
    LArbysImageAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageAna",this); }
    ~LArbysImageAnaProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new LArbysImageAna(instance_name); }
  };

}

#endif
