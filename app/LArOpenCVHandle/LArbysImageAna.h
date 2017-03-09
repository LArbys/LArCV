#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImage.h"

#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"

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
    double _vtx3d_x;
    double _vtx3d_y;
    double _vtx3d_z;
    std::vector<double> _vtx2d_x_v;
    std::vector<double> _vtx2d_y_v;
    uint _vtx3d_id;
    std::vector<uint> _ntrack_par_v;
    std::vector<uint> _nshower_par_v;
    std::vector<double> _circle_x_v;
    std::vector<double> _circle_y_v;
    std::vector<uint> _circle_xs_v;
    std::vector<uint> _par_multi;
    std::string _combined_vertex_name;
    uint _combined_particle_offset;

    //multiple vertex per event
    std::vector<const larocv::data::Vertex3D*> _vertex3d_v;
    //per plane
    std::vector<std::vector<std::vector<const larocv::data::ParticleCluster*> > > _particle_cluster_vvv;
    //per vertex, per plane, multiple per plane
    std::vector<std::vector<std::vector<const larocv::data::TrackClusterCompound*> > > _track_compound_vvv;
    
  };

  class LArbysImageAnaProcessFactory : public ProcessFactoryBase {
  public:
    LArbysImageAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageAna",this); }
    ~LArbysImageAnaProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new LArbysImageAna(instance_name); }
  };

}

#endif
