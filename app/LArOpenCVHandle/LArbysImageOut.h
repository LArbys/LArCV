#ifndef __LARBYSIMAGEOUT_H__
#define __LARBYSIMAGEOUT_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "DataFormat/EventImage2D.h"

#include "/Users/vgenty/sw/larcv/core/DataFormat/Vertex.h"

#include "LArbysImage.h"

#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"
#include "LArOpenCV/ImageCluster/AlgoClass/VertexAnalysis.h"


namespace larcv {

  class LArbysImageOut : public ProcessBase {

  public:
    
    LArbysImageOut(const std::string name="LArbysImageOut");
    ~LArbysImageOut(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

    void SetManager(const::larocv::ImageClusterManager* icm)
    { _mgr_ptr = icm; }
    
  private:
    
    TTree* _event_tree;
    TTree* _vtx3d_tree;
    
    std::string _track_vertex_estimate_algo_name;

    const ::larocv::ImageClusterManager* _mgr_ptr;

    /// Clear vertex
    void ClearVertex();

    std::string _combined_vertex_name;
    
    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;
    uint _entry;
    
    // vertex3d data
    uint _n_vtx3d;
    uint _vtx3d_n_planes;
    uint _vtx3d_type;
    uint _vtx3d_id;
    uint _combined_particle_offset;
    double _vtx3d_x;
    double _vtx3d_y;
    double _vtx3d_z;
    std::vector<double> _vtx2d_x_v;
    std::vector<double> _vtx2d_y_v;
    std::vector<double> _circle_x_v;
    std::vector<double> _circle_y_v;
    std::vector<uint> _ntrack_par_v;
    std::vector<uint> _nshower_par_v;
    std::vector<uint> _circle_xs_v;
    std::vector<uint> _par_multi;


    /*
    //per vertex
    //per particle
    std::vector<std::vector<EventImage2D> > _particle_vv;
    //per vertex
    //per particle
    std::vector<std::vector<larcv::Vertex> > _particle_start_vv;
    std::vector<std::vector<larcv::Vertex> > _particle_end_vv;
    //per vertex
    //per particle
    //per plane
    std::vector<std::vector<std::vector<larcv::Vertex> > > _particle_start2d_vvv;
    std::vector<std::vector<std::vector<larcv::Vertex> > > _particle_end2d_vvv;
    */
    
    //multiple vertex per event
    std::vector<larocv::data::Vertex3D> _vertex3d_v;
    //per plane
    std::vector<std::vector<std::vector<larocv::data::ParticleCluster> > > _particle_cluster_vvv;
    //per vertex, per plane, multiple per plane
    std::vector<std::vector<std::vector<larocv::data::TrackClusterCompound> > > _track_compound_vvv;

    larocv::VertexAnalysis _vtx_ana;
    /*
      larocv::data::Vertex3DArray _vertex3d_array;
      std::vector<const larocv::data::ParticleClusterArray*> _particle_cluster_array_v;
      std::vector<const larocv::data::TrackClusterCompoundArray*> _track_cluster_compound_array_v;
      larocv::data::AlgoDataAssManager _ass_man;
    */
    
  };

  class LArbysImageOutProcessFactory : public ProcessFactoryBase {
  public:
    LArbysImageOutProcessFactory() { ProcessFactory::get().add_factory("LArbysImageOut",this); }
    ~LArbysImageOutProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new LArbysImageOut(instance_name); }
  };

}

#endif
