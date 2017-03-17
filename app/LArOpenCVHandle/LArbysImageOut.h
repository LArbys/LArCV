#ifndef __LARBYSIMAGEOUT_H__
#define __LARBYSIMAGEOUT_H__

#include <TTree.h>
#include "LArbysImageAnaBase.h"

#include "DataFormat/EventImage2D.h"
#include "DataFormat/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/Vertex.h"
#include "LArOpenCV/ImageCluster/AlgoData/ParticleCluster.h"
#include "LArOpenCV/ImageCluster/AlgoData/TrackClusterCompound.h"
#include "LArOpenCV/ImageCluster/AlgoClass/VertexAnalysis.h"

#include "/Users/vgenty/sw/larcv/core/DataFormat/Vertex.h"

namespace larcv {

  class LArbysImageOut : public LArbysImageAnaBase {

  public:
    
    LArbysImageOut(const std::string name="LArbysImageOut");
    ~LArbysImageOut(){}

    void Configure(const PSet&);
    void Initialize();
    bool Analyze(larocv::ImageClusterManager& mgr);
    void Finalize(TFile* fout=nullptr);

  private:

    TTree* _event_tree;
    TTree* _vtx3d_tree;
    
    std::string _track_vertex_estimate_algo_name;

    /// Clear vertex
    void ClearVertex();

    /// Vtx3d data
    std::string _combined_vertex_name;

    // store info
    bool _analyze;
    
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

    ///*
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
    //*/
    
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

}

#endif
