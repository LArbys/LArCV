/**
 * \file LArbysImageAna.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class LArbysImageAna
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __LARBYSIMAGEANA_H__
#define __LARBYSIMAGEANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImage.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class LArbysImageAna ... these comments are used to generate
     doxygen documentation!
  */
  class LArbysImageAna : public ProcessBase {

  public:
    
    /// Default constructor
    LArbysImageAna(const std::string name="LArbysImageAna");
    
    /// Default destructor
    ~LArbysImageAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);
    
    void finalize();
    
    void SetManager(const::larocv::ImageClusterManager* icm) { _mgr_ptr = icm; }
    
  private:

    TTree* _event_tree;

    TTree* _hip_event_tree;
    TTree* _hip_plane_tree;
    
    TTree* _defect_event_tree;
    TTree* _defect_defect_tree;
    
    TTree* _vtx3d_tree;
    TTree* _particle_tree;

    TTree* _track_tree;
    TTree* _shower_tree;
    
    const ::larocv::ImageClusterManager* _mgr_ptr;

    void ClearEvent();
    void ClearHIPCluster();
    void ClearDefect();
    void ClearVertex();
    void ClearParticle();
    void ClearTracks();
    void ClearShowers();
    
    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;

    
    
    /// HIP cluster vars
    uint _hip_cluster_plane;
    
    uint _hip_per_plane;
    uint _num_hip_plane;
    uint _num_mip_plane;
    
    uint _num_mips;
    uint _num_hips;

    std::vector<uint> _npx_v;
    std::vector<float> _q_sum_v;
    std::vector<float> _q_avg_v;
    std::vector<uint> _is_hip_v;

    float _long_hip_length;
    float _long_mip_length;

    float _avg_long_hip_length;
    float _avg_long_mip_length;
    
    /// Defect cluster data -- per r/s/e & plane & defect

    uint    _defect_n_defects;
    size_t  _defect_id;
    uint    _defect_plane_id;
    double  _defect_dist_start_end;
    double  _defect_dist;
    uint    _defect_n_atomics;

    std::vector<float> _defect_atomic_len_v; //per atomic // per plane
    std::vector<float> _defect_atomic_qsum_v; // charge sum
    std::vector<float> _defect_atomic_npts_v; // number of points in atomic contour
    std::vector<float> _defect_atomic_qavg_v; // average of charge in cluster
    
    /// Refine2D data
    uint _n_vtx3d;
    double _vtx3d_x, _vtx3d_y, _vtx3d_z;
    std::vector<double> _vtx2d_x_v, _vtx2d_y_v;

    uint _vtx3d_id;
    
    std::vector<double> _circle_x_v;
    std::vector<double> _circle_y_v;
    std::vector<uint>   _circle_xs_v;

    uint _vtx3d_n_planes;

    uint _vtx3d_type;
    
    /// VertexTrackCluster
    uint _n_vtx_cluster;

    uint _num_planes;

    std::vector<uint>    _num_clusters_v;
    std::vector<uint>    _num_pixels_v;
    std::vector<double>  _num_pixel_frac_v;

    double   _sum_pixel_frac;
    double   _prod_pixel_frac;

    std::vector<double> _circle_vtx_r_v;
    std::vector<double> _circle_vtx_angle_v;
    
    //dQdXProfilerAlgo
    uint _plane_id;
    uint _n_pars;

    std::vector<double> _qsum_v;
    std::vector<uint> _npix_v;
    
    std::vector<uint>   _num_atoms_v;
    std::vector<double> _start_x_v;
    std::vector<double> _start_y_v;
    std::vector<double> _end_x_v;
    std::vector<double> _end_y_v;
    std::vector<double> _start_end_length_v;
    std::vector<double> _atom_sum_length_v;
    std::vector<double> _first_atom_cos_v;

    std::vector< std::vector<float> > _dqdx_vv;
    std::vector< std::vector<uint> > _dqdx_start_idx_vv;
    
    /// Configuration pset
    std::string _hipcluster_name;
    std::string _defectcluster_name;
    std::string _pcacandidates_name;
    std::string _refine2dvertex_name;
    std::string _vertexcluster_name;
    std::string _linearvtxfilter_name;
    std::string _dqdxprofiler_name;

    std::string _lineartrackcluster_name;
    std::string _vertexsingleshower_name;

    /// LinearTrackCluster
    uint _n_trackclusters;

    std::vector<float> _edge2D_1_x_v;
    std::vector<float> _edge2D_1_y_v;
    std::vector<float> _edge2D_2_x_v;
    std::vector<float> _edge2D_2_y_v;

    //VertexSignleShower
    uint _n_showerclusters;
    
    uint _shower_id;
    uint _shower_ass_id;
    uint _shower_ass_type;

    float _shower_vtx3D_x;
    float _shower_vtx3D_y;   
    float _shower_vtx3D_z;
    
    std::vector<float>_start2D_x_v;
    std::vector<float> _start2D_y_v;
    
    std::vector<float> _dir2D_x_v;
    std::vector<float> _dir2D_y_v;
    
  };

  /**
     \class larcv::LArbysImageAnaFactory
     \brief A concrete factory class for larcv::LArbysImageAna
  */
  class LArbysImageAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageAnaProcessFactory() { ProcessFactory::get().add_factory("LArbysImageAna",this); }
    /// dtor
    ~LArbysImageAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImageAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 


