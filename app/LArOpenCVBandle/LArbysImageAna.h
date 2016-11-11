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
    void Project3D(const larcv::ImageMeta& meta,
		   double _parent_x,double _parent_y,double _parent_z,uint plane,
		   double& xpixel, double& ypixel);
    
    TTree* _event_tree;
    TTree* _vtx3d_tree;
    
    const ::larocv::ImageClusterManager* _mgr_ptr;

    void ClearEvent();
    void ClearVertex();
    
    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;
    
    /// HIP cluster vars
    std::vector<uint> _n_mip_ctors_v;
    std::vector<uint> _n_hip_ctors_v;

    /// Refine2D data
    uint _n_vtx3d;
    double _vtx3d_x, _vtx3d_y, _vtx3d_z;
    std::vector<double> _vtx2d_x_v, _vtx2d_y_v;

    uint _vtx3d_id;
    
    std::vector<double> _circle_x_v;
    std::vector<double> _circle_y_v;

    uint _vtx3d_n_planes;
    
    /// VertexTrackCluster
    uint _n_vtx_cluster;

    uint _num_planes;

    std::vector<uint>    _num_clusters_v;
    std::vector<uint>    _num_pixels_v;
    std::vector<double> _num_pixel_frac_v;  

    std::vector<double> _circle_vtx_r_v;
    std::vector<double> _circle_vtx_angle_v;
        
    /// Configuration pset
    std::string _hipcluster_name;
    std::string _defectcluster_name;
    std::string _pcacandidates_name;
    std::string _refine2dvertex_name;
    std::string _vertexcluster_name;
    std::string _linearvtxfilter_name;
    std::string _dqdxprofiler_name;
    
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

