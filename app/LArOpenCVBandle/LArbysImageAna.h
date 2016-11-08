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
    
    TTree* _reco_tree;
    
    const ::larocv::ImageClusterManager* _mgr_ptr;

    void Clear();

    /// Unique event keys
    uint _run;
    uint _subrun;
    uint _event;
    
    /// HIP cluster vars
    std::vector<uint> _n_mip_ctors_v;
    std::vector<uint> _n_hip_ctors_v;

    /// Refine2D data
    uint _n_vtx3d;
    std::vector<double> _x_v, _y_v, _z_v;

    uint _n_circle_vtx;
    std::vector<std::vector<double> > _x_vv;
    std::vector<std::vector<double> > _y_vv;

    std::vector<uint> _vtx3d_n_planes_v;
    
    /// VertexTrackCluster
    uint _n_vtx_cluster;

    std::vector<uint> _num_planes_v;

    std::vector<std::vector<uint> >    _num_clusters_vv;
    std::vector<std::vector<uint> >    _num_pixels_vv;
    std::vector<std::vector<double> > _num_pixel_frac_vv;    

    /// Configuration pset
    std::string _hipcluster_name;
    std::string _defectcluster_name;
    std::string _pcacandidates_name;
    std::string _refine2dvertex_name;
    std::string _vertexcluster_name;
    
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

