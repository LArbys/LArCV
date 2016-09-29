/**
 * \file LArbysImage.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class LArbysImage
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __LARBYSIMAGE_H__
#define __LARBYSIMAGE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "Core/ImageManager.h"
#include "ImageCluster/Base/ImageClusterManager.h"
#include "ImageCluster/Base/ImageClusterViewer.h"
#include "DataFormat/user_info.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class LArbysImage ... these comments are used to generate
     doxygen documentation!
  */
  class LArbysImage : public ProcessBase {

  public:
    
    /// Default constructor
    LArbysImage(const std::string name="LArbysImage");
    
    /// Default destructor
    ~LArbysImage(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  protected:

    void store_clusters(IOManager& mgr);
    void extract_image(IOManager& mgr);

    TTree* _tree;
    ::larlite::event_user* _eui;
    
    ::larocv::ImageClusterManager _alg_mgr;
    ::larocv::ImageManager _img_mgr;

    bool   _debug;
    double _charge_to_gray_scale;
    double _charge_min;
    double _charge_max;
    std::vector<float> _plane_weights;
    std::string _producer;
    double _process_count;
    double _process_time_image_extraction;
    double _process_time_analyze;
    double _process_time_cluster_storage;
    void Report() const;
  };

  /**
     \class larcv::LArbysImageFactory
     \brief A concrete factory class for larcv::LArbysImage
  */
  class LArbysImageProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    LArbysImageProcessFactory() { ProcessFactory::get().add_factory("LArbysImage",this); }
    /// dtor
    ~LArbysImageProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new LArbysImage(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

