/**
 * \file PIDImageMaker.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class PIDImageMaker
 *
 * @author ran
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __PIDIMAGEMAKER_H__
#define __PIDIMAGEMAKER_H__

#include "utility"
#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArbysImageMaker.h"
#include "LArOpenCV/ImageCluster/Base/ImageClusterManager.h"
#include "DataFormat/ImageMeta.h"
#include "DataFormat/EventROI.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventPixel2D.h"
#include "DataFormat/EventImage2D.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class PIDImageMaker ... these comments are used to generate
     doxygen documentation!
  */
  class PIDImageMaker : public ProcessBase {

  public:
    
    /// Default constructor
    PIDImageMaker(const std::string name="PIDImageMaker");
    
    /// Default destructor
    ~PIDImageMaker(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    //Function to get image from vertex reco for SPID
    void SPIDRecoImgFiller(std::map<larcv::PlaneID_t, std::vector<larcv::Pixel2DCluster>> ev_pcluster_array,
			   std::vector<larcv::Image2D>& p0_img_v,
			   std::vector<larcv::Image2D>& p1_img_v);

    //Function to get image from vertex reco for MPID
    void MPIDRecoImgFiller(std::map<larcv::PlaneID_t, std::vector<larcv::Pixel2DCluster>> ev_pcluster_array,
			   std::vector<larcv::Image2D>& p01_img_v);
    
    //Funttion to get empty image
    void VoidImgFiller(std::vector<larcv::Image2D>& p0_img_v,
		       std::vector<larcv::Image2D>& p1_img_v);
    //Function to get image from CROI form MPID
    void CROIImgFiller(ROI croi, std::vector<larcv::Image2D>& p01_img_v);
    
  private:

    size_t _nevents;
    size_t _nevents_passing_nueLL;

    std::pair<int, int> _outimage_dim;
    
    std::string _roi_input_producer;
    std::string _p0_roi_output_producer;
    std::string _p1_roi_output_producer;
    
    std::string _pgraph_producer;
    std::string _pixel2d_ctor_producer;
    std::string _pixel2d_img_producer;

    std::string _p0_image_producer;
    std::string _p1_image_producer;
    
    std::string _multi_image_producer;

    larocv::ImageClusterManager _alg_mgr;
    
    LArbysImageMaker _LArbysImageMaker;
  };

  /**
     \class larcv::PIDImageMakerFactory
     \brief A concrete factory class for larcv::PIDImageMaker
  */
  class PIDImageMakerProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    PIDImageMakerProcessFactory() { ProcessFactory::get().add_factory("PIDImageMaker",this); }
    /// dtor
    ~PIDImageMakerProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new PIDImageMaker(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

