#ifndef __SSNETTRUTHANA_H__
#define __SSNETTRUTHANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larcv {

  class SSNetTruthAna : public ProcessBase {

  public:

   
    /// Default constructor
    SSNetTruthAna(const std::string name="SSNetTruthAna");
    
    /// Default destructor
    ~SSNetTruthAna(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();
    
  private:

    TTree* _tree;
    
    // Indices
    int _entry;
    int _run;
    int _subrun;
    int _event;
    
    std::string _ev_img2d_prod;
    std::string _ev_trk2d_prod;
    std::string _ev_shr2d_prod;

    std::string _roi_prod;
    std::string _true_roi_prod;
    size_t _crop_radius;

    larutil::SpaceChargeMicroBooNE _sce;

    float _trkpixel0;
    float _trkpixel1;
    float _trkpixel2;

    float _shrpixel0;
    float _shrpixel1;
    float _shrpixel2;

    float _npixel0;
    float _npixel1;
    float _npixel2;

    float _shr_ratiopixel0;
    float _shr_ratiopixel1;
    float _shr_ratiopixel2;

    float _trk_ratiopixel0;
    float _trk_ratiopixel1;
    float _trk_ratiopixel2;

    float _npixelsum;
    float _npixelavg;

    float _trkpixelsum;
    float _trkpixelavg;

    float _shrpixelsum;
    float _shrpixelavg;

    float _trk_ratiopixelsum;
    float _trk_ratiopixelavg;

    float _shr_ratiopixelsum;
    float _shr_ratiopixelavg;

    float _n_valid_planes;
    
  };

  /**
     \class larcv::SSNetTruthAnaFactory
     \brief A concrete factory class for larcv::SSNetTruthAna
  */
  class SSNetTruthAnaProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    SSNetTruthAnaProcessFactory() { ProcessFactory::get().add_factory("SSNetTruthAna",this); }
    /// dtor
    ~SSNetTruthAnaProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new SSNetTruthAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

