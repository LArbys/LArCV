#ifndef __LARCVAPP_PURITY_MONITOR_MASK_H__
#define __LARCVAPP_PURITY_MONITOR_MASK_H__

// cstdlib
#include <vector>

// larcv
#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/Pixel2DCluster.h"

// ROOT
#include "TTree.h"

namespace larcv {

  class PurityMonitorMask : public larcv::ProcessBase  {

  public: 

    PurityMonitorMask( std::string instance_name );
    virtual ~PurityMonitorMask();

    void configure(const PSet&);
    void initialize();
    bool process( IOManager& mgr );
    void finalize();

    bool process( const std::vector<larcv::Image2D>& adc_v, std::vector<larcv::Image2D>& masked_v );

    bool process( const std::vector<larcv::Image2D>& adc_v,
                  std::vector<larcv::Image2D>& masked_v,
                  std::vector<larcv::Pixel2DCluster>& pixel_v,
                  const float threshold );
    
  protected:

    std::string _name;
    
    int _run;
    int _subrun;
    int _event;
    int _row;
    float _qsum;
    int _naboveth;

    TTree* _tree;
    bool   _save_ana;

  };


  /**
     \class larcv::SliceImagesFactory
     \brief A concrete factory class for larcv::SliceImages
  */
  class PurityMonitorMaskProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    PurityMonitorMaskProcessFactory() { ProcessFactory::get().add_factory("PurityMonitorMask",this); }
    /// dtor
    ~PurityMonitorMaskProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new larcv::PurityMonitorMask(instance_name); }
  };
  
  
}

#endif
