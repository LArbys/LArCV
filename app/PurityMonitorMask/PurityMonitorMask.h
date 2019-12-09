#ifndef __LARCVAPP_PURITY_MONITOR_MASK_H__
#define __LARCVAPP_PURITY_MONITOR_MASK_H__

// cstdlib
#include <vector>

// larcv
#include "Processor/ProcessBase.h"
#include "DataFormat/Image2D.h"

// ROOT
#include "TTree.h"

namespace larcv {

  class PurityMonitorMask : public larcv::ProcessBase  {

  public: 

    PurityMonitorMask();
    virtual ~PurityMonitorMask();

    void configure(const PSet&);
    void initialize();
    bool process( IOManager& mgr );
    void finalize();

    bool process( const std::vector<larcv::Image2D>& adc_v, std::vector<larcv::Image2D>& masked_v );

    bool process( const std::vector<larcv::Image2D>& adc_v,
                  std::vector<larcv::Image2D>& masked_v,
                  const float threshold );
    
  protected:

    int _run;
    int _subrun;
    int _event;
    int _row;
    int _qsum;
    int _naboveth;

    TTree* _tree;

  };

}

#endif
