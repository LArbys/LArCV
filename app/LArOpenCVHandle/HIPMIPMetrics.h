/**
 * \file HIPMIPMetrics.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class HIPMIPMetrics
 *
 * @author dcianci
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __HIPMIPMETRICS_H__
#define __HIPMIPMETRICS_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class HIPMIPMetrics ... these comments are used to generate
     doxygen documentation!
  */
  class HIPMIPMetrics : public ProcessBase {

  public:
    
    /// Default constructor
    HIPMIPMetrics(const std::string name="HIPMIPMetrics");
    
    /// Default destructor
    ~HIPMIPMetrics(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

	private:
		int protonPixelCt[3], protonADCCt[3];
		
		TTree* _tree;
		int _run;
		int _subrun;
		int _event;
		int _entry;

		int	_totalProtonPix_plane0;
		int _totalProtonADC_plane0;
		int	_totalProtonPix_plane1;
		int _totalProtonADC_plane1;
		int	_totalProtonPix_plane2;
		int _totalProtonADC_plane2;

		int _particle_segment_id,	_particle_adc_threshold;

  };

  /**
     \class larcv::HIPMIPMetricsFactory
     \brief A concrete factory class for larcv::HIPMIPMetrics
  */
  class HIPMIPMetricsProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    HIPMIPMetricsProcessFactory() { ProcessFactory::get().add_factory("HIPMIPMetrics",this); }
    /// dtor
    ~HIPMIPMetricsProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new HIPMIPMetrics(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

