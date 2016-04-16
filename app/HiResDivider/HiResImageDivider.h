/**
 * \file HiResImageDivider.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class HiResImageDivider
 *
 * @author twongjirad
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __HIRESIMAGEDIVIDER_H__
#define __HIRESIMAGEDIVIDER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include <string>
#include <vector>

#include "DataFormat/ImageMeta.h"
#include "DataFormat/ROI.h"
#include "PMTWeights/PMTWireWeights.h"
#include "DivisionDef.h"

namespace larcv {
  namespace hires {
    /**
       \class ProcessBase
       User defined class HiResImageDivider ... these comments are used to generate
       doxygen documentation!
    */
    class HiResImageDivider : public ProcessBase {
      
    public:
      
      /// Default constructor
      HiResImageDivider(const std::string name="HiResImageDivider");
      
      /// Default destructor
      ~HiResImageDivider(){}
      
      void configure(const PSet&);
      
      void initialize();
      
      bool process(IOManager& mgr);
      
      void finalize(TFile* ana_file);
      
    protected:

      std::string fDivisionFile;
      int fNPlanes;
      int fTickStart;
      int fTickDownSample;
      int fMaxWireImageWidth;
      int fMaxWireInRegion;
      std::string fInputImageProducer;
      std::string fInputROIProducer;
      std::vector< larcv::hires::DivisionDef > m_divisions;

      bool decideToContinueBasedOnROI( const larcv::ROI& roi );
      int findVertexDivisionUsingROI( const larcv::ROI& roi );
      bool decideToKeepBasedOnROI( const larcv::ROI& roi );
      
    };
    
    /**
     \class larcv::HiResImageDividerFactory
     \brief A concrete factory class for larcv::HiResImageDivider
    */
    class HiResImageDividerProcessFactory : public ProcessFactoryBase {
    public:
      /// ctor
      HiResImageDividerProcessFactory() { ProcessFactory::get().add_factory("HiResImageDivider",this); }
      /// dtor
      ~HiResImageDividerProcessFactory() {}
      /// creation method
      ProcessBase* create(const std::string instance_name) { return new HiResImageDivider(instance_name); }
    };
  }
}

#endif
/** @} */ // end of doxygen group 

