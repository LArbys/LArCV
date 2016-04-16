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

#include "PMTWeights/PMTWireWeights.h"

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

      std::string fGeoFile;
      larcv::pmtweights::PMTWireWeights* m_WireInfo; // this is used because it loads wire data for us (shouldfactor that portion out)
      
    protected:
      
      float cross2D( float a[], float b[] );

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

