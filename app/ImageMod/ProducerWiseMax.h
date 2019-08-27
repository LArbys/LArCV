/**
 * \file ProducerWiseMax.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class ProducerWiseMax
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __CHANNELMAX_H__
#define __CHANNELMAX_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
namespace larcv {

  /**
     \class ProcessBase
     
     Find Maximum element across Image2D sets
  */
  class ProducerWiseMax : public ProcessBase {

  public:
    
    /// Default constructor
    ProducerWiseMax(const std::string name="ProducerWiseMax");
    
    /// Default destructor
    ~ProducerWiseMax(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

  private:
    
    size_t _nproducers;
    std::vector<std::string> _in_producer_v;
    std::string _out_producer;
    std::vector<float> _producer_weight_v;
    std::vector<float> _producer_mask_v;
  };

  /**
     \class larcv::ProducerWiseMaxFactory
     \brief A concrete factory class for larcv::ProducerWiseMax
  */
  class ProducerWiseMaxProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    ProducerWiseMaxProcessFactory() { ProcessFactory::get().add_factory("ProducerWiseMax",this); }
    /// dtor
    ~ProducerWiseMaxProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new ProducerWiseMax(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

