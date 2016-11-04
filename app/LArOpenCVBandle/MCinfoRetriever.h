/**
 * \file MCinfoRetriever.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class MCinfoRetriever
 *
 * @author Rui
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __MCINFORETRIEVER_H__
#define __MCINFORETRIEVER_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include <vector>
#include "TH1D.h"
#include "TTree.h"

namespace larcv {

  /**
     \class ProcessBase
     User defined class MCinfoRetriever ... these comments are used to generate
     doxygen documentation!
  */
  class MCinfoRetriever : public ProcessBase {

  public:
    
    /// Default constructor
    MCinfoRetriever(const std::string name="MCinfoRetriever");
    
    /// Default destructor
    ~MCinfoRetriever(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    //  private:
  protected:
    std::string _producer;
    
    TTree* _mc_tree;
    int _run;
    int _subrun;
    int _event;
    
    int _parent_pdg;//primary particle pdg
    double _energy_deposit;
    double _parent_x;
    double _parent_y;  
    double _parent_z;  
    double _parent_t;  
    double _parent_px;
    double _parent_py;  
    double _parent_pz;  

    short _current_type;
    short _interaction_type;

  };

  /**
     \class larcv::MCinfoRetrieverFactory
     \brief A concrete factory class for larcv::MCinfoRetriever
  */
  class MCinfoRetrieverProcessFactory : public ProcessFactoryBase {
  public:
    /// ctor
    MCinfoRetrieverProcessFactory() { ProcessFactory::get().add_factory("MCinfoRetriever",this); }
    /// dtor
    ~MCinfoRetrieverProcessFactory() {}
    /// creation method
    ProcessBase* create(const std::string instance_name) { return new MCinfoRetriever(instance_name); }

  };

}

#endif
/** @} */ // end of doxygen group 

