/**
 * \file MCInteraction.h
 *
 * \ingroup DataFormat
 * 
 * \brief Class def header for a class MCInteraction
 *
 * @author kazuhiro
 */

/** \addtogroup DataFormat

    @{*/
#ifndef MCINTERACTION_H
#define MCINTERACTION_H

#include <iostream>

namespace larcv {
  
  /**
     \class MCInteraction
     User defined class MCInteraction ... these comments are used to generate
     doxygen documentation!
  */
  class MCInteraction{
    
  public:
    
    /// Default constructor
    MCInteraction(){}
    
    /// Default destructor
    ~MCInteraction(){}
    
    MCIndex_t _index;
    InteractionID _id;
    std::vector<larcv::MCParticle> _mcpart_v;
    
  };
}
#endif
/** @} */ // end of doxygen group 

