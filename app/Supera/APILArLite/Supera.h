/**
 * \file Supera.h
 *
 * \ingroup APILArLite
 * 
 * \brief Class def header for a class Supera
 *
 * @author kazuhiro
 */

/** \addtogroup APILArLite

    @{*/

#ifndef LARLITE_SUPERA_H
#define LARLITE_SUPERA_H

#include "Analysis/ana_base.h"
#include "DataFormat/opdetwaveform.h"
#include "DataFormat/wire.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/simch.h"
#include "SuperaCore.h"
namespace larlite {
  /**
     \class Supera
     User custom analysis class made by SHELL_USER_NAME
   */
  class Supera : public ana_base{
  
  public:

    /// Default constructor
    Supera();

    /// Default destructor
    virtual ~Supera(){}

    /** IMPLEMENT in Supera.cc!
        Initialization method to be called before the analysis event loop.
    */ 
    virtual bool initialize();

    /** IMPLEMENT in Supera.cc! 
        Analyze a data event-by-event  
    */
    virtual bool analyze(storage_manager* storage);

    /** IMPLEMENT in Supera.cc! 
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();

    void set_config(const std::string cfg_file) {_config_file=cfg_file;}

    void supera_fname(std::string name) { _core.supera_fname(name); }

  protected:

    ::larcv::supera::SuperaCore<larlite::opdetwaveform, larlite::wire,
				larlite::mctruth,
				larlite::mctrack, larlite::mcshower,
				larlite::simch> _core;
    std::string _config_file;
    
  };
}

//template class larcv::supera::ImageExtractor<larlite::wire>;

#endif

//**************************************************************************
// 
// For Analysis framework documentation, read Manual.pdf here:
//
// http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
//
//**************************************************************************

/** @} */ // end of doxygen group 
