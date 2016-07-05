/**
 * \file ChStatusFinder.h
 *
 * \ingroup APILArLite
 * 
 * \brief Class def header for a class ChStatusFinder
 *
 * @author kterao
 */

/** \addtogroup APILArLite

    @{*/

#ifndef LARLITE_CHSTATUSFINDER_H
#define LARLITE_CHSTATUSFINDER_H

#include "Analysis/ana_base.h"
#include "DataFormat/IOManager.h"
namespace larlite {
  /**
     \class ChStatusFinder
     User custom analysis class made by SHELL_USER_NAME
   */
  class ChStatusFinder : public ana_base{
  
  public:

    /// Default constructor
    ChStatusFinder() : _io(larcv::IOManager::kREAD) { _name="ChStatusFinder"; _fout=0;}

    /// Default destructor
    virtual ~ChStatusFinder(){}

    /** IMPLEMENT in ChStatusFinder.cc!
        Initialization method to be called before the analysis event loop.
    */ 
    virtual bool initialize();

    /** IMPLEMENT in ChStatusFinder.cc! 
        Analyze a data event-by-event  
    */
    virtual bool analyze(storage_manager* storage);

    /** IMPLEMENT in ChStatusFinder.cc! 
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();

    size_t _num_entries;
    std::string _in_producer;
    std::string _out_producer;
    larcv::IOManager _io;
  };
}
#endif

//**************************************************************************
// 
// For Analysis framework documentation, read Manual.pdf here:
//
// http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
//
//**************************************************************************

/** @} */ // end of doxygen group 
