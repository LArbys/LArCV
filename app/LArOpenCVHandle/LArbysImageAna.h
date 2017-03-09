/**
 * \file LArbysImageAna.h
 *
 * \ingroup LArOpenCVHandle
 * 
 * \brief Class def header for a class LArbysImageAna
 *
 * @author vgenty
 */

/** \addtogroup LArOpenCVHandle

    @{*/

#ifndef LARLITE_LARBYSIMAGEANA_H
#define LARLITE_LARBYSIMAGEANA_H

#include "Analysis/ana_base.h"

namespace larlite {
  /**
     \class LArbysImageAna
     User custom analysis class made by SHELL_USER_NAME
   */
  class LArbysImageAna : public ana_base{
  
  public:

    /// Default constructor
    LArbysImageAna(){ _name="LArbysImageAna"; _fout=0;}

    /// Default destructor
    virtual ~LArbysImageAna(){}

    /** IMPLEMENT in LArbysImageAna.cc!
        Initialization method to be called before the analysis event loop.
    */ 
    virtual bool initialize();

    /** IMPLEMENT in LArbysImageAna.cc! 
        Analyze a data event-by-event  
    */
    virtual bool analyze(storage_manager* storage);

    /** IMPLEMENT in LArbysImageAna.cc! 
        Finalize method to be called after all events processed.
    */
    virtual bool finalize();

  protected:
    
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
