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
#include "Base/larcv_logger.h"
#include "DataFormat/IOManager.h"

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

    const ::larcv::logger& logger() const { return _logger;}

    void set_config(const std::string cfg_file) {_config_file=cfg_file;}

  protected:

    ::larcv::logger _logger;
    ::larcv::IOManager _larcv_io;
    ::larcv::msg::Level_t _cropper_verbosity;
    ::larcv::msg::Level_t _mctree_verbosity;
    std::string _config_file;
    std::string _producer_wire;
    std::string _producer_gen;
    std::string _producer_mcreco;
    std::vector<size_t> _event_image_cols;
    std::vector<size_t> _event_image_rows;
    std::vector<size_t> _event_comp_rows;
    std::vector<size_t> _event_comp_cols;
    double _min_time;
    double _min_wire;

    //::larcv::supera::ImageExtractor<larlite::wire> _extractor;
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
