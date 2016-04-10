/**
 * \file SuperaCore.h
 *
 * \ingroup APILArLite
 * 
 * \brief Class def header for a class SuperaCore
 *
 * @author kazuhiro
 */

/** \addtogroup APILArLite

    @{*/
#ifndef __SUPERACORE_H__
#define __SUPERACORE_H__

#include "Base/larcv_logger.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ChStatus.h"
#include "MCParticleTree.h"
#include "FMWKInterface.h"

namespace larcv {
  namespace supera {
    /**
       \class SuperaCore
       User defined class SuperaCore ... these comments are used to generate
       doxygen documentation!
    */
    template<class S, class T, class U, class V, class W>
    class SuperaCore{
      
    public:
      
      /// Default constructor
      SuperaCore();
      
      /// Default destructor
      ~SuperaCore(){}

      void configure(const Config_t& cfg);

      void initialize();

      void set_id(unsigned int run, unsigned int subrun, unsigned int event)
      { _larcv_io.set_id(run, subrun, event); }

      bool process_event(const std::vector<S>&,  // Wire
			 const std::vector<T>&,  // MCTruth
			 const std::vector<U>&,  // MCTrack
			 const std::vector<V>&,  // MCShower
			 const std::vector<W>&); // SimCh

      void finalize();

      void set_chstatus(unsigned int ch, short status);
      const std::string& producer_digit     () const { return _producer_digit;  }
      const std::string& producer_wire      () const { return _producer_wire;   }
      const std::string& producer_generator () const { return _producer_gen;    }
      const std::string& producer_mcreco    () const { return _producer_mcreco; }
      const std::string& producer_simch     () const { return _producer_simch;  }
      bool use_mc() const { return _use_mc; }
      bool store_chstatus() const { return _store_chstatus; }
      const ::larcv::logger& logger() const { return _logger;}

    private:

      void fill(Image2D& img, const std::vector<S>& wires, const int time_offset=0);
      
      MCParticleTree<T,U,V,W> _mctp;
      std::map<larcv::PlaneID_t,larcv::ChStatus> _status_m;
      std::vector<std::pair<unsigned short,unsigned short> > _channel_to_plane_wire;
      larcv::logger _logger;
      larcv::IOManager _larcv_io;
      std::string _producer_key;
      std::string _producer_digit;
      std::string _producer_simch;
      std::string _producer_wire;
      std::string _producer_gen;
      std::string _producer_mcreco;
      std::vector<size_t> _event_image_cols;
      std::vector<size_t> _event_image_rows;
      std::vector<size_t> _event_comp_rows;
      std::vector<size_t> _event_comp_cols;
      double _min_time;
      double _min_wire;
      larcv::Image2D _full_image;
      bool _skip_empty_image;
      bool _configured;
      bool _use_mc;
      bool _store_chstatus;
    };
  }
}

#include "SuperaCore.inl"

#endif
/** @} */ // end of doxygen group 

