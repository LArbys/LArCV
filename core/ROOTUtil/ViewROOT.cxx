#include "ViewROOT.h"

// ROOT
#include "TCanvas.h"
#include "TDirectory.h"
#include "TH2D.h"

// LARCV
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "ROOTUtil.h"

namespace larcv {

  ViewROOT*  ViewROOT::_global_instance = nullptr;
  
  ViewROOT::ViewROOT( std::string fname )
    : _init(false),
      _io(nullptr),
      _canv_nplanes(1),
      _canv(nullptr),
      _default_image2d(nullptr),
      _current_image2d("__default__"),
      _current_image2d_id(-1)
  {
    
    initialize(fname);
  };
  
  ViewROOT::~ViewROOT() {
    if ( !_init ) return;
    clearVisItems();
    delete _default_image2d;
    delete _canv;
    _io->reset();
    _io->finalize();
  }

  void ViewROOT::initialize( std::string fname ) {
    if ( _init ) return;
    if ( gDirectory->GetFile()==nullptr && fname=="" ) {
      std::cout << "[ViewROOT::initialize] no filename provided nor ROOT file loaded" << std::endl;
      return;
    }

    _default_image2d = new TH2D( "__larcv_view_root_default", "", 100, 0, 3456, 100, 2400, 8448 );
    _canv = new TCanvas( "larcvCanvas", "LArCV Canvas", 1200, 600 );

    _io = new IOManager( larcv::IOManager::kREAD );
    if ( fname!="" ) {
      _io->add_in_file( fname );
    }
    else if ( gDirectory->GetFile() ) {
      _io->add_in_file( gDirectory->GetFile()->GetName() );
    }
    _io->initialize();
  }

  void ViewROOT::clearVisItems() {
    _canv->Clear();
    for ( auto& p : _items )
      delete p.second.item;
    _items.clear();
  }

  void ViewROOT::setEntry( int entry ) {
    _io->read_entry( entry );
    // drawCanvas()
  }

  ViewROOT& ViewROOT::getViewROOT( std::string fname ) {
    if ( _global_instance==nullptr )
      _global_instance = new ViewROOT( fname );

    return *_global_instance;
  }

  void ViewROOT::draw( std::string datatype, std::string producer, int instance, std::string opt ) {

    std::stringstream ss;
    ss << datatype << "_" << producer << "_" << instance;
    std::string keyname = ss.str();

    // LARCV ITEMS
    if ( datatype=="image2d" ) {
      // image2d special -- it sets the items
      _current_image2d    = producer;
      if ( instance<0 )
	_current_image2d_id = 0;
      else
	_current_image2d_id = instance;

      auto ev_img = (larcv::EventImage2D*)_io->get_data( larcv::kProductImage2D, producer );
      
      TH2D* hist = new TH2D( larcv::as_th2d( ev_img->Image2DArray()[instance], keyname ) );
      VisInfo_t info;
      info.producer   = producer;
      info.datatype   = datatype;
      info.instanceid = instance;
      info.item       = hist;
      info._keyname   = keyname;
      
      _items[ keyname ] = info;
    }
  }

}
