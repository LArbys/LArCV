#ifndef __MCNUSTREAM_CXX__
#define __MCNUSTREAM_CXX__

#include "MCNuStream.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventROI.h"
namespace larcv {

  static MCNuStreamProcessFactory __global_MCNuStreamProcessFactory__;

  MCNuStream::MCNuStream(const std::string name)
    : ImageHolder(name)
  {}
    
  void MCNuStream::configure(const PSet& cfg)
  {
    _tpc_image_producer = cfg.get<std::string> ( "TPCImageProducer" );
    _pmt_image_producer = cfg.get<std::string> ( "PMTImageProducer" );
    _segment_producer   = cfg.get<std::string> ( "SegmentProducer"  );
    _roi_producer       = cfg.get<std::string> ( "ROIProducer"      );

    _min_energy_deposit = cfg.get<double>      ( "MinEnergyDeposit" );
    _min_energy_init = cfg.get<double>( "MinEnergyInit" );
    _min_width       = cfg.get<double>( "MinWidth"      );
    _min_height      = cfg.get<double>( "MinHeight"     );

    _gaus_mean_v  = cfg.get<std::vector<double> >( "ADCSmearingMean"  );
    _gaus_sigma_v = cfg.get<std::vector<double> >( "ADCSmearingSigma" );

    _gaus_pool_size_v.clear();
    _gaus_pool_size_v.resize(3,9600*3456);
    _gaus_pool_size_v  = cfg.get<std::vector<size_t> >( "RandomPoolSize", _gaus_pool_size_v);

    if(_gaus_mean_v.size() != _gaus_sigma_v.size()) {
      LARCV_CRITICAL() << "ADCSmearing Mean & Sigma must be of same length!" << std::endl;
      throw larbys();
    }
    if(_gaus_mean_v.size() != _gaus_pool_size_v.size()) {
      LARCV_CRITICAL() << "ADCSmearing Mean/Sigma and pool size must be of same length!" << std::endl;
      throw larbys();
    }

    _randg_v.clear();
    for(size_t i=0; i<_gaus_mean_v.size(); ++i) {

      //RandomGaus rg(_gaus_mean_v[i], _gaus_sigma_v[i], _gaus_pool_size_v[i]);
      //_randg_v.emplace_back(std::move(rg));
      _randg_v.emplace_back(_gaus_mean_v[i], _gaus_sigma_v[i], _gaus_pool_size_v[i]);

    }

  }

  void MCNuStream::initialize()
  {}

  bool MCNuStream::process(IOManager& mgr)
  {
    _tpc_image_v.clear();
    _tpc_segment_v.clear();
    _roi_v.clear();
    
    // Retrieve ROI that match our condition
    auto event_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));

    bool found=false;
    for(auto const& roi : event_roi->ROIArray()) {

      if(roi.Type() != kROIBNB) continue;

      if( roi.EnergyDeposit() < _min_energy_deposit ) return false;
      if( roi.EnergyInit()    < _min_energy_init    ) return false;
      double min_width  = 1e12;
      double min_height = 1e12;
      for(auto const& bb : roi.BB()) {
	if( bb.width()  < min_width  ) min_width  = bb.width();
	if( bb.height() < min_height ) min_height = bb.height();
      }
      if( min_width  < _min_width  ) return false;
      if( min_height < _min_height ) return false;

      found=true;
      break;
    }
    if(!found) return false;

    _roi_v = event_roi->ROIArray();

    // Retrieve TPC image
    auto event_tpc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_tpc_image_producer));

    if(!event_tpc_image || event_tpc_image->Image2DArray().empty()) return false;

    _tpc_image_v = event_tpc_image->Image2DArray();

    // Smear ADCs if random gaussian is provided
    if(!_randg_v.empty()) {
      // Make sure enough generators exist (# images)
      if(_tpc_image_v.size() < _randg_v.size()) {
	LARCV_CRITICAL() << "# images > # of gaussian generator... check config!" << std::endl;
	throw larbys();	
      }
      // Loop over images
      for(size_t i=0; i<_tpc_image_v.size(); ++i) {
	
	//auto& randg = _randg_v[i];
	auto& tpc_image = _tpc_image_v[i];
	// Throw warning: @ this code it "should be" index = plane id
	if(tpc_image.meta().plane() != i)
	  LARCV_WARNING() << "Image index != plane ID is detected... " << std::endl;
	// Prepare random image and fill from gaussian generator
	std::vector<float> rand_img;
	_randg_v[i].get(rand_img);
	// If size is too small regenerate
	if(rand_img.size() < tpc_image.size()) {
	  LARCV_CRITICAL() << "Detected image size > random number pool size!" << std::endl;
	  throw larbys();
	}
	// Perform elt-wise multiplication. Allow random image to be larger in size
	tpc_image.eltwise(rand_img,true);
	// Start thread for gaussian random number generation
	_randg_v[i].start_filling();
      }
    }

    // Retrieve PMT image
    auto event_pmt_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_pmt_image_producer));

    if(!event_pmt_image || event_pmt_image->Image2DArray().empty()) return false;

    _pmt_image = event_pmt_image->Image2DArray().front();

    // Retrieve TPC segment
    auto event_tpc_segment = (EventImage2D*)(mgr.get_data(kProductImage2D,_segment_producer));

    _tpc_segment_v = event_tpc_segment->Image2DArray();
    
    retrieve_id(event_tpc_image);

    return true;
    
  }

  void MCNuStream::finalize(TFile* ana_file)
  {}

}
#endif
