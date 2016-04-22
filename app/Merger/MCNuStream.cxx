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

    if(_gaus_mean_v.size() != _gaus_sigma_v.size()) {
      LARCV_CRITICAL() << "ADCSmearing Mean & Sigma must be of same length!" << std::endl;
      throw larbys();
    }

    /*
    // Threaded gaussian commented out (too big pool size)
    _gaus_pool_size_v.clear();
    _gaus_pool_size_v.resize(3,9600*3456);
    _gaus_pool_size_v  = cfg.get<std::vector<size_t> >( "RandomPoolSize", _gaus_pool_size_v);

    if(_gaus_mean_v.size() != _gaus_pool_size_v.size()) {
      LARCV_CRITICAL() << "ADCSmearing Mean/Sigma and pool size must be of same length!" << std::endl;
      throw larbys();
    }

    _randg_v.clear();
    for(size_t i=0; i<_gaus_mean_v.size(); ++i)

      _randg_v.emplace_back(_gaus_mean_v[i], _gaus_sigma_v[i], _gaus_pool_size_v[i]);
    */
    
    LARCV_INFO() << "Configured..." << std::endl;
  }

  void MCNuStream::initialize()
  {
    LARCV_INFO() << "Initialized..." << std::endl;
  }

  bool MCNuStream::process(IOManager& mgr)
  {
    LARCV_INFO() << "Clearing attributes..." << std::endl;
    _tpc_image_v.clear();
    _tpc_segment_v.clear();
    _roi_v.clear();

    // Retrieve ROI that match our condition
    LARCV_INFO() << "Reading in ROI " << _roi_producer << std::endl;
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

    LARCV_INFO() << "Copying ROIs.." <<std::endl;
    _roi_v = event_roi->ROIArray();

    // Retrieve TPC image
    LARCV_INFO() << "Reading in TPC Image2D " << _tpc_image_producer << std::endl;
    auto event_tpc_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_tpc_image_producer));
    
    if(!event_tpc_image || event_tpc_image->Image2DArray().empty()) return false;

    LARCV_INFO() << "Copying TPC Image2D " << _tpc_image_producer << std::endl;
    event_tpc_image->Move(_tpc_image_v);

    // Smear ADCs if random gaussian is provided
    if(!_gaus_mean_v.empty()) {

      for(size_t i=0; i<_tpc_image_v.size(); ++i) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> d(_gaus_mean_v[i],_gaus_sigma_v[i]);
	
	//auto& randg = _randg_v[i];
	auto& tpc_image = _tpc_image_v[i];
	// Throw warning: @ this code it "should be" index = plane id
	if(tpc_image.meta().plane() != i)
	  LARCV_WARNING() << "Image index != plane ID is detected... " << std::endl;
	
	auto const& img_vec = tpc_image.as_vector();
	
	for(size_t i=0; i<img_vec.size(); ++i) {
	  
	  if(img_vec[i] < 1.) continue;
	  float factor = d(gen);
	  size_t col = i / tpc_image.meta().rows();
	  size_t row = i - col * tpc_image.meta().rows();
	  tpc_image.set_pixel(row,col,img_vec[i] * factor);
	}
      }
    }
    /*
    // Thread method commented out... too much memory for pool!
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
	LARCV_INFO() << "ElementWise mult on TPC Image2D " << std::endl;
	tpc_image.eltwise(rand_img,true);
	// Start thread for gaussian random number generation
	sleep(5);
	
	_randg_v[i].start_filling();
      }
    }
    */
    
    // Retrieve PMT image
    LARCV_INFO() << "Reading in PMT Image2D " << _pmt_image_producer << std::endl;
    auto event_pmt_image = (EventImage2D*)(mgr.get_data(kProductImage2D,_pmt_image_producer));
    if(!event_pmt_image || event_pmt_image->Image2DArray().empty()) return false;

    LARCV_INFO() << "Copying in PMT Image2D " << _pmt_image_producer << std::endl;
    std::vector<larcv::Image2D> tmp_v;
    event_pmt_image->Move(tmp_v);
    if(tmp_v.size())
      _pmt_image = std::move(tmp_v[0]);
    
    // Retrieve TPC segment
    LARCV_INFO() << "Reading in Segment Image2D " << _segment_producer << std::endl;
    auto event_tpc_segment = (EventImage2D*)(mgr.get_data(kProductImage2D,_segment_producer));

    LARCV_INFO() << "Copying in Segment Image2D " << _segment_producer << std::endl;    
    event_tpc_segment->Move(_tpc_segment_v);

    retrieve_id(event_tpc_image);

    return true;
    
  }

  void MCNuStream::finalize(TFile* ana_file)
  {}

}
#endif
