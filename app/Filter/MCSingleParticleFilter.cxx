#ifndef __MCSINGLEPARTICLEFILTER_CXX__
#define __MCSINGLEPARTICLEFILTER_CXX__

#include "MCSingleParticleFilter.h"
#include "DataFormat/EventROI.h"
namespace larcv {

  static MCSingleParticleFilterProcessFactory __global_MCSingleParticleFilterProcessFactory__;

  MCSingleParticleFilter::MCSingleParticleFilter(const std::string name)
    : ProcessBase(name)
  {}
    
  void MCSingleParticleFilter::configure(const PSet& cfg)
  {
    _roi_producer = cfg.get<std::string>("ROIProducer");
    _shower_min_energy = cfg.get<double>("ShowerMinEnergy");
    _track_min_energy = cfg.get<double>("TrackMinEnergy");
    _proton_min_energy = cfg.get<double>("ProtonMinEnergy");
  }

  void MCSingleParticleFilter::initialize()
  {}

  bool MCSingleParticleFilter::process(IOManager& mgr)
  {
    auto ev_roi = (EventROI*)(mgr.get_data(kProductROI,_roi_producer));
    size_t part_ctr = 0;
    for(auto const& roi : ev_roi->ROIArray()) {
      
      if(roi.MCSTIndex() == kINVALID_USHORT) continue;

      if( (roi.PdgCode() == 11 || roi.PdgCode() == -11 || roi.PdgCode() == 22 || roi.PdgCode() == 111) && 
	  roi.EnergyDeposit() < _shower_min_energy) {
	LARCV_INFO() << "Ignoring Shower (PdgCode=" << roi.PdgCode() << ") with energy " << roi.EnergyDeposit() << std::endl;
	continue;
      }

      if(roi.PdgCode() == 2212 && roi.EnergyDeposit() < _proton_min_energy) {

	LARCV_INFO() << "Ignoring Proton with energy " << roi.EnergyDeposit() << std::endl;
	continue;

      }else if(roi.Shape() == kShapeTrack && roi.EnergyDeposit() < _track_min_energy) {

	LARCV_INFO() << "Ignoring TRACK (PdgCode=" << roi.PdgCode() << ") with energy " << roi.EnergyDeposit() << std::endl;
	continue;

      }

      LARCV_INFO() << "Counting particle (PdgCode=" << roi.PdgCode() << ") with energy " << roi.EnergyDeposit() << std::endl;

      ++part_ctr;

    }
    return (part_ctr == 1);
  }

  void MCSingleParticleFilter::finalize()
  {}

}
#endif
