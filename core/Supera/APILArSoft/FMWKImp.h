#ifndef __SUPERA_FMWKIMP_H__
#define __SUPERA_FMWKIMP_H__

#include "SimulationBase/MCTruth.h"
#include "lardata/MCBase/MCShower.h"
#include "lardata/MCBase/MCTrack.h"
#include "larsim/Simulation/SimChannel.h"

#include "Cropper.h"
template class larcv::supera::Cropper<sim::MCTrack,sim::MCShower,sim::SimChannel>;

#include "MCParticleTree.h"
template class larcv::supera::MCParticleTree<simb::MCTruth,sim::MCTrack,sim::MCShower,sim::SimChannel>;

#endif
