#ifndef __SUPERA_FMWKIMP_H__
#define __SUPERA_FMWKIMP_H__

#include "nusimdata/SimulationBase/MCTruth.h"
#include "lardataobj/MCBase/MCShower.h"
#include "lardataobj/MCBase/MCTrack.h"
#include "lardataobj/Simulation/SimChannel.h"

#include "Cropper.h"
template class larcv::supera::Cropper<sim::MCTrack,sim::MCShower,sim::SimChannel>;

#include "MCParticleTree.h"
template class larcv::supera::MCParticleTree<simb::MCTruth,sim::MCTrack,sim::MCShower,sim::SimChannel>;

#endif
