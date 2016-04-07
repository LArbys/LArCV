#ifndef __SUPERA_FMWKIMP_H__
#define __SUPERA_FMWKIMP_H__

#include "SimulationBase/MCTruth.h"
#include "lardata/MCBase/MCShower.h"
#include "lardata/MCBase/MCTrack.h"

#include "Cropper.h"
template class larcv::supera::Cropper<sim::MCTrack,sim::MCShower>;

#include "MCParticleTree.h"
template class larcv::supera::MCParticleTree<simb::MCTruth,sim::MCTrack,sim::MCShower>;

#endif
