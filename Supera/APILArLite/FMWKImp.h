#ifndef __SUPERA_FMWKIMP_H__
#define __SUPERA_FMWKIMP_H__

#include "DataFormat/mctruth.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/simch.h"

#include "Cropper.h"
template class larcv::supera::Cropper<larlite::mctrack,larlite::mcshower,larlite::simch>;

#include "MCParticleTree.h"
template class larcv::supera::MCParticleTree<larlite::mctruth,larlite::mctrack,larlite::mcshower,larlite::simch>;

#endif
