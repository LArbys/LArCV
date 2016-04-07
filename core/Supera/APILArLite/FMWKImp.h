#ifndef __SUPERA_FMWKIMP_H__
#define __SUPERA_FMWKIMP_H__

#include "DataFormat/mctruth.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrack.h"

#include "Cropper.h"
template class larcv::supera::Cropper<larlite::mctrack,larlite::mcshower>;

#include "MCParticleTree.h"
template class larcv::supera::MCParticleTree<larlite::mctruth,larlite::mctrack,larlite::mcshower>;

#endif
