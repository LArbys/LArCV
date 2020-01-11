/**
 * \file SaveCutVariables
 *
 * \brief Class def header for a class SaveCutVariables
 * An overarching class that reads in all necessary root files for a run,subrun,event:
 *  Calculate all needed cut variables
 *  Save to an output root file that can be hadded
 *
 * @author katie
 */

/** \addtogroup core_DataFormat

    @{*/
#ifndef __UTILS_H__
#define __UTILS_H__

// includes
#include <iostream>
#include <map>
#include <utility>
#include <string>
#include <cstring>
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TH3F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TVector3.h"
// larlite
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/opflash.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctruth.h"
#include "DataFormat/mcnu.h"
#include "DataFormat/track.h"
#include "DataFormat/vertex.h"
#include "DataFormat/event_ass.h"
// larutil
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
// larcv
#include "Base/larcv_base.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventChStatus.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventROI.h"

#include <cstdlib>
#include <math.h>

namespace larcv {
namespace showermask {

  class Utils{

  public:
    Utils(){};
    virtual ~Utils() {};

    // initialize member functions
    // MCC9 SCE correction
    std::vector<double> MakeSCECorrection(std::vector<double> newvec,
    											std::vector<double> original, std::vector<double> offset);
    std::vector<double> GetPosOffsets(std::vector<double> const& point,
    	 																							TH3F* hX, TH3F* hY, TH3F* hZ);
    std::vector<double> GetOffsetsVoxel(std::vector<double> const& point,
    																								TH3F* hX, TH3F* hY, TH3F* hZ);
    std::vector<double> Transform( std::vector<double> const& point);
    double TransformX(double x);
    double TransformY(double y);
    double TransformZ(double z);
    bool IsInsideBoundaries(std::vector<double> const& point);
    bool IsTooFarFromBoundaries(std::vector<double> const& point);
    std::vector<double> PretendAtBoundary(std::vector<double> const& point);
    std::vector<int> getProjectedPixel(const std::vector<double>& pos3d,
    				   const larcv::ImageMeta& meta,
    				   const int nplanes,
    				   const float fracpixborder=1.5 );
    bool InsideFiducial(float x,float y, float z);
    float CalculateSliceWeight(TH1D* muon_slice,TH1D* proton_slice);
    int GetSliceNumber(float resrange);

  protected:

    //if any protected variables needed


  };

}
}


#endif
