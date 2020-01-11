/**
 * \file SaveCutVariables
 *
 * \brief Class def header for a class ShowerMask
 * An overarching class that reads in all necessary root files for a run,subrun,event:
 * loads in reco2d2d hits and makss hist with bad shower scores
 *
 * @author katie
 */

/** \addtogroup core_DataFormat

    @{*/
#ifndef __SAVECUTVARIABLES_H__
#define __SAVECUTVARIABLES_H__

// includes
#include <iostream>
#include <map>
#include <utility>
#include <string>
#include <cstring>
#include <bits/stdc++.h>
#include <fstream>
// ROOT
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2D.h"
#include "TH3D.h"
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
#include "DataFormat/shower.h"
#include "DataFormat/vertex.h"
#include "DataFormat/event_ass.h"
#include "DataFormat/pfpart.h"
#include "DataFormat/hit.h"
#include "DataFormat/cluster.h"
#include "DataFormat/potsummary.h"
// larutil
#include "LArUtil/LArProperties.h"
#include "LArUtil/DetectorProperties.h"
#include "LArUtil/Geometry.h"
#include "LArUtil/ClockConstants.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

// larcv
#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "Base/larcv_base.h"
#include "DataFormat/IOManager.h"
#include "DataFormat/EventImage2D.h"
#include "DataFormat/EventChStatus.h"
#include "DataFormat/EventPGraph.h"
#include "DataFormat/EventROI.h"

#include <cstdlib>
#include <math.h>
#include <bits/stdc++.h>
#include <cstring>
#include <stdio.h>

#include "Utils.h"

namespace larcv {
namespace showermask {

  class ShowerMask{

  public:
    ShowerMask(){};
    virtual ~ShowerMask() {};

    //main code to run.
    void configure( const larcv::PSet& );
    void initialize( );
    bool process(larcv::IOManager& io,larlite::storage_manager& ioll,
        larcv::IOManager& ioforward,larlite::storage_manager& outputdata );
    void finalize();  // function to get true_vtx_location


    // side functions
    std::vector<std::vector<double>> GetRecoVtxLocs(larcv::EventPGraph* ev_pgraph);
    void HitDistanceFromVtx();
    void HitLocation(larlite::event_hit* ev_hitsreco2d,larcv::ImageMeta wireu_meta,
        larcv::ImageMeta wirev_meta,larcv::ImageMeta wirey_meta);
    void ChooseHitstoKeep(std::vector<larcv::Image2D> shower_score_u, std::vector<larcv::Image2D>shower_score_v,
                          std::vector<larcv::Image2D> shower_score_y, float threshold);

  protected:

    TFile* OutFile;

    int run;
    int subrun;
    int event;
    //input hits
    std::string _input_hits_producer;
    //input wire Image
    std::string _input_wire_producer;
    //ssnet inputs
    std::string _input_ssnet_uplane_producer;
    std::string _input_ssnet_vplane_producer;
    std::string _input_ssnet_yplane_producer;
    //vtx input
    std::string _input_vtx_producer;

    //vtx locations
    std::vector<std::vector<double>> reco_vtx_location_3D_v; //vector of locations of reco vtx
    std::vector<std::vector<int>> reco_vtx_location_2D_v; //vector of locations of reco vtx
    std::vector<int> hitdistances_v; //for each hit the closest 2d vtx distance
    std::vector<std::vector<float>> hit_location;//vector of location of each hit
    std::vector<int> hitstokeep_v;
    std::vector<larlite::hit> kepthits_v; //vector of kept hit objects

  };

}
}


#endif
