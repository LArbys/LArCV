/**
 * \file ReadJarrettFile.h
 *
 * \ingroup Package_Name
 *
 * \brief Class def header for a class ReadJarrettFile
 *
 * @author hourlier
 */

/** \addtogroup Package_Name

 @{*/
#ifndef __READJARRETTFILE_H__
#define __READJARRETTFILE_H__

#include "TH2D.h"
#include "TH1D.h"
#include "TTree.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/track.h"

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

    /**
     \class ProcessBase
     User defined class ReadJarrettFile ... these comments are used to generate
     doxygen documentation!
     */
    class ReadJarrettFile : public ProcessBase {

    public:

        /// Default constructor
        ReadJarrettFile(const std::string name="ReadJarrettFile");

        /// Default destructor
        ~ReadJarrettFile(){}

        void configure(const PSet&);

        void initialize();

        bool process(IOManager& mgr);

        bool IsGoodVertex(int run, int subrun, int event/*, int ROIid*/, int vtxID);
        bool IsGoodEntry(int run, int subrun, int event);
        void ReadVertexFile(std::string filename);
        std::vector<TVector3> GetJarretVertex(int run, int subrun, int event);

        void finalize();

        void MCevaluation();

    private :

        int iTrack;
        larlite::storage_manager _storage;
        larcv::AStarTracker tracker;
        std::vector< std::vector<int> > _vertexInfo;
        TH2D *hEcomp;
        TH1D *hAverageIonization;
        TH1D *hEcomp1D;
        TH1D *hEcomp1D_m;
        TH1D *hEcomp1D_p;
        TH2D *hEcomp_m;
        TH2D *hEcomp_p;
        TH1D *hEnuReco;
        TH1D *hEnuTh;
        TH2D *hEnuvsPM_th;
        TH1D *hPM_th_Reco_1D;
        TH2D *hPM_th_Reco;
        TH2D *hEnuComp;
        TH1D *hEnuComp1D;
        double NeutrinoEnergyTh;
        double NeutrinoEnergyReco;
        TH2D *hEcompdQdx;
        TH2D *hIonvsLength;
        double Ep_t;
        double Em_t;
        int run;
        int subrun;
        int event;
        int NvertexSubmitted;
        int NgoodReco;
        std::vector<std::string> checkEvents;
        std::string _filename;

        TTree *_recoTree;
        std::vector<double> _E_muon_v;
        std::vector<double> _E_proton_v;
        std::vector<double> _Length_v;
        std::vector<double> _Avg_Ion_v;
        std::vector<double> _vertexPhi;
        std::vector<double> _vertexTheta;
        std::vector<std::vector<double> > _Angle_v;
        std::vector<bool>   _Reco_goodness_v;
        std::vector<larlite::event_track> _EventRecoVertices;


        TVector3 MCvertex;
        TVector3 RecoVertex;

        bool GoodVertex;
        bool _missingTrack;
        bool _nothingReconstructed;
        bool _tooShortDeadWire;
        bool _tooShortFaintTrack;
        bool _tooManyTracksAtVertex;
        bool _possibleCosmic;
        bool _possiblyCrossing;
        bool _branchingTracks;
        bool _jumpingTracks;
        bool _isMC;

	std::string _input_pgraph_producer;
    };

    /**
     \class larcv::ReadJarrettFileFactory
     \brief A concrete factory class for larcv::ReadJarrettFile
     */
    class ReadJarrettFileProcessFactory : public ProcessFactoryBase {
    public:
        /// ctor
        ReadJarrettFileProcessFactory() { ProcessFactory::get().add_factory("ReadJarrettFile",this); }
        /// dtor
        ~ReadJarrettFileProcessFactory() {}
        /// creation method
        ProcessBase* create(const std::string instance_name) { return new ReadJarrettFile(instance_name); }

    };
    
}

#endif
/** @} */ // end of doxygen group 

