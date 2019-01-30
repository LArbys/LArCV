/**
 * \file Run3DTracker.h
 *
 * \ingroup Package_Name
 *
 * \brief Class def header for a class Run3DTracker
 *
 * @author hourlier
 */

/** \addtogroup Package_Name

 @{*/
#ifndef __RUN3DTRACKER_H__
#define __RUN3DTRACKER_H__

#include "TH2D.h"
#include "TH1D.h"
#include "TTree.h"
#include "TVector3.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/track.h"

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

    /**
     \class ProcessBase
     User defined class Run3DTracker ... these comments are used to generate
     doxygen documentation!
     */
    class Run3DTracker : public ProcessBase {

    public:

        /// Default constructor
        Run3DTracker(const std::string name="Run3DTracker");

        /// Default destructor
        ~Run3DTracker(){}

        void configure(const PSet&);
        void initialize();
        bool process(IOManager& mgr);

        void SetSplineLocation(const std::string& fpath);
        void SetLLOutName(const std::string& foutll) { _foutll = foutll; }
        void advance_larlite();
        void FillMC(const std::vector<ROI>& mc_roi_v);
        void ClearEvent();
        void ClearVertex();
        void SetOutDir(std::string s){out_dir = s;}

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
        double NeutrinoEnergyTh;
        double NeutrinoEnergyReco;
        double Ep_t;
        double Em_t;

        int _run;
        int _subrun;
        int _event;
        int _nentry;
        int _entry;
        int _Nreco;

        int _vtx_id;
        int NvertexSubmitted;
        int NgoodReco;
        int NtracksReco;
        std::vector<std::string> checkEvents;
        std::string _filename;

        TTree *_recoTree;
        std::vector<int>    _trk_id_v;
        std::vector<double> _E_muon_v;
        std::vector<double> _E_proton_v;
        std::vector<double> _Length_v;
        std::vector<double> _Avg_Ion_v;
        std::vector<double> _Avg_IonY_v;
        std::vector<double> _vertexPhi;
        std::vector<double> _vertexTheta;
        std::vector<double> _closestWall;
        std::vector<double> _Ion_5cm_v;
        std::vector<double> _Ion_10cm_v;
        std::vector<double> _Ion_tot_v;

        std::vector<double> _IonY_5cm_v;
        std::vector<double> _IonY_10cm_v;
        std::vector<double> _IonY_tot_v;

        std::vector<double> _IondivLength_v;
        std::vector<std::vector<double>> _trackQ3_v;
        std::vector<std::vector<double>> _trackQ5_v;
        std::vector<std::vector<double>> _trackQ10_v;
        std::vector<std::vector<double>> _trackQ20_v;
        std::vector<std::vector<double>> _trackQ30_v;
        std::vector<std::vector<double>> _trackQ50_v;
        std::vector< std::vector<double> > _TotalADCvalues_v;
        std::vector< std::vector<double> > _Angle_v;
        std::vector<bool>   _Reco_goodness_v;
        std::vector<bool>  _track_Goodness_v;
        std::vector<larlite::event_track> _EventRecoVertices;

        std::vector<TVector3> MCVertices;
        TVector3 MCvertex;
        TVector3 RecoVertex;
        TVector3 _RecoVertex;

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

        float _vtx_x;
        float _vtx_y;
        float _vtx_z;

        double _Ep_t;
        double _Em_t;
        double _Ee_t;

        //
        // start point
        //
        double _MuonStartPoint_X;
        double _ProtonStartPoint_X;
        double _ElectronStartPoint_X;

        double _MuonStartPoint_Y;
        double _ProtonStartPoint_Y;
        double _ElectronStartPoint_Y;

        double _MuonStartPoint_Z;
        double _ProtonStartPoint_Z;
        double _ElectronStartPoint_Z;

        //
        // end point
        //
        double _MuonEndPoint_X;
        double _ProtonEndPoint_X;
        double _ElectronEndPoint_X;

        double _MuonEndPoint_Y;
        double _ProtonEndPoint_Y;
        double _ElectronEndPoint_Y;

        double _MuonEndPoint_Z;
        double _ProtonEndPoint_Z;
        double _ElectronEndPoint_Z;

        std::string _input_pgraph_producer;
        std::string _img2d_producer;
        std::string _par_pix_producer;
        std::string _true_roi_producer;
        std::string _spline_file;
        std::string _foutll;
        std::string out_dir;
        bool _mask_shower;
    };

    /**
     \class larcv::Run3DTrackerFactory
     \brief A concrete factory class for larcv::Run3DTracker
     */
    class Run3DTrackerProcessFactory : public ProcessFactoryBase {
    public:
        /// ctor
        Run3DTrackerProcessFactory() { ProcessFactory::get().add_factory("Run3DTracker",this); }
        /// dtor
        ~Run3DTrackerProcessFactory() {}
        /// creation method
        ProcessBase* create(const std::string instance_name) { return new Run3DTracker(instance_name); }
        
    };
    
}

#endif
/** @} */ // end of doxygen group 

