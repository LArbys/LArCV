/**
 * \file Chimera.h
 *
 * \ingroup Package_Name
 *
 * \brief Class def header for a class Chimera
 *
 * @author hourlier
 */

/** \addtogroup Package_Name

 @{*/
#ifndef __CHIMERA_H__
#define __CHIMERA_H__

#include "TH2D.h"
#include "TH1D.h"
#include "TTree.h"
#include "TVector3.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/track.h"

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "../Reco3D/AStarTracker.h" // Useful for looking at image

namespace larcv {

    /**
     \class ProcessBase
     User defined class Run3DTracker ... these comments are used to generate
     doxygen documentation!
     */
    class Chimera : public ProcessBase {

    public:

        /// Default constructor
        Chimera(const std::string name="Chimera");

        /// Default destructor
        ~Chimera(){}

        void configure(const PSet&);
        void initialize();
        bool process(IOManager& mgr);


        void SetLLOutName(const std::string& foutll) { _foutll = foutll; }
        void SetLLInName(const std::string& finll) { _finll = finll;}
        void SetRootAnaFile(const std::string& anaFile){_fana = anaFile;}
        void SetOutDir(std::string s){out_dir = s;}


        void finalize();

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

        int _run_tree;
        int _subrun_tree;
        int _event_tree;
        int _vtx_id_tree;
        int _vtx_id;
        int NvertexSubmitted;
        int NgoodReco;
        int NtracksReco;
	std::vector<std::string> checkEvents;
	std::string _filename;

        TTree *_recoTree;
	std::vector< std::vector<int> > TreeMap;
	std::vector< std::vector<int> > SelectedList;
	std::vector<int>    _trk_id_v;
	std::vector<double> _E_muon_v;
	std::vector<double> _E_proton_v;
	std::vector<double> *_Length_v;
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

	std::vector<double> _Trunc_dQdX1_v;
	std::vector<double> _Trunc_dQdX3_v;
	std::vector<double> _IondivLength_v;
	std::vector<std::vector<double>> _trackQ3_v;
	std::vector<std::vector<double>> _trackQ5_v;
	std::vector<std::vector<double>> _trackQ10_v;
	std::vector<std::vector<double>> _trackQ20_v;
	std::vector<std::vector<double>> _trackQ30_v;
	std::vector<std::vector<double>> _trackQ50_v;
	std::vector< std::vector<double> > _TotalADCvalues_v;
	std::vector< std::vector<double> > _Angle_v;
	std::vector<bool>   *_Reco_goodness_v;
	std::vector<bool>  *_track_Goodness_v;
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
	std::string _finll;
	std::string _fana;
	std::string out_dir;
	std::string eventListFile;
        bool _mask_shower;

      

    };

    /**
     \class larcv::ChimeraFactory
     \brief A concrete factory class for larcv::Chimera
     */
    class ChimeraProcessFactory : public ProcessFactoryBase {
    public:
        /// ctor
        ChimeraProcessFactory() { ProcessFactory::get().add_factory("Chimera",this); }
        /// dtor
        ~ChimeraProcessFactory() {}
        /// creation method
        ProcessBase* create(const std::string instance_name) { return new Chimera(instance_name); }
        
    };
    
}

#endif
/** @} */ // end of doxygen group 

