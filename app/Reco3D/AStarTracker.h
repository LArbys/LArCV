/**
 * \file AStarTracker.h
 *
 * \ingroup Adrien
 *
 * \brief Class def header for a class AStarTracker
 *
 * @author Adrien
 */

/** \addtogroup Adrien

 @{*/

#ifndef LARLITE_AStarTracker_H
#define LARLITE_AStarTracker_H

#include <string>
#include <fstream>

#include "Analysis/ana_base.h"
#include "DataFormat/track.h"
#include "DataFormat/mctrack.h"
#include <TVector3.h>
#include "DataFormat/Image2D.h"
#include "DataFormat/ImageMeta.h"
#include "DataFormat/wire.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TSpline.h"

//#include "LArCV/core/DataFormat/ChStatus.h"
//#include "larcv/app/LArOpenCVHandle/LArbysUtils.h"
#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

namespace larcv {
    /**
     \class AStarTracker
     User custom analysis class made by SHELL_USER_NAME
     */
    class AStarTracker {

    public:

        /// Default constructor
        AStarTracker(){
            //_name="AStarTracker";
            //_fout=0;//TFile::Open("output.root","RECREATE");
            _track_producer="dl";
            _chstatus_producer = "chstatus";
            _mctrack_producer  = "mcreco";
            _wire_producer     = "caldata";
            _hit_producer      = "gaushit";
            //_speedOffset=-2;
            _speedOffset=0;
            _verbose = 0;
            _ADCthreshold = 10;
            _compressionFactor_t = 6;
            _compressionFactor_w = 1;
            _DrawOutputs = false;
        }

        /// Default destructor
        ~AStarTracker(){}

        /** IMPLEMENT in AStarTracker.cc!
         Initialization method to be called before the analysis event loop.
         */
        bool initialize();

        /** IMPLEMENT in AStarTracker.cc!
         Analyze a data event-by-event
         */
        //virtual bool analyze(storage_manager* storage);

        /** IMPLEMENT in AStarTracker.cc!
         Finalize method to be called after all events processed.
         */
        bool finalize();

        void set_producer(std::string track_producer,std::string chstatus_producer){
            _track_producer = track_producer;
            _chstatus_producer = chstatus_producer;
        }

        void set_trackProducer(  std::string track_producer   ){_track_producer    = track_producer;   }
        void set_chstatProducer( std::string chstatus_producer){_chstatus_producer = chstatus_producer;}
        void set_mctrackProducer(std::string mctrack_producer ){_mctrack_producer  = mctrack_producer; }
        void set_wireProducer(   std::string wire_producer    ){_wire_producer     = wire_producer;    }
        void set_hitProducer(    std::string hit_producer     ){_hit_producer      = hit_producer;     }

        void SetVerbose(int v){_verbose = v;}
        void SetDrawOutputs(bool d){_DrawOutputs = d;}
        void SetCompressionFactors(int compress_w, int compress_t){_compressionFactor_w = compress_w; _compressionFactor_t = compress_t;}
        void SetImages(std::vector<larcv::Image2D> images){hit_image_v = images;}
        void ReadProtonTrackFile();
        void ReadSplineFile();
        void SetTrackInfo(int run, int subrun, int event, int track){_run = run; _subrun = subrun; _event = event; _track = track;}

        void tellMe(std::string s, int verboseMin);
        void CreateDataImage(std::vector<larlite::wire> wire_v);
        void SetTimeAndWireBounds();
        void SetTimeAndWireBoundsProtonsErez();
        void SetTimeAndWireBounds(TVector3 pointStart, TVector3 pointEnd);
        void SetTimeAndWireBounds(std::vector<TVector3> points);
        void SetTimeAndWireBounds(std::vector<TVector3> points, std::vector<larcv::ImageMeta> meta);
        void SetEndPoints(TVector3 vertex, TVector3 endpoint){start_pt = vertex; end_pt = CheckEndPoints(endpoint);}
        void DrawTrack();
        void RegularizeTrack();
        void DrawROI();
        void TellMeRecoedPath();
        void Make3DpointList();
        void ComputeLength();
        void ComputedQdX();
        void Reconstruct();

        bool CheckEndPointsInVolume(TVector3 point);
        bool IsGoodTrack();
        bool CheckEndPoints(std::vector< std::pair<int,int> > endPix);

        int  GetCompressionFactorTime(){return _compressionFactor_t;}
        int  GetCompressionFactorWire(){return _compressionFactor_w;}

        double EvalMinDist(TVector3 point);
        double EvalMinDist(TVector3 point, std::vector< std::pair<int,int> > endPix);
        double GetLength(){return _Length3D;}
        double GetEnergy(std::string partType, double Length);
        double GetTotalDepositedCharge();
        double X2Tick(double x, size_t plane) const;   // X[cm] to TPC tick (waveform index) conversion
        double Tick2X(double tick, size_t plane)const; // TPC tick (waveform index) to X[cm] conversion

        TVector3        CheckEndPoints(TVector3 point);
        TVector3        CheckEndPoints(TVector3 point,std::vector< std::pair<int,int> > endPix);

        std::vector<TVector3>   GetOpenSet(TVector3 newPoint, double dR);
        std::vector<TVector3>   GetTrack(){return _3DTrack;}
        std::vector<TVector3>   Reconstruct(int run,
                                            int subrun,
                                            int event,
                                            int track,
                                            std::vector<larcv::Image2D> images,
                                            TVector3 vertex, TVector3 endPoint){
            SetTrackInfo(run, subrun, event, track);
            SetImages(images);
            SetEndPoints(vertex, endPoint);
            Reconstruct();
            RegularizeTrack();
            return GetTrack();
        }
        
        std::vector<std::vector<int> > _SelectableTracks;
        std::vector< std::vector<double> > GetdQdx(){return _dQdx;}


        std::vector<std::pair<double,double> > GetTimeBounds(){return time_bounds;}
        std::vector<std::pair<double,double> > GetWireBounds(){return wire_bounds;}
        std::vector<std::pair<int, int> >      GetWireTimeProjection(TVector3 point);

        std::vector<larcv::Image2D> CropFullImage2bounds(std::vector<TVector3> EndPoints,std::vector<larcv::Image2D> Full_image_v);

        TSpline3* GetProtonRange2T(){return sProtonRange2T;}
        TSpline3* GetMuonRange2T(){return sMuonRange2T;}
        TSpline3* GetProtonT2dEdx(){return sProtonT2dEdx;}
        TSpline3* GetMuonT2dEdx(){return sMuonT2dEdx;}


    protected:
        
        std::string _track_producer;
        std::string _chstatus_producer;
        std::string _mctrack_producer;
        std::string _wire_producer;
        std::string _hit_producer;

        int _run;
        int _subrun;
        int _event;
        int _track;
        int _compressionFactor_t;
        int _compressionFactor_w;
        int _eventTreated;
        int _eventSuccess;
        int _verbose;

        double _ADCthreshold;
        double _speedOffset;
        double _Length3D;
        double GetDist2line(TVector3 A, TVector3 B, TVector3 C);

        bool _DrawOutputs;

        TH1D *hdQdx;
        TH1D *hdQdxEntries;
        TH1D *hDistance2MC;
        TH1D *hDistance2MCX;
        TH1D *hDistance2MCY;
        TH1D *hDistance2MCZ;
        TH1D *hDistance2Hit;
        TH1D *hDistanceMC2Hit;
        TH2D *hdQdX2D;
        //TH2D *hdQdX2DNorm;

        std::vector< std::vector<double> > _dQdx;
        std::vector<TVector3> CorrectedPath;
        std::vector<TVector3> _3DTrack;

        std::vector<larcv::AStar3DNode> RecoedPath;

        std::vector<larcv::Image2D> hit_image_v;
        std::vector<larcv::Image2D> chstatus_image_v;

        TVector3 start_pt;
        TVector3 end_pt;

        std::vector<std::pair<double,double> > time_bounds;
        std::vector<std::pair<double,double> > wire_bounds;

        TSpline3 *sMuonRange2T;
        TSpline3 *sMuonT2dEdx;
        TSpline3 *sProtonRange2T;
        TSpline3 *sProtonT2dEdx;

        TCanvas *c2;
    };
}
#endif

//**************************************************************************
//
// For Analysis framework documentation, read Manual.pdf here:
//
// http://microboone-docdb.fnal.gov:8080/cgi-bin/ShowDocument?docid=3183
//
//**************************************************************************

/** @} */ // end of doxygen group 
