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
#include "TGraph2D.h"
#include "TGraph.h"
#include "TFile.h"
#include "TRandom3.h"

//#include "LArCV/core/DataFormat/ChStatus.h"
//#include "larcv/app/LArOpenCVHandle/LArbysUtils.h"
#include "AStar3DAlgo.h"
#include "AStar3DAlgoProton.h"

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larcv {
    class AStarTracker {

    public:

        /// Default constructor
        AStarTracker(){
            _track_producer    = "dl";
            _chstatus_producer = "chstatus";
            _mctrack_producer  = "mcreco";
            _wire_producer     = "caldata";
            _hit_producer      = "gaushit";
            _spline_file       = "";
            _speedOffset=0;
            _verbose = 0;
            _ADCthreshold = 10;
            _compressionFactor_t = 6;
            _compressionFactor_w = 1;
            _DrawOutputs = false;
        }

        /// Default destructor
        ~AStarTracker(){}

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
        void SetIsMCC9(bool isMCC9){_IsMCC9 = isMCC9;}
        void SetDrawOutputs(bool d){_DrawOutputs = d;}
        void SetCompressionFactors(int compress_w, int compress_t){_compressionFactor_w = compress_w; _compressionFactor_t = compress_t;}
        void ReadSplineFile();
        void SetSplineFile(const std::string& fpath);
        void SetOutputDir(std::string outdir){_outdir=outdir;}
        void SetTrackInfo(int run, int subrun, int event, int track){_run = run; _subrun = subrun; _event = event; _track = track;}
        void tellMe(std::string s, int verboseMin);
        void CreateDataImage(std::vector<larlite::wire> wire_v);
        void ResetRecoveries(){NumberRecoveries = 0;}
        void SetTimeAndWireBounds();
        void SetDrawVertical(bool b){_DrawVertical=b;}
        void SetDrawBlack(bool b){_DrawBlack=b;}
        void SetTimeAndWireBoundsProtonsErez();
        void SetTimeAndWireBounds(TVector3 pointStart, TVector3 pointEnd);
        void SetTimeAndWireBounds(std::vector<TVector3> points);
        void SetTimeAndWireBounds(std::vector<TVector3> points, std::vector<larcv::ImageMeta> meta);
        void SetEndPoints(TVector3 vertex, TVector3 endpoint){start_pt = vertex; end_pt = endpoint;}
        void SetEndPoint(TVector3 endpoint){end_pt = CheckEndPoints(endpoint);}
        void SetImages(          std::vector<larcv::Image2D> images        ){hit_image_v           = images;}
        void SetOriginalImage(   std::vector<larcv::Image2D> originalimage ){original_full_image_v = originalimage;}
        void SetTaggedImage(     std::vector<larcv::Image2D> taggedImage   ){taggedPix_v           = taggedImage;}
        void SetVertexEndPoints( std::vector<TVector3> vertexEndPoints     ){_vertexEndPoints = vertexEndPoints;}
        void SetVertexID(int vtxID){_vtxID = vtxID;}
        void FeedLarliteVertexTracks(larlite::event_track recoedVertexTracks){_vertexLarliteTracks = recoedVertexTracks;Get3DtracksFromLarlite();}
        void FeedVtxGoodness(std::vector<bool> goodTracks_v);
        void Get3DtracksFromLarlite();
        void SetSingleVertex(TVector3 vertexPoint){NumberRecoveries=0; start_pt = vertexPoint;
            std::cout <<"Setting vertex to : "<< vertexPoint.X() << "," << vertexPoint.Y() << "," << vertexPoint.Z() << std::endl; }
        void SetEventVertices(   std::vector<TVector3> vertex_v            ){_eventVertices   = vertex_v;}
        void FeedTrack(std::vector<TVector3> newTrack);
        void WorldInitialization();

        void DrawTrack();
        void DrawVertex();
        void DrawVertexVertical();
        void DrawVertexHorizontal();
        void RegularizeTrack();
        void DrawROI();
        void TellMeRecoedPath();
        void Make3DpointList();
        void MakeVertexTrack();
        void MakeTrack();
        void ComputeLength();
        void ComputeClosestWall();
        void ComputeClosestWall_SCE();
        void ComputedQdX();
        void ComputeNewdQdX();
        void Reconstruct();
        void ReconstructVertex();
        void ConstructTrack();
        void ConstructVertex();
        void ReconstructEvent();
        void ImprovedCluster();
        void PreSortAndOrderPoints();
        void SortAndOrderPoints();
        void MaskVertex();
        void MaskTrack();
        void CleanUpVertex();
        void DiagnoseVertex();
        void DiagnoseTrack(size_t itrack);
        void ShaveTracks();
        void RecoverFromFail();
        void EnhanceDerivative();
        void DumpTrack();
        void FillInTrack();
        void SetMinLength(double MinLength){_MinLength = MinLength;}
        void DrawVertex3D();
        void SetRandomSeed(int this_seed){randomSeed = this_seed;}

        bool initialize();
        bool finalize();
        bool ArePointsEqual(TVector3 A, TVector3 B);
        bool CheckEndPointsInVolume(TVector3 point);
        bool IsGoodTrack();
        bool IsGoodVertex();
        bool IsInSegment(TVector3 A, TVector3 B, TVector3 C);
        bool CheckEndPoints(std::vector< std::pair<int,int> > endPix);
        bool IsTrackIn(std::vector<TVector3> trackA, std::vector<TVector3> trackB);

        int GetCompressionFactorTime(){return _compressionFactor_t;}
        int GetCompressionFactorWire(){return _compressionFactor_w;}
        int GetRandomSeed(){return randomSeed;}


        double EvalMinDist(TVector3 point);
        double EvalMinDist(TVector3 point, std::vector< std::pair<int,int> > endPix);
        double GetLength(){return _Length3D;}
        std::vector<double> GetVertexLength();
        std::vector<double> GetClosestWall();
        std::vector<double> GetClosestWall_SCE();

        void Add_SCE_to_Tracks();

        std::vector<double> GetOldVertexAngle(double dAverage);
        std::vector<int> GetRecoGoodness();
        std::vector<int> GetVtxQuality();
        double GetEnergy(std::string partType, double Length);
        double GetMinLength(){return _MinLength;}
        double ComputeLength(int node);
        double GetTotalDepositedCharge();
        double X2Tick(double x, size_t plane) const;   // X[cm] to TPC tick (waveform index) conversion
        double Tick2X(double tick, size_t plane)const; // TPC tick (waveform index) to X[cm] conversion
        double GetDist2track(TVector3 thisPoint, std::vector<TVector3> thisTrack);

        TVector3        CheckEndPoints(TVector3 point);
        TVector3        CheckEndPoints(TVector3 point,std::vector< std::pair<int,int> > endPix);
        TVector3        GetFurtherFromVertex();

        std::vector<TVector3>   GetOpenSet(TVector3 newPoint, double dR);
        std::vector<TVector3>   GetOpenSet(TVector3 newPoint, int BoxSize, double dR);
        std::vector<TVector3>   GetTrack(){return _3DTrack;}
        std::vector<TVector3>   OrderList(std::vector<TVector3> list);
        std::vector<TVector3>   FitBrokenLine();
        std::vector<TVector3>   GetEndPoints();
        std::vector<TVector3>   Reconstruct(int run, int subrun,int event,int track,std::vector<larcv::Image2D> images,TVector3 vertex, TVector3 endPoint){
            SetTrackInfo(run, subrun, event, track);
            SetImages(images);
            SetEndPoints(vertex, endPoint);
            Reconstruct();
            RegularizeTrack();
            return GetTrack();
        }
        std::vector<std::vector<TVector3> > GetVertexTracks(){return _vertexTracks;}

        void ClearDeadWireList();
        void MakeDeadWireList();

        std::vector<double>  GetAverageIonization(double distAvg = -1);// average pixel intensity over reconstructed points
        std::vector<double>  GetAverageIonization_Yplane(double distAvg = -1);// average Y plane pixel intensity over reconstructed points
        std::vector<double>  GetTotalIonization_Yplane(double distAvg = -1);// total Y plane plane pixel intensity over reconstructed points
        std::vector<double>  GetTotalIonization(double distAvg = -1);// total pixel intensity over reconstructed points
        std::vector<double>  ComputeTruncateddQdX(double);
        std::vector<double>  GetVertexPhi(){return _vertexPhi;}
        std::vector<double>  GetVertexTheta(){return _vertexTheta;}

        std::vector< std::vector<int> >    _SelectableTracks;
        std::vector< std::vector<double> > GetVertexAngle(double dAverage, double dMin);
        std::vector< std::vector<double> > GetDeadWireList(){return _deadWires_v;}
        std::vector< std::vector<double> > GetdQdx(){return _dQdx;}
        std::vector< std::vector<double> > GetEnergies();
        std::vector< std::vector<double> > GetTotalPixADC();
        std::vector< std::vector<double> > GetTotalPixADC(float tkLen);

        larlite::event_track GetReconstructedVertexTracks(){return _vertexLarliteTracks;}



        std::vector<std::pair<double,double> > GetTimeBounds(){return time_bounds;}
        std::vector<std::pair<double,double> > GetWireBounds(){return wire_bounds;}
        std::vector<std::pair<int, int> >      GetWireTimeProjection(TVector3 point);

        std::vector<larcv::Image2D> GetFullImage();
        std::vector<larcv::Image2D> CropFullImage2bounds(std::vector<TVector3> EndPoints);
        std::vector<larcv::Image2D> CropFullImage2bounds(std::vector< std::vector<TVector3> > _vertex_v);
        void CropFullImage2boundsIntegrated(std::vector<TVector3> EndPoints){hit_image_v = CropFullImage2bounds(EndPoints);/*EnhanceDerivative();*/ShaveTracks();}

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
        std::string _spline_file;

        int _run;
        int _subrun;
        int _event;
        int _track;
        int _vtxID;
        int _compressionFactor_t;
        int _compressionFactor_w;
        int _eventTreated;
        int _eventSuccess;
        int _verbose;
        int _deadWireValue;
        int failedPlane;
        int NumberRecoveries;
        int randomSeed;

        double _ADCthreshold;
        double _speedOffset;
        double _Length3D;
        double _MinLength;
        double GetDist2line(TVector3 A, TVector3 B, TVector3 C);

        bool _IsMCC9;
        bool _DrawOutputs;
        bool _missingTrack;
        bool _nothingReconstructed;
        bool _tooManyTracksAtVertex;
        bool _possibleCosmic;
        bool _tooShortDeadWire;
        bool _tooShortFaintTrack;
        bool _possiblyCrossing;
        bool _branchingTracks;
        bool _jumpingTracks;
        bool _DrawVertical;
        bool _DrawBlack;

        std::vector<std::vector<double> > _deadWires_v;
        std::vector<bool> _tooShortDeadWire_v;
        std::vector<bool> _tooShortFaintTrack_v;
        std::vector<bool> _possiblyCrossing_v;
        std::vector<bool> _branchingTracks_v;
        std::vector<bool> _jumpingTracks_v;
        std::vector<bool> _goodTrack_v;

        TH1D *hdQdx;
        TH1D *hLength;
        TH1D *hdQdxEntries;
        TH1D *hDistance2MC;
        TH1D *hDistance2MCX;
        TH1D *hDistance2MCY;
        TH1D *hDistance2MCZ;
        TH1D *hDistance2Hit;
        TH1D *hDistanceMC2Hit;
        TH2D *hdQdX2D;
        TH2D *hAngleLengthGeneral;
        TH2D *hLengthdQdX;
        TH1D *hDist2point;

        std::vector< std::vector<double> > _dQdx;
        std::vector< std::vector<double> > _vertexQDQX;

        std::vector<TVector3> CorrectedPath;
        std::vector<TVector3> _3DTrack;
        std::vector<TVector3> _vertexEndPoints;
        std::vector<TVector3> _eventVertices;
        std::vector<std::vector<TVector3> > _vertexTracks;
        std::vector<std::vector<TVector3> > _vertexTracks_SCE;

        std::vector<larcv::AStar3DNode> RecoedPath;

        std::vector<larcv::Image2D> hit_image_v;
        std::vector<larcv::Image2D> original_full_image_v;
        std::vector<larcv::Image2D> chstatus_image_v;
        std::vector<larcv::Image2D> taggedPix_v;
        std::vector<larcv::Image2D> CroppedTaggedPix_v;

        TVector3 start_pt;
        TVector3 end_pt;

        std::string _outdir;

        std::vector<std::pair<double,double> > time_bounds;
        std::vector<std::pair<double,double> > wire_bounds;

        TSpline3 *sMuonRange2T;
        TSpline3 *sMuonT2dEdx;
        TSpline3 *sProtonRange2T;
        TSpline3 *sProtonT2dEdx;
        TRandom3 ran;

        TGraph   *gdQdXperPlane[3];

        std::vector<double> track_dQdX_v;
        std::vector<double> _vertexLength;
        std::vector<double> _vertexPhi;
        std::vector<double> _vertexTheta;
        std::vector<double> _closestWall;
        std::vector<double> _closestWall_SCE;
        std::vector<std::vector<double> > dQdXperPlane_v;
        std::vector<std::vector<TGraph*> > eventdQdXgraphs;
        std::vector<std::vector<double> > _vertex_dQdX_v;
        std::vector<double> TruncateddQdXperPlane_v;
        std::vector<std::vector<double> > RawdQdXperPlane_v;
        TGraph2D *gDetector;
        TGraph2D *gWorld;

        TFile *fEventOutput;

        larlite::track _thisLarliteTrack;
        larlite::event_track _vertexLarliteTracks;
        larlite::event_track _vertexLarliteTracks_sceadded;

        TCanvas *c2;

    private:
        //	larutil::SpaceChargeMicroBooNE _sce;
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
