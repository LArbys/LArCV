#ifndef __READNUEFILE_H__
#define __READNUEFILE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

  class ReadNueFile : public ProcessBase {

  public:

    ReadNueFile(const std::string name="ReadNueFile");
    ~ReadNueFile(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();
    
    void SetSplineLocation(const std::string& fpath);
    void SetLLOutName(const std::string& foutll) { _foutll = foutll; }
    void ClearEvent();
    void ClearVertex();
    void FillMC(const std::vector<ROI>& mc_roi_v);

  private:
    
    void advance_larlite();

    std::string _foutll;
    int iTrack;
    larlite::storage_manager _storage;
    larcv::AStarTracker tracker;

    std::vector< std::vector<int> > _vertexInfo;
    
    std::string _spline_file;

    TTree *_recoTree;
    std::string _img2d_producer;
    std::string _pgraph_producer;
    std::string _par_pix_producer;
    std::string _true_roi_producer;

    std::vector<std::string> checkEvents;

    int _run;
    int _subrun;
    int _event;
    int _nentry;

    bool _mask_shower;

    std::vector<larcv::ImageMeta> _Full_meta_v;
    std::vector<larcv::Image2D> _Full_image_v;
    std::vector<larcv::Image2D> _Tagged_Image;

    std::vector<double> _E_muon_v;
    std::vector<double> _E_proton_v;
    std::vector<double> _Length_v;
    std::vector<double> _Avg_Ion_v;
    std::vector<double> _vertexPhi_v;
    std::vector<double> _vertexTheta_v;
    std::vector<double> _closestWall_v;
    std::vector<double> _Ion_5cm_v;
    std::vector<double> _Ion_10cm_v;
    std::vector<double> _Ion_tot_v;
    std::vector<double> _IondivLength_v;
    std::vector<double> _Trunc_dQdX1_v;
    std::vector<double> _Trunc_dQdX3_v;
    std::vector<std::vector<double> > _Angle_v;
    std::vector<int> _Reco_goodness_v;
    std::vector<larlite::event_track> _EventRecoVertices;


    int _missingTrack;
    int _nothingReconstructed;
    int _tooShortDeadWire;
    int _tooShortFaintTrack;
    int _tooManyTracksAtVertex;
    int _possibleCosmic;
    int _possiblyCrossing;
    int _branchingTracks;
    int _jumpingTracks;
    bool _isMC;

    int   _vtx_id;
    float _vtx_x;
    float _vtx_y;
    float _vtx_z;

    double _Ep_t;
    double _Em_t;
    double _Ee_t;

    int _GoodVertex;
    int _Nreco;
    int _entry;

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
    
  };

  class ReadNueFileProcessFactory : public ProcessFactoryBase {
  public:
    ReadNueFileProcessFactory() { ProcessFactory::get().add_factory("ReadNueFile",this); }
    ~ReadNueFileProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new ReadNueFile(instance_name); }
  };
    
}

#endif

