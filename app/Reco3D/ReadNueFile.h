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
    void ClearEvent();
    void ClearVertex();

  private :
    int iTrack;
    larcv::AStarTracker tracker;

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

    std::vector<double> _E_muon_v;
    std::vector<double> _E_proton_v;
    std::vector<double> _Length_v;
    std::vector<double> _Avg_Ion_v;
    std::vector<double> _Angle_v;

    std::vector<int> _Reco_goodness_v;

    double _Ep_t;
    double _Em_t;
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

