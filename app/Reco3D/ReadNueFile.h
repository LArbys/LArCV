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
    void Clear();

  private :
    int iTrack;
    larcv::AStarTracker tracker;

    std::string _spline_file;

    TTree *_reco3d_tree;

    TH2D *hEcomp;
    TH1D *hAverageIonization;
    TH1D *hEcomp1D;
    TH1D *hEcomp1D_m;
    TH1D *hEcomp1D_p;
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
    std::vector<std::string> checkEvents;

    void MCevaluation();
    int _run;
    int _subrun;
    int _event;
    std::vector<double> _E_muon_v;
    std::vector<double> _E_proton_v;
    std::vector<double> _Length_v;
    std::vector<double> _avg_ion_v;

  };

  class ReadNueFileProcessFactory : public ProcessFactoryBase {
  public:
    ReadNueFileProcessFactory() { ProcessFactory::get().add_factory("ReadNueFile",this); }
    ~ReadNueFileProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new ReadNueFile(instance_name); }
  };
    
}

#endif

