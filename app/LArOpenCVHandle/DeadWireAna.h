#ifndef __DEADWIREANA_H__
#define __DEADWIREANA_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"

#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larcv {

  class DeadWireAna : public ProcessBase {

  public:
    DeadWireAna(const std::string name="DeadWireAna");
    ~DeadWireAna(){}

    void configure(const PSet&);
    void initialize();
    bool process(IOManager& mgr);
    void finalize();

  private:

    TTree *_tree;

    int _run;
    int _subrun;
    int _event;
    int _entry;

    int _vertex_in_dead_plane0;
    int _vertex_in_dead_plane1;
    int _vertex_in_dead_plane2;
    int _vertex_in_dead;

    int _vertex_near_dead_plane0;
    int _vertex_near_dead_plane1;
    int _vertex_near_dead_plane2;
    int _vertex_near_dead;

    int _nearest_wire_error;

    std::string _ev_img2d_prod;
    std::string _seg_roi_prod;
    float _d_dead;

    larutil::SpaceChargeMicroBooNE _sce;
  };

  class DeadWireAnaProcessFactory : public ProcessFactoryBase {
  public:
    DeadWireAnaProcessFactory() { ProcessFactory::get().add_factory("DeadWireAna",this); }
    ~DeadWireAnaProcessFactory() {}
    ProcessBase* create(const std::string instance_name) { return new DeadWireAna(instance_name); }
  };

}

#endif
/** @} */ // end of doxygen group 

