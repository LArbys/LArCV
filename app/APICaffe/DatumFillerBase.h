/**
 * \file DatumFillerBase.h
 *
 * \ingroup Package_Name
 * 
 * \brief Class def header for a class DatumFillerBase
 *
 * @author kazuhiro
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __DATUMFILLERBASE_H__
#define __DATUMFILLERBASE_H__

#include "Processor/ProcessBase.h"
#include "DataFormat/Image2D.h"
#include "DataFormat/ROI.h"
namespace larcv {
  class ThreadDatumFiller;

  /**
     \class ProcessBase
     User defined class DatumFillerBase ... these comments are used to generate
     doxygen documentation!
  */
  class DatumFillerBase : public ProcessBase {
    friend class ThreadDatumFiller;
  public:
    
    /// Default constructor
    DatumFillerBase(const std::string name="DatumFillerBase");
    
    /// Default destructor
    ~DatumFillerBase(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    virtual bool is(const std::string question) const
    { return (question == "DatumFiller"); }

    size_t entries() const { return _nentries; }

    const std::vector<float>& data(bool image=true) const
    { return (image ? _image_data : _label_data); }

    virtual const std::vector<int> dim(bool image=true) const = 0;

    size_t entry_image_size() const { return _entry_image_size; }

    size_t entry_label_size() const { return _entry_label_size; }

    virtual void child_configure(const PSet&) = 0;

    virtual void child_initialize()  {}

    virtual void child_batch_begin() {}

    virtual void child_batch_end()   {}

    virtual void child_finalize()    {}

  protected:

    virtual void fill_entry_data (const larcv::EventBase* image, const larcv::EventBase* label) = 0;

    virtual size_t compute_image_size(const larcv::EventBase* image) = 0;

    virtual size_t compute_label_size(const larcv::EventBase* image) = 0;

    const std::vector<float>& entry_data(bool image=true) const 
    { return (image ? _entry_image_data : _entry_label_data); }

    std::vector<float> _entry_image_data;
    std::vector<float> _entry_label_data;
    ProductType_t _image_product_type;
    ProductType_t _label_product_type;

  private:

    void batch_begin();

    void batch_end();

    size_t _nentries;
    
    std::string _image_producer;
    std::string _label_producer;
    ProducerID_t _image_producer_id;
    ProducerID_t _label_producer_id;
    std::vector<float> _image_data;
    std::vector<float> _label_data;
    size_t _current_entry;
    size_t _entry_image_size;
    size_t _entry_label_size;
  };

}

#endif
/** @} */ // end of doxygen group 

