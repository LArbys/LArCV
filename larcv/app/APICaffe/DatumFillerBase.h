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
    enum FillerDataType_t {
      kFillerImageData,
      kFillerLabelData,
      kFillerWeightData,
      kFillerMultiplicityData
    };
  public:
    
    /// Default constructor
    DatumFillerBase(const std::string name="DatumFillerBase");
    
    /// Default destructor
    ~DatumFillerBase(){}

    void configure(const PSet&);

    void initialize();

    bool process(IOManager& mgr);

    void finalize();

    const std::string& producer(FillerDataType_t type) const;

    virtual bool is(const std::string question) const;

    size_t entries() const;

    const std::vector<std::vector<larcv::ImageMeta> >& meta() const;

    const std::vector<float>& data(FillerDataType_t dtype = kFillerImageData) const;

    virtual const std::vector<int> dim(bool image=true) const = 0;

    size_t entry_image_size()  const { return _entry_image_size;  }

    size_t entry_label_size()  const { return _entry_label_size;  }

    size_t entry_weight_size() const { return _entry_weight_size; }

    size_t entry_multiplicity_size() const { return _entry_multiplicity_size; }

    virtual void child_configure(const PSet&) = 0;

    virtual void child_initialize()  {}

    virtual void child_batch_begin() {}

    virtual void child_batch_end()   {}

    virtual void child_finalize()    {}

  protected:

    virtual void fill_entry_data (const larcv::EventBase* image, 
				  const larcv::EventBase* label,
				  const larcv::EventBase* weight,
				  const larcv::EventBase* multiplicity) = 0;

    virtual size_t compute_image_size(const larcv::EventBase* image) = 0;

    virtual size_t compute_label_size(const larcv::EventBase* label) = 0;

    virtual size_t compute_multiplicity_size(const larcv::EventBase* multiplicity) = 0;

    const std::vector<float>& entry_data(FillerDataType_t dtype = kFillerImageData) const;

    const std::vector<larcv::ImageMeta>& entry_meta() const
    { return _entry_meta_data;}

    std::vector<float> _entry_image_data;
    std::vector<float> _entry_label_data;
    std::vector<float> _entry_weight_data;
    std::vector<float> _entry_multiplicity_data;
    ProductType_t _image_product_type;
    ProductType_t _label_product_type;
    ProductType_t _weight_product_type;
    ProductType_t _multiplicity_product_type;
    std::vector<larcv::ImageMeta>  _entry_meta_data;

  private:

    void batch_begin();

    void batch_end();

    size_t _nentries;

    std::string _image_producer;
    std::string _label_producer;
    std::string _weight_producer;
    std::string _multiplicity_producer;
    ProducerID_t _image_producer_id;
    ProducerID_t _label_producer_id;
    ProducerID_t _weight_producer_id;
    ProducerID_t _multiplicity_producer_id;
    std::vector<float> _image_data;
    std::vector<float> _label_data;
    std::vector<float> _weight_data;
    std::vector<float> _multiplicity_data;
    std::vector<std::vector<larcv::ImageMeta> > _meta_data;
    size_t _current_entry;
    size_t _entry_image_size;
    size_t _entry_label_size;
    size_t _entry_weight_size;
    size_t _entry_multiplicity_size;
  };

}

#endif
/** @} */ // end of doxygen group 

