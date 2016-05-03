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

    bool is(const std::string& question) const;

    size_t rows() const { return _rows; }
    size_t cols() const { return _cols; }
    size_t channels() const { return _num_channels; }

    const std::vector<larcv::Image2D>& mean_image() const { return _mean_image_v; }
    const std::vector<float>& mean_adc()            const { return _mean_adc_v;   }
    const std::vector<float>& data()                const { return _data;         }
    const std::vector<float>& labels()              const { return _labels;       }

    virtual void child_configure(const PSet&) = 0;

    virtual void child_initialize()  {}

    virtual void child_batch_begin() {}

    virtual void child_batch_end()   {}

    virtual void child_finalize()    {}

  protected:

    virtual void set_dimension(const std::vector<larcv::Image2D>&) = 0;
    virtual void fill_entry_data(const std::vector<larcv::Image2D>&,const std::vector<larcv::ROI>&) = 0;
    const std::vector<float>& entry_data() const { return _entry_data; }
    std::vector<float> _entry_data;

    size_t _nentries;
    size_t _num_channels;
    size_t _rows;
    size_t _cols;
    float  _label;

  private:
    void batch_begin();
    void batch_end();
    
    std::string _image_producer;
    std::string _roi_producer;
    std::vector<float> _data;
    std::vector<float> _labels;
    ProducerID_t _image_producer_id;
    ProducerID_t _roi_producer_id;
    size_t _current_entry;
    size_t _entry_data_size;
    std::vector<larcv::Image2D> _mean_image_v;
    std::vector<float> _mean_adc_v;
  };

}

#endif
/** @} */ // end of doxygen group 

