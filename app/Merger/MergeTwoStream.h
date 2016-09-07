/**
 * \file MergeTwoStream.h
 *
 * \ingroup Merger
 * 
 * \brief Class def header for a class MergeTwoStream
 *
 * @author drinkingkazu
 */

/** \addtogroup Merger

    @{*/
#ifndef MERGETWOSTREAM_H
#define MERGETWOSTREAM_H

#include "ImageHolder.h"
#include "ImageMerger.h"
#include "Processor/ProcessDriver.h"

namespace larcv {
  /**
     \class MergeTwoStream
     User defined class MergeTwoStream ... these comments are used to generate
     doxygen documentation!
  */
  class MergeTwoStream : public larcv_base {
    
  public:
    
    /// Default constructor
    MergeTwoStream();
    
    /// Default destructor
    ~MergeTwoStream(){}

    void configure(std::string cfg_file);

    void override_input_file(const std::vector<std::string>& nu_flist = std::vector<std::string>(),
			     const std::vector<std::string>& cosmic_flist = std::vector<std::string>());
    void override_ana_file(std::string out_ana_fname);

    void override_output_file(std::string out_fname);

    void initialize();
    
    bool process();

    void batch_process();

    void finalize();

  private:

    ProcessDriver _nu_driver;
    ImageHolder*  _nu_proc;
    std::string   _nu_proc_name;
    
    ProcessDriver _cosmic_driver;
    ImageHolder*  _cosmic_proc;
    std::string   _cosmic_proc_name;

    ProcessDriver _merge_driver;
    ImageMerger*  _merge_proc;
    std::string   _merge_proc_name;
    
    bool _prepared;
    size_t _num_nu;
    size_t _num_cosmic;
    size_t _num_processed;
    size_t _num_input_max;
    size_t _num_output_max;
    size_t _num_frac;
  };
}
  
#endif
/** @} */ // end of doxygen group 

