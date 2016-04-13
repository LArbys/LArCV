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

    void initialize();
    
    bool process();

    void batch_process();

    void finalize();

  private:

    IOManager _io;
    ProcessDriver _driver1;
    ProcessDriver _driver2;
    ImageHolder*  _proc1;
    ImageHolder*  _proc2;
    std::string   _proc1_name;
    std::string   _proc2_name;
    bool _prepared;
    size_t _num_proc1;
    size_t _num_proc2;
    size_t _num_proc_max;
    size_t _num_proc_frac;
  };
}
  
#endif
/** @} */ // end of doxygen group 

