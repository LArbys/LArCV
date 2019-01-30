/**
 * \file CircularBuffer.h
 *
 * \ingroup MultiThreadTools
 * 
 * \brief Class def header for a class CircularBuffer
 *
 * @author kazuhiro
 */

/** \addtogroup MultiThreadTools

    @{*/
#ifndef LARCV_CIRCULARBUFFERBASE_H
#define LARCV_CIRCULARBUFFERBASE_H

#include <iostream>
#include <thread>
#include <vector>
#include <memory>

namespace larcv {

  enum BufferState_t {
    kEmpty,
    kInit,
    kReady,
    kClean,
    kUndefined
  };
  
  /**
     \class CircularBufferBase
     User defined class CircularBufferBase ... these comments are used to generate
     doxygen documentation!
  */
  template <class T>
  class CircularBufferBase{
    
  public:
    
    /// Default constructor
    CircularBufferBase(size_t num_buf=10);
    
    /// Default destructor
    virtual ~CircularBufferBase();

    size_t size() const { return _buffer_v.size(); }
    
    BufferState_t state(const size_t id) const;
    
    bool flush(const size_t id);
    
    bool init(const size_t id);
    
    bool busy(const size_t id, const bool raise=false) const;
    
    T& get(const size_t id);

    void lock(const size_t id);

    void unlock(const size_t id);

  protected:

    std::vector<std::shared_ptr<T> > _buffer_v;

    virtual std::shared_ptr<T> construct() const;
    
    virtual void destruct(std::shared_ptr<T>);
    
  private:
    
    void check_id(const size_t id) const;
    
    void fill_buffer(const size_t id);
    
    void clean_buffer(const size_t id);
    
    std::vector<larcv::BufferState_t> _state_v;
    std::vector<bool> _lock_v;
#ifndef __CLING__
#ifndef __CINT__
    std::vector<std::thread> _thread_v;
#endif
#endif
  };
}

#endif
/** @} */ // end of doxygen group 

