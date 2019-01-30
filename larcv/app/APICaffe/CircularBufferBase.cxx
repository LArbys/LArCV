#ifndef CIRCULARBUFFERBASE_CXX
#define CIRCULARBUFFERBASE_CXX

#include "CircularBufferBase.h"
#include <unistd.h>
namespace larcv {

  template <class T>
  CircularBufferBase<T>::CircularBufferBase(size_t num_buf)
    : _buffer_v ( num_buf, nullptr )
    , _state_v  ( num_buf, kEmpty  )
    , _lock_v   ( num_buf, false   )
    , _thread_v ( num_buf          )
  {}

  template <class T>
  CircularBufferBase<T>::~CircularBufferBase()
  {
    // loop over buffers
    for(size_t i=0; i<_buffer_v.size(); ++i) {
      // if thread is running, wait
      if(_thread_v[i].joinable())
	_thread_v[i].join();
      // then flush
      if( _state_v[i] == kReady)
	flush(i);
    }
    // make sure destructor holds till flushing is done
    for(size_t i=0; i<_buffer_v.size(); ++i)
      if(_thread_v[i].joinable()) _thread_v[i].join();

    /*
    for(size_t i=0; i<_thread_v.size(); ++i) {
      std::cout<<i<<" ... state " << _state_v[i]<< " ... "<<(_thread_v[i].joinable() ? "joinable" : "not joinable")<<std::endl;
    }
    */
  }

  template <class T>
  void CircularBufferBase<T>::lock(const size_t id)
  {
    busy(id,true);
    _lock_v[id] = true;
  }

  template <class T>
  void CircularBufferBase<T>::unlock(const size_t id)
  {
    busy(id,true);
    _lock_v[id] = true;
  }

  template <class T>
  BufferState_t CircularBufferBase<T>::state(const size_t id) const
  {
    check_id(id);
    return _state_v[id];
  }

  template <class T>
  bool CircularBufferBase<T>::flush(const size_t id)
  {
    busy(id);
    if(_thread_v[id].joinable()) _thread_v[id].join();
    std::thread t(&CircularBufferBase<T>::clean_buffer, this, id);
    _thread_v[id] = std::move(t);
    usleep(500);
    return true;
  }

  template <class T>
  bool CircularBufferBase<T>::init(const size_t id)
  {
    busy(id,true);
    if(_thread_v[id].joinable()) _thread_v[id].join();
    std::thread t(&CircularBufferBase<T>::fill_buffer, this, id);
    _thread_v[id] = std::move(t);
    usleep(500);
    return true;
  }

  template <class T>
  bool CircularBufferBase<T>::busy(const size_t id, const bool raise) const
  {
    check_id(id);
    bool state = (_state_v[id] == kClean || _state_v[id] == kInit);
    
    if(state && raise) {
      std::cerr << "Buffer ID " << id << " is in busy state..." << std::endl;
      throw std::exception();
    }
    if(!state && _lock_v[id]) {
      state = true;
      if(raise) {
	std::cerr << "Buffer ID " << id << " is in locked state..." << std::endl;
	throw std::exception();
      }
    }
    
    return state;
  }

  template <class T>
  T& CircularBufferBase<T>::get(const size_t id)
  {
    check_id(id);
    if(busy(id)) {
      std::cerr << "Buffer " << id << " is busy!" << std::endl;
      throw std::exception();
    }
    if(!_buffer_v[id]) {
      std::cerr << "Buffer " << id << " does not exist!" << std::endl;
      throw std::exception();
    }
    return *(_buffer_v[id]);
  }

  template <class T>
  void CircularBufferBase<T>::check_id(const size_t id) const
  { if( id < _state_v.size() ) return;
    std::cerr << "Invalid buffer id requested!" << std::endl;
    throw std::exception();
  }

  template <class T>
  std::shared_ptr<T> CircularBufferBase<T>::construct() const
  { return std::shared_ptr<T>(new T); }

  template <class T>
  void CircularBufferBase<T>::destruct(std::shared_ptr<T> ptr)
  { }
  
  template <class T>
  void CircularBufferBase<T>::fill_buffer(const size_t id)
  {
    if(_state_v[id] != kEmpty) {
      std::cerr << "Buffer id " << id << " is not in kEmpty state!" << std::endl;
      throw std::exception();
    }
    _state_v  [id] = kInit;
    _buffer_v [id] = this->construct();
    _state_v  [id] = kReady;
  }

  template <class T>
  void CircularBufferBase<T>::clean_buffer(const size_t id)
  {
    if(_state_v[id] != kReady) {
      std::cerr << "Buffer id " << id << " is not in kReady state!" << std::endl;
      throw std::exception();
    }
    _state_v[id]  = kClean;
    this->destruct(_buffer_v[id]);
    _buffer_v[id] = nullptr;
    _state_v[id]  = kEmpty;
  }
}

template class larcv::CircularBufferBase<std::vector<double> >;

#endif
