#ifndef __LARCVUTIL_RANDOM_GAUS_H__
#define __LARCVUTIL_RANDOM_GAUS_H__

#include <random>
#include <cmath>
#include <thread>
#include <vector>
#include <exception>

namespace larcv {

  class RandomGaus {
  public:
    RandomGaus(double mean=1., double sigma=0.1, size_t pool_size=10000000);

    //RandomGaus(const RandomGaus& rhs) = delete;
    RandomGaus(RandomGaus&& rhs) : _pool(std::move(rhs._pool))
				 , _mean(rhs._mean)
				 , _sigma(rhs._sigma)
				 , _th(std::move(rhs._th))
    {}

    ~RandomGaus(){ if(_th.joinable()) _th.join(); }

    void reset(double mean, double sigma);

    void start_filling();

    void get(std::vector<float>& container);

  private:

    void _fill_();
    
    std::vector<float> _pool;
    double _mean;
    double _sigma;
    std::thread _th;
  }; 
}

#endif
