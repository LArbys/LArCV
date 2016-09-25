/**
 * \file ANNAlgo.h
 *
 * \ingroup ANN
 * 
 * \brief Interface to ANN tree search package
 *
 * @author twongjirad
 */

/** \addtogroup Package_Name

    @{*/
#ifndef __ANN_ALGO__
#define __ANN_ALGO__

#include <ANN/ANN.h>
#include <vector>
#include <string>

namespace ann {

  class ANNAlgo {

  public:

    ANNAlgo( int npoints, int ndims );
    virtual ~ANNAlgo();

    void setPoint( int idx, const std::vector<double>& point ); ///< add point, which is a list of [ndims] doubles, returns point index
    void getPoint( int idx, std::vector<double>& point );       ///< get point, which is a list of [ndims] doubles, returns point index
    ANNpoint getPoint( int idx );
    std::vector<int> regionQuery( int idx, double eps2, double approx );
    static void cleanup() { annClose(); };
    void dump( std::string outfile );
    void printdata();
    void initialize();
    void deinitialize();

  protected:

    int fNdims;    ///< number of dimensions
    int fNpoints;  ///< number of points
    bool _init;

    ANNcoord* datablock;
    ANNpoint* points;
    ANNpointArray array;

    ANNbd_tree* bdtree;

    void alloc_data_block();
    void dealloc_data_block();

  };


}

#endif

/** @} */ // end of doxygen group 
