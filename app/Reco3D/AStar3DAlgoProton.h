#ifndef __ASTAR_3D_ALGOPROTON__
#define __ASTAR_3D_ALGOPROTON__

/**

 AStar algorithm assuming 3D!! grid points.

 Uses Image2D to hold image.

 **/

#include <iostream>
#include <queue>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <array>

// larcv
#include "DataFormat/Image2D.h"
#include "Base/PSet.h"
#include "AStar3DAlgo.h"

namespace larcv {

    // ALGO
    class AStar3DAlgoProton {

        AStar3DAlgoProton() { verbose=2; doPxValEstimate = false; };
    public:

        AStar3DAlgoProton( AStar3DAlgoConfig config ) {
            _config = config;
            setVerbose( _config.verbosity );
        };
        virtual ~AStar3DAlgoProton() {};

        void setVerbose( int v ) { verbose = v; };
        void setPixelValueEsitmation( bool doIt ) { doPxValEstimate = doIt; };

        std::vector<AStar3DNode> findpath( const std::vector<larcv::Image2D>& img_v,
                                          const std::vector<larcv::Image2D>& badch_v,
                                          const std::vector<larcv::Image2D>& tagged_v,
                                          const int start_row, const int goal_row,
                                          const std::vector<int>& start_cols,
                                          const std::vector<int>& goal_cols,
                                          int& goal_reached );

        std::vector<AStar3DNode> makeRecoPath( AStar3DNode* start, AStar3DNode* goal, bool& path_completed );

        void evaluateNeighborNodes( AStar3DNode* current,
                                   const AStar3DNode* start,
                                   const AStar3DNode* goal,
                                   const std::vector<larcv::Image2D>& img_v,
                                   const std::vector<larcv::Image2D>& badch_v,
                                   const std::vector<larcv::Image2D>& tagged_v,
                                   AStar3DNodePtrList& openset,
                                   AStar3DNodePtrList& closedset,
                                   Lattice& lattice );

        bool evaluteLatticePoint( const A3DPixPos_t& latticept,
                                 AStar3DNode* current,
                                 const AStar3DNode* start,
                                 const AStar3DNode* goal,
                                 const std::vector<larcv::Image2D>& img_v,
                                 const std::vector<larcv::Image2D>& badch_v,
                                 const std::vector<larcv::Image2D>& tagged_v,
                                 AStar3DNodePtrList& openset,
                                 AStar3DNodePtrList& closedset,
                                 Lattice& lattice );

        float distanceFromCentralLine( const std::vector<float>& start_tyz, const std::vector<float>& end_tyz, const std::vector<float>& testpt_tyz );

        const std::vector<larcv::Image2D>& getScoreImages() { return m_visualizedimgs; }

    protected:

        AStar3DAlgoConfig _config;
        int verbose;
        bool doPxValEstimate;
        
        std::vector< larcv::Image2D > m_visualizedimgs;
    };
    
    
}

#endif
