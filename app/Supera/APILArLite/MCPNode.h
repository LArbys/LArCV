#ifndef __MCPNODE_H__
#define __MCPNODE_H__

#include <vector>

namespace larcv {
  namespace supera {

    class MCPNode {
    public:

      enum NodeSource_t { kTruth, kTrack, kShower };
      enum NodeOrigin_t { kNeutrino, kCosmic };

      MCPNode( int tid, int pid, int pdgcode, float t, int parent_node, NodeSource_t src, NodeOrigin_t o )
        : trackid(tid), parentid(pid), pdg(pdgcode), time(t), source(src), origin(o), idx_parentnode(parent_node)  {};
      virtual ~MCPNode() {};

      MCPNode( const MCPNode& src ) {
	trackid = src.trackid;
	parentid = src.parentid;
	pdg = src.pdg;
	time = src.time;
	source = src.source;
	origin = src.origin;
	idx_parentnode = src.idx_parentnode;
	for (int i=0; i<(int)src.idx_daughternodes.size(); i++)
	  idx_daughternodes.push_back( src.idx_daughternodes.at(i) );
      };

      bool operator<( const MCPNode& rhs ) {
	if ( time < rhs.time ) return true;
	return false;
      };

      int trackid;
      int parentid;
      int pdg;
      float time;
      NodeSource_t source;
      NodeOrigin_t origin;
      int idx_parentnode;
      std::vector<int> idx_daughternodes;
    };
  }
}


#endif
