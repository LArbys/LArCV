/**
 * \file ReadJarrettFile.h
 *
 * \ingroup Package_Name
 *
 * \brief Class def header for a class ReadJarrettFile
 *
 * @author hourlier
 */

/** \addtogroup Package_Name

 @{*/
#ifndef __READJARRETTFILE_H__
#define __READJARRETTFILE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

    /**
     \class ProcessBase
     User defined class ReadJarrettFile ... these comments are used to generate
     doxygen documentation!
     */
    class ReadJarrettFile : public ProcessBase {

    public:

        /// Default constructor
        ReadJarrettFile(const std::string name="ReadJarrettFile");

        /// Default destructor
        ~ReadJarrettFile(){}

        void configure(const PSet&);

        void initialize();

        bool process(IOManager& mgr);

        bool IsGoodVertex(int run, int subrun, int event, int ROIid, int vtxID);
        void ReadVertexFile(std::string filename);

        void finalize();

    private :
        int iTrack;
        larcv::AStarTracker tracker;
        std::vector< std::vector<int> > _vertexInfo;

    };

    /**
     \class larcv::ReadJarrettFileFactory
     \brief A concrete factory class for larcv::ReadJarrettFile
     */
    class ReadJarrettFileProcessFactory : public ProcessFactoryBase {
    public:
        /// ctor
        ReadJarrettFileProcessFactory() { ProcessFactory::get().add_factory("ReadJarrettFile",this); }
        /// dtor
        ~ReadJarrettFileProcessFactory() {}
        /// creation method
        ProcessBase* create(const std::string instance_name) { return new ReadJarrettFile(instance_name); }

    };
    
}

#endif
/** @} */ // end of doxygen group 

