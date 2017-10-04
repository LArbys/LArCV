/**
 * \file ReadRuiFile.h
 *
 * \ingroup Package_Name
 *
 * \brief Class def header for a class ReadRuiFile
 *
 * @author vgenty
 */

/** \addtogroup Package_Name

 @{*/
#ifndef __READRUIFILE_H__
#define __READRUIFILE_H__

#include "Processor/ProcessBase.h"
#include "Processor/ProcessFactory.h"
#include "AStarTracker.h"

namespace larcv {

    /**
     \class ProcessBase
     User defined class ReadRuiFile ... these comments are used to generate
     doxygen documentation!
     */
    class ReadRuiFile : public ProcessBase {

    public:

        /// Default constructor
        ReadRuiFile(const std::string name="ReadRuiFile");

        /// Default destructor
        ~ReadRuiFile(){}

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
     \class larcv::ReadRuiFileFactory
     \brief A concrete factory class for larcv::ReadRuiFile
     */
    class ReadRuiFileProcessFactory : public ProcessFactoryBase {
    public:
        /// ctor
        ReadRuiFileProcessFactory() { ProcessFactory::get().add_factory("ReadRuiFile",this); }
        /// dtor
        ~ReadRuiFileProcessFactory() {}
        /// creation method
        ProcessBase* create(const std::string instance_name) { return new ReadRuiFile(instance_name); }

    };
    
}

#endif
/** @} */ // end of doxygen group 

