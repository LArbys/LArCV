# define ROOTLIB

set( LARCV_LIB -L$ENV{LARCV_LIBDIR} -lLArCVBase -lLArCVDataFormat -lLArCVProcessor )

include_directories ( $ENV{LARCV_INCDIR} )

link_libraries( ${LIB_NAME} -L$ENV{BOOST_LIB} -lboost_system ${ROOTLIB} ${LARCV_LIB}) # All execs need this.

add_subdirectory(job)

art_make( BASENAME_ONLY

          EXCLUDE Supera_module.cc
          LIBRARY_NAME     LArSupera
          LIB_LIBRARIES
                   larcore_Geometry
                   #uboone_Geometry
                   larcore_Geometry_Geometry_service
                   larsim_Simulation
                   lardataobj_RecoBase
                   nusimdata_SimulationBase
                   lardataobj_MCBase
                   larcoreobj_SummaryData
                   lardataobj_RawData
                   lardata_RecoObjects
                   lardata_DetectorInfoServices_DetectorClocksServiceStandard_service
                   ${SIMULATIONBASE}
                   ${LARRECO_LIB}
                   ${LARDATA_LIB}
                   ${ART_FRAMEWORK_CORE}
                   ${ART_FRAMEWORK_PRINCIPAL}
                   ${ART_FRAMEWORK_BASIC}
                   ${ART_FRAMEWORK_SERVICES_REGISTRY}
                   ${ART_FRAMEWORK_SERVICES_OPTIONAL}
                   ${ART_FRAMEWORK_SERVICES_OPTIONAL_TFILESERVICE_SERVICE}
                   ${ART_PERSISTENCY_COMMON}
                   ${ART_PERSISTENCY_PROVENANCE}
                   ${ART_UTILITIES}
                   ${MF_MESSAGELOGGER}
                   ${MF_UTILITIES}
                   ${FHICLCPP}
                   ${CETLIB}
                   ${ROOT_GEOM}
                   ${ROOT_XMLIO}
                   ${ROOT_GDML}
                   ${ROOT_BASIC_LIB_LIST}
)

simple_plugin( Supera "module"
                   LArSupera
                   larcore_Geometry
                   #uboone_Geometry
                   larcore_Geometry_Geometry_service
                   larsim_Simulation
                   nusimdata_SimulationBase
                   lardata_Utilities
                   lardataobj_MCBase
                   larcoreobj_SummaryData
                   lardataobj_RawData
                   lardataobj_RecoBase
                   lardata_RecoObjects
                   larevt_Filters
                   larevt_CalibrationDBI_IOVData
                   larevt_CalibrationDBI_Providers
                   ${UBOONECODE_LIB}
                   ${SIMULATIONBASE}
                   ${LARRECO_LIB}
                   ${LARDATA_LIB}
                   ${ART_FRAMEWORK_CORE}
                   ${ART_FRAMEWORK_PRINCIPAL}
                   ${ART_FRAMEWORK_BASIC}
                   ${ART_FRAMEWORK_SERVICES_REGISTRY}
                   ${ART_FRAMEWORK_SERVICES_OPTIONAL}
                   ${ART_FRAMEWORK_SERVICES_OPTIONAL_TFILESERVICE_SERVICE}
                   ${ART_PERSISTENCY_COMMON}
                   ${ART_PERSISTENCY_PROVENANCE}
                   ${ART_UTILITIES}
                   ${MF_MESSAGELOGGER}
                   ${MF_UTILITIES}
                   ${FHICLCPP}
                   ${CETLIB}
                   ${ROOT_GEOM}
                   ${ROOT_XMLIO}
                   ${ROOT_GDML}
                   ${ROOT_BASIC_LIB_LIST}
             )

# install_headers()
install_fhicl()
install_source()
