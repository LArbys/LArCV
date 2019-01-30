# CORE
sed -i 's|\"Base/|\"larcv/core/Base/|g' *.cxx
sed -i 's|\"Base/|\"larcv/core/Base/|g' *.h

sed -i 's|\"DataFormat/|\"larcv/core/DataFormat/|g' *.cxx
sed -i 's|\"DataFormat/|\"larcv/core/DataFormat/|g' *.h

sed -i 's|\"CPPUtil/|\"larcv/core/CPPUtil/|g' *.cxx
sed -i 's|\"CPPUtil/|\"larcv/core/CPPUtil/|g' *.h

sed -i 's|\"Processor/|\"larcv/core/Processor/|g' *.cxx
sed -i 's|\"Processor/|\"larcv/core/Processor/|g' *.h

sed -i 's|\"ROOTUtil/|\"larcv/core/ROOTUtil/|g' *.cxx
sed -i 's|\"ROOTUtil/|\"larcv/core/ROOTUtil/|g' *.h

sed -i 's|\"CVUtil/|\"larcv/core/CVUtil/|g' *.cxx
sed -i 's|\"CVUtil/|\"larcv/core/CVUtil/|g' *.h

sed -i 's|\"PyUtil/|\"larcv/core/PyUtil/|g' *.cxx
sed -i 's|\"PyUtil/|\"larcv/core/PyUtil/|g' *.h

sed -i 's|\"TorchUtil/|\"larcv/core/TorchUtil/|g' *.cxx
sed -i 's|\"TorchUtil/|\"larcv/core/TorchUtil/|g' *.h

# APP
sed -i 's|\"ImageMod/|\"larcv/app/ImageMod/|g' *.cxx
sed -i 's|\"ImageMod/|\"larcv/app/ImageMod/|g' *.h
