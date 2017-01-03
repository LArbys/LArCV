import os,sys

# 7/13/2016 tmw: 62359 events

samweb_definition = "prod_ext_reco_inclusive_v2"
pdeflist = os.popen("samweb list-definition-files %s" % ( samweb_definition ))
deflist = pdeflist.readlines()

totentries = 0
for entry in deflist:
    
    pmeta = os.popen("samweb get-metadata %s"%(entry.strip()))
    meta = pmeta.readlines()

    for line in meta:
        if line.split(":")[0].strip()=="Event Count":
            file_entries = int( line.split(":")[1].strip() )
            totentries += file_entries
            print entry.strip(),": ",file_entries," TOTAL=",totentries
            break

print "Total entries: ",totentries
