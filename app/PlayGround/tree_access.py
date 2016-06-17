


from ROOT import TChain

treename = "image2d_tpc_tree"
filename = "supera_mc_muminus.root"

# Create TChain
ch=TChain(treename)

# Load file
ch.AddFile(filename)

# Get # entries
print "How many entries?", ch.GetEntries()

# Get entry 0
ch.GetEntry(0)

# EventImage2D object
br = None
exec('br = ch.%s' % treename.replace('_tree','_branch'))
print "EventImage2D object pointer:", br
