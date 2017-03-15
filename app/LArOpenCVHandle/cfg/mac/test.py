import ROOT
tf=ROOT.TFile("ssnet_numu_nopp_notunique.root","READ")
for i in xrange(11):
    tf.EventTree.GetEntry(i)
    print "AAA--",i
    print tf.EventTree.Vertex3D_v.size()
    print tf.EventTree.ParticleCluster_vvv.size()
    print tf.EventTree.TrackClusterCompound_vvv.size()
