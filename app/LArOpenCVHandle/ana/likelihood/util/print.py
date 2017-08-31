import pandas as pd

def print_stats(name,comb_df,cut_df):
    print "@ sample",name

    print
    if comb_df is not None:
        print "N input vertices....",comb_df.index.size
        print "N input events......",len(comb_df.groupby(rse))
    if cut_df is not None:
        print "N cut vertices......",cut_df.index.size
        print "N cut events........",len(cut_df.groupby(rse))
    print

    scedr = 5
    print "Good vertex scedr < 5"
    if comb_df is not None:
        print "N input vertices....",comb_df.query("scedr<@scedr").index.size
        print "N input events......",len(comb_df.query("scedr<@scedr").groupby(rse))
    if cut_df is not None:
        print "N cut vertices......",cut_df.query("scedr<@scedr").index.size
        print "N cut events........",len(cut_df.query("scedr<@scedr").groupby(rse))
    print

    print "Bad vertex scedr > 5"
    if comb_df is not None:
        print "N input vertices....",comb_df.query("scedr>@scedr").index.size
        print "N input events......",len(comb_df.query("scedr>@scedr").groupby(rse))
    if cut_df is not None:
        print "N cut vertices......",cut_df.query("scedr>@scedr").index.size
        print "N cut events........",len(cut_df.query("scedr>@scedr").groupby(rse))
    print

