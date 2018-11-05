import sys,os

if len(sys.argv) != 7:
    print ""
    print "SIG_PKL         = str(sys.argv[1])"
    print "COS_PKL         = str(sys.argv[2])"
    print "BNB_PKL         = str(sys.argv[3])"
    print "SIG_MCINFO_ROOT = str(sys.argv[4])"
    print "BNB_MCINFO_ROOT = str(sys.argv[5])"
    print "OUT_DIR         = str(sys.argv[6])"
    print ""
    sys.exit(1)

SIG_PKL         = str(sys.argv[1])
COS_PKL         = str(sys.argv[2])
BNB_PKL         = str(sys.argv[3])
SIG_MCINFO_ROOT = str(sys.argv[4])
BNB_MCINFO_ROOT = str(sys.argv[5])

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)

from lib.ll_functions import LLpcY_v, LLemY_v
from lib.xgb_functions import prepare_xgb, XGB_DL

xgb_dl = XGB_DL("xgb_dl",{},0.8,50,LLpcY_v,LLemY_v)

df_sig, df_cos, df_bnb = prepare_xgb(SIG_PKL,
                                     COS_PKL,
                                     BNB_PKL,
                                     SIG_MCINFO_ROOT,
                                     BNB_MCINFO_ROOT)

xgb_dl.set_df(df_sig,df_cos,df_bnb)

xgb_dl.train_models()

sys.exit(0)
