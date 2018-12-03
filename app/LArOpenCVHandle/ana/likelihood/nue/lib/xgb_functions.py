import os,sys,time
import xgboost as xgb
import numpy as np
import pandas as pd
import root_numpy as rn
from common_functions import bless_scedr

def pred_xgb(df,pdf_v,bst):
        data = df[pdf_v].values
        dmat = xgb.DMatrix(data)
        pred = bst.predict(dmat)
        return pred[:,0]


def prepare_xgb(df1,df2,df3,mcinfo1,mcinfo2):
        
        print "Start"
        
        precut = "(passed_ll_precuts_y==1)"
        SS1_   = "(locv_scedr<5 and locv_selected1L1P==1 and locv_energyInit<1e3)"
        SS2_   = "(mcinfo_scedr<5 and parentPDG==14)"
        
        df_sig = pd.read_pickle(df1)
        print "Read df_sig sz=",df_sig.index.size
        df_sig.query(precut,inplace=True)
        print "... after precut sz=",df_sig.index.size
        
        df_cos = pd.read_pickle(df2)
        print "Read df_cos sz=",df_cos.index.size
        df_cos.query(precut,inplace=True)
        print "... after precut sz=",df_cos.index.size
        
        df_bnb = pd.read_pickle(df3)
        print "Read df_bnb sz=",df_bnb.index.size
        df_bnb.query(precut,inplace=True)
        print "... after precut sz=",df_bnb.index.size
        
        df_sig_mcinfo = pd.DataFrame(rn.root2array(mcinfo1,treename="EventMCINFO_DL"))
        df_bnb_mcinfo = pd.DataFrame(rn.root2array(mcinfo2,treename="EventMCINFO_DL"))
        
        print "Read df_sig_mcinfo =",df_sig_mcinfo.index.size
        print "Read df_bnb_mcinfo =",df_bnb_mcinfo.index.size
        
        RSE=['run','subrun','event']
        
        df_sig.set_index(RSE,inplace=True)
        df_bnb.set_index(RSE,inplace=True)
        
        df_sig_mcinfo.set_index(RSE,inplace=True)
        df_bnb_mcinfo.set_index(RSE,inplace=True)
        
        df_sig = df_sig.join(df_sig_mcinfo,rsuffix="mcinfo_")
        df_bnb = df_bnb.join(df_bnb_mcinfo,rsuffix="mcinfo_")
        
        df_sig.reset_index(inplace=True)
        df_bnb.reset_index(inplace=True)
        
        df_sig['mcinfo_scedr'] = df_sig.apply(bless_scedr,axis=1)
        df_bnb['mcinfo_scedr'] = df_bnb.apply(bless_scedr,axis=1)
        
        df_sig.query(SS1_,inplace=True)
        df_bnb.query(SS2_,inplace=True)
        
        df_sig = df_sig.sort_values('mcinfo_scedr',ascending=True).groupby(RSE).head(1).copy()
        df_bnb = df_bnb.sort_values('mcinfo_scedr',ascending=True).groupby(RSE).head(1).copy()
        
        return (df_sig,df_cos,df_bnb)


class XGB_DL:

    def __init__(self,name,param,train_split,num_trees,pc_pdf_v,em_pdf_v):

        self.name = str(name)
        self.num_trees = int(num_trees)
        self.train_split = float(train_split)

        if param == {}:
            self.param = {
                'max_depth': 3,  # the maximum depth of each tree
                'eta': 0.3,  # the training step for each iteration
                'silent': 1,  # logging mode - quiet
                'objective': 'multi:softprob',  # error evaluation for multiclass training
                'num_class': 2 # the number of classes that exist in this datset
            }  
        else:
            self.param = dict(param)
        
        self.init = False

        self.bst_pc = xgb.Booster()
        self.bst_em = xgb.Booster()

        self.train_dmat_pc = xgb.DMatrix(None)
        self.test_dmat_pc = xgb.DMatrix(None)

        self.train_dmat_em = xgb.DMatrix(None)
        self.test_dmat_em = xgb.DMatrix(None)

        self.pc_pdf_v = pc_pdf_v
        self.em_pdf_v = em_pdf_v
        
        self.train_sig_acc = float(-1)
        self.train_bkg_acc = float(-1)

        self.test_sig_acc = float(-1)
        self.test_bkg_acc = float(-1)


        return

        
    def set_df(self,df_sig,df_cos,df_bnb):
        self.df_sig = df_sig
        self.df_cos = df_cos
        self.df_bnb = df_bnb
        self.init = True

        return

    def train_models(self):
        
        if self.init == False:
            print "Must call set_df or prepare_df first" 
            return 

        bst_pc, train_dmat_pc, test_dmat_pc = self.train_xgb(self.df_sig,self.df_cos,self.pc_pdf_v)
        bst_em, train_dmat_em, test_dmat_em = self.train_xgb(self.df_sig,self.df_bnb,self.em_pdf_v)
        
        self.bst_pc = bst_pc
        self.bst_em = bst_em

        self.train_dmat_pc = train_dmat_pc
        self.test_dmat_pc  = test_dmat_pc

        self.train_dmat_em = train_dmat_em
        self.test_dmat_em  = test_dmat_em
        
        return


    def train_xgb(self,df_sig,df_bkg,pdf_v):

        df_sig_v = df_sig[pdf_v].values
        df_bkg_v = df_bkg[pdf_v].values
        
        n_sig = df_sig_v.shape[0]
        n_bkg = df_bkg_v.shape[0]
        
        sig_indices = np.random.permutation(n_sig)
        bkg_indices = np.random.permutation(n_bkg)

        n_sig_train = int(float(n_sig)*self.train_split+0.5)
        n_bkg_train = int(float(n_bkg)*self.train_split+0.5)
        
        train_sig_idx, test_sig_idx = sig_indices[:n_sig_train], sig_indices[n_sig_train:]
        train_bkg_idx, test_bkg_idx = bkg_indices[:n_bkg_train], bkg_indices[n_bkg_train:]
        
        n_sig_train = train_sig_idx.size
        n_bkg_train = train_bkg_idx.size
        
        n_sig_test = test_sig_idx.size
        n_bkg_test = test_bkg_idx.size

        df_sig_train = df_sig_v[train_sig_idx]
        df_bkg_train = df_bkg_v[train_bkg_idx]
        
        df_sig_test  = df_sig_v[test_sig_idx]
        df_bkg_test  = df_bkg_v[test_bkg_idx]
        
        train_data_v  = np.vstack((df_sig_train,df_bkg_train))
        test_data_v   = np.vstack((df_sig_test,df_bkg_test))
        
        train_label_v = np.hstack((np.zeros(n_sig_train),np.ones(n_bkg_train)))
        test_label_v  = np.hstack((np.zeros(n_sig_test),np.ones(n_bkg_test)))
        
        train_rand_v = np.random.permutation(n_sig_train+n_bkg_train)
        test_rand_v  = np.random.permutation(n_sig_test+n_bkg_test)

        train_data_v  = train_data_v[train_rand_v]
        train_label_v = train_label_v[train_rand_v]

        test_data_v  = test_data_v[test_rand_v]
        test_label_v = test_label_v[test_rand_v]
        
        print "Prepare DMatrix"
        print "n_sig=",train_sig_idx.size
        print "n_bkg=",train_bkg_idx.size
        stime = time.time()
        train_dmat = xgb.DMatrix(train_data_v, label=train_label_v)
        etime = time.time()
        print "(%f) sec" % (etime - stime)
        
        print "n_sig_test=",test_sig_idx.size
        print "n_bkg_test=",test_bkg_idx.size
        stime = time.time()
        test_dmat = xgb.DMatrix(test_data_v, label=test_label_v)
        etime = time.time()
        print "(%f) sec" % (etime - stime)
        print
        
        print "Training"
        print "param=",self.param
        print "num_trees=",self.num_trees
        stime = time.time()
        bst = xgb.train(self.param, train_dmat, num_boost_round=self.num_trees)
        etime = time.time()
        print "(%f) sec" % (etime - stime)
        print

        print "Testing"
        print "train sample"
        stime = time.time()
        pred_train = bst.predict(train_dmat)
        etime = time.time()
        sig_train_rand_index = np.where(train_label_v==0)[0]
        bkg_train_rand_index = np.where(train_label_v==1)[0]
        self.train_sig_acc = np.where(pred_train[sig_train_rand_index][:,0]>0.5)[0].shape[0]/float(n_sig_train)
        self.train_bkg_acc = np.where(pred_train[bkg_train_rand_index][:,0]<0.5)[0].shape[0]/float(n_bkg_train)
        print "sig acc=",self.train_sig_acc
        print "bkg acc=",self.train_bkg_acc
        print "(%f) sec" % (etime - stime)
        print "test sample"
        stime = time.time()
        pred_test = bst.predict(test_dmat)
        etime = time.time()
        sig_test_rand_index = np.where(test_label_v==0)[0]
        bkg_test_rand_index = np.where(test_label_v==1)[0]
        self.test_sig_acc = np.where(pred_test[sig_test_rand_index][:,0]>0.5)[0].shape[0]/float(n_sig_test)
        self.test_bkg_acc = np.where(pred_test[bkg_test_rand_index][:,0]<0.5)[0].shape[0]/float(n_bkg_test)
        print "sig acc=",self.test_sig_acc
        print "bkg acc=",self.test_bkg_acc
        print "(%f) sec" % (etime - stime)
        print

        return (bst, train_dmat, test_dmat)

        
