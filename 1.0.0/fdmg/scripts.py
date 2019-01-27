'''
Created on May 16, 2018

@author: cef

significant scripts for calculating damage within the ABMRI framework
    for secondary data loader scripts, see fdmg.datos.py


'''
#===============================================================================
# # IMPORT STANDARD MODS -------------------------------------------------------
#===============================================================================
import logging, os,  time, re, math, copy, gc, weakref

"""
unused
sys, imp,
"""

import pandas as pd
import numpy as np

import scipy.integrate

#===============================================================================
# shortcuts
#===============================================================================
#from collections import OrderedDict
from hp.dict import MyOrderedDict as OrderedDict

from weakref import WeakValueDictionary as wdict
from weakref import proxy

from hp.basic import OrderedSet

#===============================================================================
#  IMPORT CUSTOM MODS ---------------------------------------------------------
#===============================================================================
import hp.plot

import hp.basic

import hp.pd


import hp.oop

import hp.data

import fdmg.datos as datos

import matplotlib.pyplot as plt
import matplotlib
#import matplotlib.animation as animation #load the animation module (with the new search path)

import udev.scripts

mod_logger = logging.getLogger(__name__)
mod_logger.debug('initilized')
#===============================================================================
#module level defaults ------------------------------------------------------
#===============================================================================
#datapars_cols = [u'dataname', u'desc', u'datafile_tailpath', u'datatplate_tailpath', u'trim_row'] #headers in the data tab

datafile_types_list = ['.csv', '.xls']


idx = pd.IndexSlice

       

class Fdmg( #flood damage model
           hp.sel.Sel_controller, #no init
           hp.dyno.Dyno_wrap, #add some empty containers
           hp.plot.Plot_o, #build the label
           hp.sim.Sim_model, #Sim_wrap: attach the reset_d. Sim_model: inherit attributes
           hp.oop.Trunk_o, #no init
                            #Parent_cmplx: attach empty kids_sd
                            #Parent: set some defaults
           hp.oop.Child): 
    """
    #===========================================================================
    # INPUTS
    #===========================================================================
    pars_path ==> pars_file.xls
        main external parameter spreadsheet.
        See description in file for each column
        
        dataset parameters
        tab = 'data'. expected columns: datapars_cols
        
        session parameters
        tab = 'gen'. expected rows: sessionpars_rows
            
    """
    #===========================================================================
    # program parameters
    #===========================================================================
    name = 'fdmg'

    #list of attribute names to try and inherit from the session
    try_inherit_anl = set(['ca_ltail', 'ca_rtail', 'mind', \
                       'dbg_fld_cnt', 'legacy_binv_f', 'gis_area_max', \
                       'fprob_mult', 'flood_tbl_nm', 'gpwr_aep', 'dmg_rat_f',\
                       'joist_space', 'G_anchor_ht', 'bsmt_opn_ht_code','bsmt_egrd_code', \
                       'damp_func_code', 'cont_val_scale', 'hse_skip_depth', \
                        'area_egrd00', 'area_egrd01', 'area_egrd02',
                        'fhr_nm', 'write_fdmg_sum', 'dfeat_xclud_price', 'write_fdmg_sum_fly'])
    

    
    fld_aep_spcl = 100 #special flood to try and include in db runs
    bsmt_egrd   = 'wet' #default value for bsmt_egrd
    
    legacy_binv_f = True #flag to indicate that the binv is in legacy format (use indicies rather than column labels)
    
    gis_area_max = 3500
    
    
    #lsit of data o names expected on the fdmg tab
    
    
    #state = 'na' #for tracking what flood aep is currently in the model
    

    
    'consider allowing the user control of these'
    gis_area_min        = 5
    gis_area_max        = 5000
    
    write_fdmg_sum_fly = False
    write_dmg_fly_first = True #start off to signifiy first run
    #===========================================================================
    # debuggers
    #===========================================================================
    beg_hist_df = None
    #===========================================================================
    # user provided values
    #===========================================================================
    #legacy pars
    floor_ht = 0.0
    
    
    
    mind     = '' #column to match between data sets and name the house objects
    
    #EAD calc
    ca_ltail    ='flat'
    ca_rtail    =2 #aep at which zero value is assumeed. 'none' uses lowest aep in flood set
    
    
    #Floodo controllers
    gpwr_aep    = 100 #default max aep where gridpower_f = TRUE (when the power shuts off)
    
    dbg_fld_cnt = 0
    
    #area exposure
    area_egrd00 = None
    area_egrd01 = None
    area_egrd02 = None
    


    #Dfunc controllers

    place_codes = None
    dmg_types = None
        
    flood_tbl_nm = None #name of the flood table to use  
    
    
    #timeline deltas
    'just keeping this on the fdmg for simplicitly.. no need for flood level heterogenieyt'
    wsl_delta = 0.0
    fprob_mult = 1.0 #needs to be a float for type matching
    
    
    
    dmg_rat_f = False
    
    #Fdmg.House pars
    joist_space = 0.3
    G_anchor_ht = 0.6
    bsmt_egrd_code = 'plpm'
    damp_func_code = 'seep'
    bsmt_opn_ht_code = '*min(2.0)'
    
    hse_skip_depth = -4 #depth to skip house damage calc
    
    fhr_nm = ''
    
    cont_val_scale = .25
    
    write_fdmg_sum = True
    
    dfeat_xclud_price = 0.0

    #===========================================================================
    # calculation parameters
    #===========================================================================
    res_fancy = None
    gpwr_f              = True #placeholder for __init__ calcs

    fld_aep_l = None
    
    dmg_dx_base = None #results frame for writing
    
    plotr_d = None #dictionary of EAD plot workers
    
    dfeats_d = dict() #{tag:dfeats}. see raise_all_dfeats()
    
    fld_pwr_cnt = 0
    seq = 0
    
    
    #damage results/stats
    dmgs_df = None
    dmgs_df_wtail = None #damage summaries with damages for the tail logic included
    ead_tot = 0
    dmg_tot = 0
    
    #===========================================================================
    # calculation data holders
    #===========================================================================
    dmg_dx      = None #container for full run results
    
    bdry_cnt = 0
    bwet_cnt = 0
    bdamp_cnt = 0
    
    def __init__(self,*vars, **kwargs):

        logger = mod_logger.getChild('Fdmg')
        
        #=======================================================================
        # initilize cascade
        #=======================================================================
        super(Fdmg, self).__init__(*vars, **kwargs) #initilzie teh baseclass
        
        #=======================================================================
        # object updates
        #=======================================================================
        self.reset_d.update({'ead_tot':0, 'dmgs_df':None, 'dmg_dx':None,\
                             'wsl_delta':0}) #update the rest attributes
        
        #=======================================================================
        # pre checks
        #=======================================================================
        self.check_pars() #check the data loaded on your tab
        
        if not self.session._write_data:
            self.write_fdmg_sum = False
        
        #=======================================================================
        #setup functions
        #=======================================================================
        #par cleaners/ special loaders
        logger.debug("load_hse_geo() \n")
        self.load_hse_geo()
        
        logger.info('load and clean dfunc data \n')
        self.load_pars_dfunc(self.session.pars_df_d['dfunc']) #load the data functions to damage type table   
           
        logger.debug('\n')
        self.setup_dmg_dx_cols()
        
        logger.debug('load_submodels() \n')
        self.load_submodels()
        logger.debug('init_dyno() \n')
        self.init_dyno()
        
        #outputting setup
        if self.write_fdmg_sum_fly:
            self.fly_res_fpath = os.path.join(self.session.outpath, '%s fdmg_res_fly.csv'%self.session.tag)
        
        if self.db_f:
            if not self.model.__repr__() == self.__repr__():
                raise IOError
            

        logger.info('Fdmg model initialized as \'%s\' \n'%(self.name))
        
    
    def check_pars(self): #check your data pars
        df_raw = self.session.pars_df_d['datos']
        
        #=======================================================================
        # check mandatory data objects
        #=======================================================================
        if not 'binv' in df_raw['name'].tolist():
            raise IOError
        
        #=======================================================================
        # check optional data objects
        #=======================================================================
        fdmg_tab_nl = ['rfda_curve', 'binv','dfeat_tbl', 'fhr_tbl']
        boolidx = df_raw['name'].isin(fdmg_tab_nl)
        
        if not np.all(boolidx):
            raise IOError #passed some unexpected data names
        
        return
    

        
    def load_submodels(self):
        logger = self.logger.getChild('load_submodels')
        self.state = 'load'
        
        #=======================================================================
        # data objects
        #=======================================================================
        'this is the main loader that builds all teh children as specified on the data tab'
        logger.info('loading dat objects from \'fdmg\' tab')
        logger.debug('\n \n')
        
        #build datos from teh data tab
        'todo: hard code these class types (rather than reading from teh control file)'
        self.fdmgo_d = self.raise_children_df(self.session.pars_df_d['datos'], #df to raise on
                                              kid_class = None) #should raise according to df entry
        

        
        self.session.prof(state='load.fdmg.datos')
        'WARNING: fdmgo_d is not set until after ALL the children on this tab are raised'   
        #attach special children
        self.binv           = self.fdmgo_d['binv']
        
        """NO! this wont hold resetting updates
        self.binv_df        = self.binv.childmeta_df"""
        
        
        #=======================================================================
        # flood tables
        #=======================================================================
        self.ftblos_d = self.raise_children_df(self.session.pars_df_d['flood_tbls'], #df to raise on
                                              kid_class = datos.Flood_tbl) #should raise according to df entry
        
        
        'initial call which only udpates the binv_df'
        self.set_area_prot_lvl()
        
        if 'fhr_tbl' in self.fdmgo_d.keys():
            self.set_fhr()
        

        #=======================================================================
        # dfeats
        #======================================================================
        if self.session.load_dfeats_first_f & self.session.wdfeats_f:
            logger.debug('raise_all_dfeats() \n')
            self.dfeats_d         = self.fdmgo_d['dfeat_tbl'].raise_all_dfeats()
        
        #=======================================================================
        # raise houses
        #=======================================================================
        logger.info('raising houses')
        logger.debug('\n')
        
        self.binv.raise_houses()
        self.session.prof(state='load.fdmg.houses')
        'calling this here so all of the other datos are raised'
        #self.rfda_curve     = self.fdmgo_d['rfda_curve']
        
        """No! we need to get this in before the binv.reset_d['childmeta_df'] is set
        self.set_area_prot_lvl() #apply the area protectino from teh named flood table"""

        
        logger.info('loading floods')
        logger.debug('\n \n')
        self.load_floods()
        self.session.prof(state='load.fdmg.floods')
        
        
        
        logger.debug("finished with %i kids\n"%len(self.kids_d))
        
        
        return
    
    def setup_dmg_dx_cols(self): #get teh columns to use for fdmg results
        """
        This is setup to generate a unique set of ordered column names with this logic
            take the damage types
            add mandatory fields
            add user provided fields
        """
        logger = self.logger.getChild('setup_dmg_dx_cols')
        
        #=======================================================================
        #build the basic list of column headers
        #=======================================================================
        #damage types at the head
        col_os = OrderedSet(self.dmg_types) #put 
        
        #basic add ons
        _ = col_os.update(['total', 'hse_depth', 'wsl', 'bsmt_egrd', 'anchor_el'])
        

        #=======================================================================
        # special logic
        #=======================================================================
        if self.dmg_rat_f:
            for dmg_type in self.dmg_types:
                _ = col_os.add('%s_rat'%dmg_type)
                
                
        if not self.wsl_delta==0:
            col_os.add('wsl_raw')
            """This doesnt handle runs where we start with a delta of zero and then add some later
            for these, you need to expplicitly call 'wsl_raw' in the dmg_xtra_cols_fat"""
            
        #ground water damage
        if 'dmg_gw' in self.session.outpars_d['Flood']:
            col_os.add('gw_f')
            
        #add the dem if necessary
        if 'gw_f' in col_os:
            col_os.add('dem_el')
                
        
        #=======================================================================
        # set pars based on user provided 
        #=======================================================================
        #s = self.session.outpars_d[self.__class__.__name__]
        
        #extra columns for damage resulst frame
        if self.db_f or self.session.write_fdmg_fancy:
            
            logger.debug('including extra columns in outputs')  
            #clewan the extra cols
            'todo: move this to a helper'
            if hasattr(self.session, 'xtra_cols'):

                try:
                    dc_l = eval(self.session.xtra_cols) #convert to a list
                except:
                    logger.error('failed to convert \'xtra_cols\' to a list. check formatting')
                    raise IOError
            else:
                dc_l = ['wsl_raw', 'gis_area', 'hse_type', 'B_f_height', 'BS_ints','gw_f']
                
            if not isinstance(dc_l, list): raise IOError
            
            col_os.update(dc_l) #add these  

        self.dmg_df_cols = col_os

        logger.debug('set dmg_df_cols as: %s'%self.dmg_df_cols)
        
        return
                  
    def load_pars_dfunc(self, df_raw=None): #build a df from the dfunc tab
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('load_pars_dfunc')
        
        dfunc_ecols = ['place_code','dmg_code','dfunc_type','anchor_ht_code']
        
        if df_raw is None: 
            df_raw = self.session.pars_df_d['dfunc']
        
        logger.debug('from df %s: \n %s'%(str(df_raw.shape), df_raw))
        #=======================================================================
        # clean
        #=======================================================================
        df1 = df_raw.dropna(axis='columns', how='all')
        df2 = df1.dropna(axis='index', how='all') #drop rows with all na
        
        #column check
        if not hp.pd.header_check(df2, dfunc_ecols, logger=logger):
            raise IOError
        #=======================================================================
        # custom columns
        #=======================================================================
        df3 = df2.copy(deep=True)
        df3['dmg_type'] = df3['place_code'] + df3['dmg_code']
        df3['name'] = df3['dmg_type']
        
        #=======================================================================
        # data loading
        #=======================================================================
        if 'tailpath' in df3.columns:
            boolidx = ~pd.isnull(df3['tailpath']) #get dfuncs with data requests
            
            self.load_raw_dfunc(df3[boolidx])
            
            df3 = df3.drop(['headpath', 'tailpath'], axis = 1, errors='ignore') #drop these columns
        
        #=======================================================================
        # garage checking
        #=======================================================================
        boolidx = np.logical_and(df3['place_code'] == 'G', df3['dfunc_type'] == 'rfda')
        if np.any(boolidx):
            logger.error('got dfunc_type = rfda for a garage curve (no such thing)')
            raise IOError

        
        #=======================================================================
        # get special lists
        #=======================================================================
        #dmg_types
        self.dmg_types = df3['dmg_type'].tolist()
        
        #damage codes
        boolidx = df3['place_code'].str.contains('total')
        self.dmg_codes = df3.loc[~boolidx, 'dmg_code'].unique().tolist()
        
        #place_codes
        place_codes = df3['place_code'].unique().tolist()
        if 'total' in place_codes: place_codes.remove('total')
        self.place_codes = place_codes
                        

        self.session.pars_df_d['dfunc'] = df3
                
        logger.debug('dfunc_df with %s'%str(df3.shape))
        
        #=======================================================================
        # get slice for houses
        #=======================================================================
        #identify all the entries except total
        boolidx = df3['place_code'] != 'total'
         
        self.house_childmeta_df = df3[boolidx] #get this trim
        
        """
        hp.pd.v(df3)
        """
        
    def load_hse_geo(self): #special loader for hse_geo dxcol (from tab hse_geo)
        logger = self.logger.getChild('load_hse_geo')
        
        #=======================================================================
        # load and clean the pars
        #=======================================================================
        df_raw = hp.pd.load_xls_df(self.session.parspath, 
                               sheetname = 'hse_geo', header = [0,1], logger = logger)
        
        df = df_raw.dropna(how='all', axis = 'index')
        

        self.session.pars_df_d['hse_geo'] = df
        
        #=======================================================================
        # build a blank starter for each house to fill
        #=======================================================================
        
        omdex = df.columns #get the original mdex

        'probably a cleaner way of doing this'
        lvl0_values = omdex.get_level_values(0).unique().tolist()
        lvl1_values = omdex.get_level_values(1).unique().tolist()
        lvl1_values.append('t')
        
        newcols = pd.MultiIndex.from_product([lvl0_values, lvl1_values], 
                                             names=['place_code','finish_code'])
        
        geo_dxcol = pd.DataFrame(index = df.index, columns = newcols) #make the frame
        
        self.geo_dxcol_blank = geo_dxcol
                
        if self.db_f:
            if np.any(pd.isnull(df)):
                raise IOError
            
            l = geo_dxcol.index.tolist()
            
            if not l == [u'area', u'height', u'per', u'inta']:
                raise IOError
            

        
        return
        
    def load_raw_dfunc(self, meta_df_raw): #load raw data for dfuncs
        logger = self.logger.getChild('load_raw_dfunc')
        
        logger.debug('with df \'%s\''%(str(meta_df_raw.shape)))
        
        d = dict() #empty container
        
        meta_df = meta_df_raw.copy()
        
        #=======================================================================
        # loop through each row and load the data
        #=======================================================================
        for indx, row in meta_df.iterrows():
            
            inpath = os.path.join(row['headpath'], row['tailpath'])
            
            df = hp.pd.load_smart_df(inpath,
                                     index_col =None, 
                                     logger = logger)
            
            d[row['name']] = df.dropna(how = 'all', axis = 'index') #store this into the dictionaryu
            
        logger.info('finished loading raw dcurve data on %i dcurves: %s'%(len(d), d.keys()))
        
        self.dfunc_raw_d = d
        
        return

    def load_floods(self):
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('load_floods')
        logger.debug('setting floods df \n')
        self.set_floods_df()
        
        df = self.floods_df
        
        logger.debug('raising floods \n')
        d = self.raise_children_df(df,   #build flood children
                                               kid_class = Flood,
                                               dup_sibs_f= True,
                                               container = OrderedDict) #pass attributes from one tot eh next
        
        #=======================================================================
        # ordered by aep
        #=======================================================================
        fld_aep_od = OrderedDict()
        
        for childname, childo in d.iteritems():
            if hasattr(childo, 'ari'):  
                fld_aep_od[childo.ari] = childo
            else: raise IOError
            

        
        logger.info('raised and bundled %i floods by aep'%len(fld_aep_od))
        
        self.fld_aep_od = fld_aep_od
        
        return 
    
    def set_floods_df(self): #build the flood meta data
        
        logger = self.logger.getChild('set_floods_df')
            
        df_raw = self.session.pars_df_d['floods']
        
        df1 = df_raw.sort_values('ari').reset_index(drop=True)
        df1['ari'] = df1['ari'].astype(np.int)
        
        
        
        
        #=======================================================================
        # slice for debug set
        #=======================================================================
        if self.db_f & (not self.dbg_fld_cnt == 'all'):
            
            #check that we even have enough to do the slicing
            if len(df1) < 2:
                logger.error('too few floods for debug slicing. pass dbg_fld_cnt == all')
                raise IOError
            
            df2 = pd.DataFrame(columns = df1.columns) #make blank starter frame
            

            dbg_fld_cnt = int(self.dbg_fld_cnt)
            
            logger.info('db_f=TRUE. selecting %i (of %i) floods'%(dbg_fld_cnt, len(df1)))
            
            #===================================================================
            # try to pull out and add the 100yr
            #===================================================================
            try:
                boolidx = df1.loc[:,'ari'] == self.fld_aep_spcl
                if not boolidx.sum() == 1:
                    logger.debug('failed to locate 1 flood')
                    raise IOError
                
                df2 = df2.append(df1[boolidx])  #add this row to the end
                df1 = df1[~boolidx] #slice out this row
                
                dbg_fld_cnt = max(0, dbg_fld_cnt - 1) #reduce the loop count by 1
                dbg_fld_cnt = min(dbg_fld_cnt, len(df1)) #double check in case we are given a very short set
                
                logger.debug('added the %s year flood to the list with dbg_fld_cnt %i'%(self.fld_aep_spcl, dbg_fld_cnt))
                
            except:
                logger.debug('failed to extract the special %i flood'%self.fld_aep_spcl)
                df2 = df1.copy()
            
            
            #===================================================================
            # build list of extreme (low/high) floods
            #===================================================================
            evn_cnt = 0
            odd_cnt = 0
            

            for cnt in range(0, dbg_fld_cnt, 1): 
                
                if cnt % 2 == 0: #evens.  pull from front
                    idxr = evn_cnt
                    evn_cnt += 1
                    
                else: #odds. pull from end
                    idxr = len(df1) - odd_cnt - 1
                    odd_cnt += 1
                    
                logger.debug('pulling flood with indexer %i'%(idxr))

                ser = df1.iloc[idxr, :] #make thsi slice

                    
                df2 = df2.append(ser) #append this to the end
                
            #clean up
            df = df2.drop_duplicates().sort_values('ari').reset_index(drop=True)
            
            logger.debug('built extremes flood df with %i aeps: %s'%(len(df), df.loc[:,'ari'].values.tolist()))
            
            if not len(df) == int(self.dbg_fld_cnt): 
                raise IOError
                    
        else:
            df = df1.copy()
                    
        if not len(df) > 0: raise IOError
        
        self.floods_df = df
        
        return
    
    def set_area_prot_lvl(self): #assign the area_prot_lvl to the binv based on your tab
        #logger = self.logger.getChild('set_area_prot_lvl')
        """
        TODO: Consider moving this onto the binv and making the binv dynamic...
        
        Calls:
        handles for flood_tbl_nm
        """
        logger = self.logger.getChild('set_area_prot_lvl')
        logger.debug('assigning  \'area_prot_lvl\' for \'%s\''%self.flood_tbl_nm)
        
        #=======================================================================
        # get data
        #=======================================================================
        ftbl_o = self.ftblos_d[self.flood_tbl_nm] #get the activated flood table object
        ftbl_o.apply_on_binv('aprot_df', 'area_prot_lvl')
        """
        hp.pd.v(binv_df)
        type(df.iloc[:, 0])
        """
        
        return True
    
    def set_fhr(self): #assign the fhz bfe and zone from the fhr_tbl data
        logger = self.logger.getChild('set_fhr')
        logger.debug('assigning for \'fhz\' and \'bfe\'')
        
        
        #get the data for this fhr set
        fhr_tbl_o = self.fdmgo_d['fhr_tbl']
        try:
            df = fhr_tbl_o.d[self.fhr_nm]
        except:
            if not self.fhr_nm in fhr_tbl_o.d.keys():
                logger.error('could not find selected fhr_nm \'%s\' in the loaded rule sets: \n %s'
                             %(self.fhr_nm, fhr_tbl_o.d.keys()))
                raise IOError
        
        
        #=======================================================================
        # loop through each series and apply
        #=======================================================================
        """
        not the most generic way of handling this... 
        
        todo:
        add generic method to the binv 
            can take ser or df
            
            updates the childmeta_df if before init
            updates the children if after init
        """
        for hse_attn in ['fhz', 'bfe']:
            ser = df[hse_attn]

        
            if not self.session.state == 'init':
                #=======================================================================
                # tell teh binv to update its houses
                #=======================================================================
                self.binv.set_all_hse_atts(hse_attn, ser = ser)
                
            else:
                logger.debug('set column \'%s\' onto the binv_df'%hse_attn)
                self.binv.childmeta_df.loc[:,hse_attn] = ser #set this column in teh binvdf
        
        """I dont like this
        fhr_tbl_o.apply_on_binv('fhz_df', 'fhz', coln = self.fhr_nm)
        fhr_tbl_o.apply_on_binv('bfe_df', 'bfe', coln = self.fhr_nm)"""
        
        return True
        
    def get_all_aeps_classic(self):   #get the list of flood aeps from the classic flood table format
        'kept this special syntax reader separate in case we want to change th eformat of the flood tables'
        
        flood_pars_df = self.session.pars_df_d['floods'] #load the data from the flood table
        
        fld_aep_l = flood_pars_df.loc[:, 'ari'].values #drop the 2 values and convert to a list 
        
        return fld_aep_l
        
    def run(self, **kwargs): #placeholder for simulation runs
        logger = self.logger.getChild('run')
        logger.debug('on run_cnt %i'%self.run_cnt)
        self.run_cnt += 1
        self.state='run'
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not isinstance(self.outpath, basestring):
                raise IOError
            
        

        logger.info('\n fdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmgfdmg')
        logger.info('for run_cnt %i'%self.run_cnt)
        
        self.calc_fld_set(**kwargs)
        
        
        return
        
    def setup_res_dxcol(self, #setup the results frame
                        fld_aep_l = None, 
                        #dmg_type_list = 'all', 
                        bid_l = None): 
        
        #=======================================================================
        # defaults
        #=======================================================================
        if bid_l == None:           bid_l = self.binv.bid_l       
        if fld_aep_l is None:       fld_aep_l = self.fld_aep_od.keys() #just get all teh keys from the dictionary
        #if dmg_type_list=='all':    dmg_type_list = self.dmg_types 
        
        
        #=======================================================================
        # setup the dxind for writing
        #=======================================================================        
        lvl0_values = fld_aep_l 

        lvl1_values = self.dmg_df_cols #include extra reporting columns
        
        #fold these into a mdex (each flood_aep has all dmg_types)        
        columns = pd.MultiIndex.from_product([lvl0_values, lvl1_values], 
                                names=['flood_aep','hse_atts'])
        
        dmg_dx = pd.DataFrame(index = bid_l, columns = columns).sort_index() #make the frame
        
        self.dmg_dx_base = dmg_dx.copy()
        
        if self.db_f:
            logger = self.logger.getChild('setup_res_dxcol')
            
            if not self.beg_hist_df == False:
                fld_aep_l.sort()
                columns = pd.MultiIndex.from_product([fld_aep_l, ['bsmt_egrd', 'cond']], 
                        names=['flood_aep','bsmt_egrd'])
                        
                
                self.beg_hist_df = pd.DataFrame(index=bid_l, columns = columns)
                logger.info('recording bsmt_egrd history with %s'%str(self.beg_hist_df.shape))
            else:
                self.beg_hist_df = None
        
        """
        dmg_dx.columns
        """
        
        return 
      
    def calc_fld_set(self,  #calc flood damage for the flood set
                    fld_aep_l = None, #list of flood aeps to calcluate
                    #dmg_type_list = 'all',  #list of damage types to calculate
                    bid_l = None, #list of building names ot calculate
                    wsl_delta = None, #delta value to add to all wsl 
                    wtf = None, #optinonal flag to control writing of dmg_dx (otherwise session.write_fdmg_set_dx is used) 
                    **run_fld): #kwargs to send to run_fld 
        
        'we could separate the object creation and the damage calculation'
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        fld_aep_l:    list of floods to calc
            this can be a custom list built by the user
            extracted from the flood table (see session.get_ftbl_aeps)
            loaded from the legacy rfda pars (session.rfda_pars.fld_aep_l)\
            
        bid_l: list of ids (matching the mind varaible set under Fdmg)
        
        #=======================================================================
        # OUTPUTS
        #=======================================================================
        dmg_dx: dxcol of flood damage across all dmg_types and floods
            mdex
                lvl0:    flood aep 
                lvl1:    dmg_type + extra cols
                    I wanted to have this flexible, so the dfunc could pass up extra headers
                    couldnt get it to work. instead used a global list and  acheck
                    new headers must be added to the gloabl list and Dfunc.
                
                
            index
                bldg_id
                
        #=======================================================================
        # TODO:
        #=======================================================================
        setup to calc across binvs as well
        """
        #=======================================================================
        # defaults
        #=======================================================================
        start = time.time()
        logger = self.logger.getChild('calc_fld_set')
        
        
        if wtf is None:       wtf = self.session.write_fdmg_set_dx
        if wsl_delta is None: wsl_delta=  self.wsl_delta
        

        #=======================================================================
        # setup and load the results frame
        #=======================================================================
        #check to see that all of these conditions pass
        if not np.all([bid_l is None, fld_aep_l is None]):
            logger.debug('non default run. rebuild the dmg_dx_base')
            #non default run. rebuild the frame
            self.setup_res_dxcol(   fld_aep_l = fld_aep_l, 
                                    #dmg_type_list = dmg_type_list,
                                    bid_l = bid_l)

        elif self.dmg_dx_base is None:  #probably the first run
            if not self.run_cnt == 1: raise IOError
            logger.debug('self.dmg_dx_base is None. rebuilding')
            self.setup_res_dxcol(fld_aep_l = fld_aep_l, 
                                    #dmg_type_list = dmg_type_list,
                                    bid_l = bid_l) #set it up with the defaults
            
        dmg_dx = self.dmg_dx_base.copy() #just start witha  copy of the base
            
        
        #=======================================================================
        # finish defaults
        #=======================================================================
        'these are all mostly for reporting'
               
        if fld_aep_l is None:       fld_aep_l = self.fld_aep_od.keys() #just get all teh keys from the dictionary
        """ leaving these as empty kwargs and letting floods handle
        if bid_l == None:           bid_l = binv_dato.bid_l
        if dmg_type_list=='all':    dmg_type_list = self.dmg_types """
        
        """
        lvl0_values = dmg_dx.columns.get_level_values(0).unique().tolist()
        lvl1_values = dmg_dx.columns.get_level_values(1).unique().tolist()"""
        
        logger.info('calc flood damage (%i) floods w/ wsl_delta = %.2f'%(len(fld_aep_l), wsl_delta))
        logger.debug('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \n')
       
        #=======================================================================
        # loop and calc eacch flood
        #=======================================================================
        fcnt = 0
        first = True
        for flood_aep in fld_aep_l: #lopo through and build each flood
            #self.session.prof(state='%s.fdmg.calc_fld_set.%i'%(self.get_id(), fcnt)) #memory profiling
            
            self.state = flood_aep 
            'useful for keeping track of what the model is doing'
            #get teh flood
            flood_dato = self.fld_aep_od[flood_aep] #pull thsi from the dictionary
            logger.debug('getting dmg_df for %s'%flood_dato.name)
            
            #===================================================================
            # run sequence
            #===================================================================
            #get damage for these depths            
            dmg_df = flood_dato.run_fld(**run_fld)  #add the damage df to this slice
            
            if dmg_df is None: continue #skip this one
            
            #===================================================================
            # wrap up
            #===================================================================
           
            dmg_dx[flood_aep] = dmg_df  #store into the frame
            
            fcnt += 1
            
            logger.debug('for flood_aep \'%s\' on fcnt %i got dmg_df %s \n'%(flood_aep, fcnt, str(dmg_df.shape)))
            
            #===================================================================
            # checking
            #===================================================================
            if self.db_f:
                #check that the floods are increasing
                if first:
                    first = False
                    last_aep = None
                else:
                    if not flood_aep > last_aep:
                        raise IOError
                last_aep = flood_aep

        #=======================================================================
        # wrap up
        #=======================================================================
        self.state = 'na' 
        
        if wtf:
            filetail = '%s %s %s %s res_fld'%(self.session.tag, self.simu_o.name, self.tstep_o.name, self.name)
            filepath = os.path.join(self.outpath, filetail)
            hp.pd.write_to_file(filepath, dmg_dx, overwrite=True, index=True) #send for writing
            
        self.dmg_dx = dmg_dx
        

        
        stop = time.time()
        
        logger.info('in %.4f secs calcd damage on %i of %i floods'%(stop - start, fcnt, len(fld_aep_l)))
        
        
        return 
    
    def get_results(self): #called by Timestep.run_dt()
        
        self.state='wrap'
        logger = self.logger.getChild('get_results')
        
        #=======================================================================
        # optionals
        #=======================================================================
        s = self.session.outpars_d[self.__class__.__name__]
        
        if (self.session.write_fdmg_fancy) or (self.session.write_fdmg_sum):
            logger.debug("calc_summaries \n")
            dmgs_df = self.calc_summaries()
            self.dmgs_df = dmgs_df.copy()
            
        else: dmgs_df = None
            
        if ('ead_tot' in s) or ('dmg_df' in s):
            logger.debug('\n')
            self.calc_annulized(dmgs_df = dmgs_df, plot_f = False)
            'this will also run calc_sumamries if it hasnt happened yet'
            
        if 'dmg_tot' in s:
            #get a cross section of the 'total' column across all flood_aeps and sum for all entries
            self.dmg_tot = self.dmg_dx.xs('total', axis=1, level=1).sum().sum()
            

        if ('bwet_cnt' in s) or ('bdamp_cnt' in s) or ('bdry_cnt' in s):
            logger.debug('get_fld_begrd_cnt')
            self.get_fld_begrd_cnt()
        
        if 'fld_pwr_cnt' in s:
            logger.debug('calc_fld_pwr_cnt \n')
            cnt = 0
            for aep, obj in self.fld_aep_od.iteritems():
                if obj.gpwr_f: cnt +=1
            
            self.fld_pwr_cnt = cnt   
            
        self.binv.calc_binv_stats()
        
        if self.session.write_fdmg_fancy:
            self.write_res_fancy()
        
        if self.write_fdmg_sum_fly: #write the results after each run
            self.write_dmg_fly()
        
        if self.db_f: 
            self.check_dmg_dx()
                        
        logger.debug('finished \n')
        
    def calc_summaries(self, #annualize the damages
                       fsts_l = ['gpwr_f', 'dmg_sw', 'dmg_gw'], #list of additional flood attributes to report in teh summary
                       dmg_dx=None, 
                       plot=False, #flag to execute plot_dmgs() at the end. better to do this explicitly with an outputr 
                       wtf=None): 
        """
        basically dropping dimensions on the outputs and adding annuzlied damages
        #=======================================================================
        # OUTPUTS
        #=======================================================================
        DROP BINV DIMENSIOn
        dmgs_df:    df with 
            columns: raw damage types, and annualized damage types
            index: each flood
            entries: total damage for binv
            
        DROP FLOODS DIMENSIOn
        aad_sum_ser
        
        DROP ALL DIMENSIONS
        ead_tot
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('calc_summaries')
        if dmg_dx is None:  dmg_dx = self.dmg_dx.copy()
        if plot is None:    plot = self.session._write_figs
        if wtf is None:     wtf = self.write_fdmg_sum
        
        
        #=======================================================================
        # #setup frame
        #=======================================================================        
        #get the columns
        dmg_types = self.dmg_types + ['total']
        
        #=======================================================================
        # #build the annualized damage type names
        #=======================================================================
        admg_types = []
        for entry in dmg_types: admg_types.append(entry+'_a')
            
        cols = dmg_types + ['prob', 'prob_raw'] + admg_types + fsts_l
        
        self.dmg_df_cols
        """
        hp.pd.v(dmg_dx)
        """
        
        dmgs_df = pd.DataFrame(columns = cols)
        dmgs_df['ari'] = dmg_dx.columns.get_level_values(0).unique()
        dmgs_df = dmgs_df.sort_values('ari').reset_index(drop=True)
        
        #=======================================================================
        # loop through and fill out the data
        #=======================================================================
        for index, row in dmgs_df.iterrows(): #loop through an dfill out
            
            dmg_df = dmg_dx[row['ari']] #get the fdmg for this aep
            
            #sum all the damage types
            for dmg_type in dmg_types: 
                row[dmg_type] = dmg_df[dmg_type].sum() #sum them all up
                       
            #calc the probability
            row['prob_raw'] = 1/float(row['ari']) #inverse of aep
            row['prob'] = row['prob_raw'] * self.fprob_mult #apply the multiplier
            
            #calculate the annualized damages
            for admg_type in admg_types: 
                dmg_type  = admg_type[:-2] #drop the a
                row[admg_type] = row[dmg_type] * row['prob']
                
            #===================================================================
            # get stats from the floodo
            #===================================================================
            floodo = self.fld_aep_od[row['ari']]
            
            for attn in fsts_l:
                row[attn] = getattr(floodo, attn)
                
            #===================================================================
            # #add this row backinto the frame
            #===================================================================
            dmgs_df.loc[index,:] = row
            
        #=======================================================================
        # get series totals
        #=======================================================================
        
        dmgs_df = dmgs_df.sort_values('prob').reset_index(drop='true')
        #=======================================================================
        # closeout
        #=======================================================================
        logger.debug('annualized %i damage types for %i floods'%(len(dmg_type), len(dmgs_df)))
        
        if wtf:
            filetail = '%s dmg_sumry'%(self.session.state)
            filepath = os.path.join(self.outpath, filetail)
            hp.pd.write_to_file(filepath, dmgs_df, overwrite=True, index=False) #send for writing
            
        
        logger.debug('set data with %s and cols: %s'%(str(dmgs_df.shape), dmgs_df.columns.tolist()))
        
        if plot: 
            self.plot_dmgs(wtf=wtf)
        
        #=======================================================================
        # post check
        #=======================================================================
        if self.db_f:
                #check for sort logic
            if not dmgs_df.loc[:,'prob'].is_monotonic: 
                raise IOError
            
            if not dmgs_df['total'].iloc[::-1].is_monotonic: #flip the order 
                logger.warning('bigger floods arent causing more damage')
                'some of the flood tables seem bad...'
                #raise IOError
            
            #all probabilities should be larger than zero
            if not np.all(dmgs_df.loc[:,'prob'] > 0): 
                raise IOError 
            
        return dmgs_df
    
    def calc_annulized(self, dmgs_df = None,
                       ltail = None, rtail = None, plot_f=None,
                       dx = 0.001): #get teh area under the damage curve
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        ltail: left tail treatment code (low prob high damage)
            flat: extend the max damage to the zero probability event
            'none': don't extend the tail
            
        rtail: right trail treatment (high prob low damage)
            'none': don't extend
            '2year': extend to zero damage at the 2 year aep

        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('calc_annulized')
        if ltail is None: ltail = self.ca_ltail
        if rtail is None: rtail = self.ca_rtail
        'plotter ignores passed kwargs here'
        
        if plot_f is None: plot_f=  self.session._write_figs
        
        #=======================================================================
        # get data
        #=======================================================================
        if dmgs_df is None:
            dmgs_df = self.calc_summaries()
        #df_raw = self.data.loc[:,('total', 'prob', 'ari')].copy().reset_index(drop=True)
        'only slicing columns for testing'
        
        df = dmgs_df.copy().reset_index(drop=True)
        
        if len(df) == 1:
            logger.warning('only got one flood entry. skipping')
            self.ead_tot = 0
            self.dmgs_df_wtail = df
            return
            
        
        logger.debug("with ltail = \'%s\', rtail = \'%s\' and df %s"%(ltail, rtail, str(df.shape)))
        
        
        if self.db_f:
            if len(df) <2:
                logger.error('didnt get enough flood entries to calcluate EAD')
                raw_input('press enter to continue any way....')
                
        
        #=======================================================================
        # left tail treatment
        #=======================================================================
        if ltail == 'flat':
            #zero probability
            'assume 1000yr flood is the max damage'
            max_dmg = df['total'].max()*1.0001
            
            df.loc[-1, 'prob'] = 0
            df.loc[-1, 'ari'] = 999999
            df.loc[-1, 'total'] = max_dmg
            
            logger.debug('ltail == flat. duplicated danage %.2f at prob 0'%max_dmg)

        elif ltail == 'none':
            pass            
        else: raise IOError
        
        'todo: add option for value multiplier'
        
        #=======================================================================
        # right tail
        #=======================================================================
        if rtail == 'none':
            pass

            
        elif hp.basic.isnum(rtail):
            
            rtail_yr = float(rtail)
            rtail_p = 1.0 / rtail_yr
            
            max_p = df['prob'].max()
            
            #floor check
            if rtail_p < max_p: 
                logger.error('rtail_p (%.2f) < max_p (%.2f)'%(rtail_p, max_p))
                raise IOError
            
            #same
            elif rtail_p == max_p:
                logger.debug("rtail_p == min(xl. no changes made")

            else:
                logger.debug("adding zero damage for aep = %.1f"%rtail_yr)
                #zero damage
                'assume no damage occurs at the passed rtail_yr'

                loc = len(df)
                df.loc[loc, 'prob'] = rtail_p
                df.loc[loc, 'ari'] = 1.0/rtail_p
                df.loc[loc, 'total'] = 0
                
                """
                hp.pd.view_web_df(self.data)
                """
            
        else: raise IOError
        

        #=======================================================================
        # clean up
        #=======================================================================
        df = df.sort_index() #resort the index
        
        if self.db_f:
            'these should still hold'
            if not df.loc[:,'prob'].is_monotonic: 
                raise IOError
            
            """see above
            if not df['total'].iloc[::-1].is_monotonic: 
                raise IOError"""
            
        x, y = df['prob'].values.tolist(), df['total'].values.tolist()
            

        #=======================================================================
        # find area under curve
        #=======================================================================
        try:
            #ead_tot = scipy.integrate.simps(y, x, dx = dx, even = 'avg')
            'this was giving some weird results'
            ead_tot = scipy.integrate.trapz(y, x, dx = dx)
        except:
            logger.warning('scipy.integrate.trapz failed. setting ead_tot to zero')
            ead_tot = 0
            raise IOError
            
            
        logger.info('found ead_tot = %.2f $/yr from %i points with tail_codes: \'%s\' and \'%s\''
                    %(ead_tot, len(y), ltail, rtail))
        
        self.ead_tot = ead_tot
        #=======================================================================
        # checks
        #=======================================================================
        if self.db_f:
            if pd.isnull(ead_tot):
                raise IOError
            
            if not isinstance(ead_tot, float):
                raise IOError
            
            if ead_tot <=0:
                raise IOError   
        #=======================================================================
        # update data with tails
        #=======================================================================
        self.dmgs_df_wtail = df.sort_index().reset_index(drop=True)
        
        #=======================================================================
        # generate plot
        #=======================================================================
        if plot_f: 
            self.plot_dmgs(self, right_nm = None, xaxis = 'prob', logx = False)
            
        return
    
    def get_fld_begrd_cnt(self): #tabulate the bsmt_egrd counts from each flood
        logger = self.logger.getChild('get_fld_begrd_cnt')
        
        #=======================================================================
        # data setup
        #=======================================================================
        dmg_dx = self.dmg_dx.copy()
        
        #lvl1_values = dmg_dx.columns.get_level_values(0).unique().tolist()
        
        #get all teh basement egrade types
        df1 =  dmg_dx.loc[:,idx[:, 'bsmt_egrd']] #get a slice by level 2 values
        
        #get occurances by value
        d = hp.pd.sum_occurances(df1, logger=logger)
        
        #=======================================================================
        # loop and calc
        #=======================================================================
        logger.debug('looping through %i bsmt_egrds: %s'%(len(d), d.keys()))
        for bsmt_egrd, cnt in d.iteritems():
            attn = 'b'+bsmt_egrd +'_cnt'
            
            logger.debug('for \'%s\' got %i'%(attn, cnt))
            
            setattr(self, attn, cnt)
        
        logger.debug('finished \n')
        
    def check_dmg_dx(self): #check logical consistency of the damage results
        logger = self.logger.getChild('check_dmg_dx')
        
        #=======================================================================
        # data setup
        #=======================================================================
        dmg_dx = self.dmg_dx.copy()
        
        mdex = dmg_dx.columns
        
        
        aep_l = mdex.get_level_values(0).astype(int).unique().values.tolist()
        aep_l.sort()
        
        
        #=======================================================================
        # check that each flood increases in damage
        #=======================================================================
        total = None
        aep_last = None
        for aep in aep_l:
            #get this slice
            df = dmg_dx[aep]
            
            if total is None:
                boolcol = np.isin(df.columns, ['MS', 'MC', 'BS', 'BC', 'GS']) #identify damage columns
                total = df.loc[:,boolcol].sum().sum()
                
                
                
                if not aep == min(aep_l):
                    raise IOError
                
            else:
                
                newtot = df.loc[:,boolcol].sum().sum()
                if not newtot >= total: 
                    logger.warning('aep %s tot %.2f < aep %s %.2f'%(aep, newtot, aep_last, total))
                    #raise IOError
                #print 'new tot %.2f > oldtot %.2f'%(newtot, total)
                total = newtot
                
            aep_last = aep
            
            
        
        return
            

    def wrap_up(self):
        
        #=======================================================================
        # update asset containers
        #=======================================================================
        """
        #building inventory
        'should be flagged for updating during House.notify()'
        if self.binv.upd_kid_f: 
            self.binv.update()"""
            
        """dont think we need this here any more.. only on udev.
        keeping it just to be save"""
        
            
        self.last_tstep = copy.copy(self.time)
        self.state='close'

    def write_res_fancy(self,  #for saving results in xls per tab. called as a special outputr
                        dmg_dx=None, 
                        include_ins = False,
                        include_raw = False,
                        include_begh = False): 
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        include_ins: whether ot add inputs as tabs.
            ive left this separate from the 'copy_inputs' flag as it is not a true file copy of the inputs
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('write_res_fancy')
        if dmg_dx is None: dmg_dx = self.dmg_dx
        if dmg_dx is None: 
            logger.warning('got no dmg_dx. skipping')
            return 
        
        #=======================================================================
        # setup
        #=======================================================================
        od = OrderedDict()
        
        #=======================================================================
        # add the parameters
        #=======================================================================
        #get the blank frame
        df = pd.DataFrame(columns = ['par','value'] )
        df['par'] = list(self.try_inherit_anl)

        for indx, row in df.iterrows():
            df.iloc[indx, 1] = getattr(self, row['par']) #set this value
            
        od['pars'] = df
            
        
        #=======================================================================
        # try and add damage summary
        #=======================================================================
        if not self.dmgs_df is None:
            od['dmg summary'] = self.dmgs_df
        
        #=======================================================================
        # #get theh dmg_dx decomposed        
        #=======================================================================
        od.update(hp.pd.dxcol_to_df_set(dmg_dx, logger=self.logger))
               
        
        #=======================================================================
        # #add dmg_dx as a raw tab
        #=======================================================================
        if include_raw:
            od['raw_res'] = dmg_dx

        #=======================================================================
        # add inputs
        #=======================================================================
        if include_ins:
            for dataname, dato in self.kids_d.iteritems():
                if hasattr(dato, 'data') & hp.pd.isdf(dato.data):
                    od[dataname] = dato.data
                    
                    
        #=======================================================================
        # add debuggers
        #=======================================================================
        if include_begh:
            if not self.beg_hist_df is None:
                od['beg_hist'] = self.beg_hist_df
            

                   
        #=======================================================================
        # #write to excel
        #=======================================================================
        filetail = '%s %s %s %s fancy_res'%(self.session.tag, self.simu_o.name, self.tstep_o.name, self.name)

        filepath = os.path.join(self.outpath, filetail)
        hp.pd.write_dfset_excel(od, filepath, engine='xlsxwriter', logger=self.logger)

        return
    
    def write_dmg_fly(self): #write damage results after each run
        
        logger = self.logger.getChild('write_dmg_fly')
        
        dxcol = self.dmg_dx #results
        
        #=======================================================================
        # build the resuults summary series
        #=======================================================================
        
        #get all the flood aeps
        lvl0vals = dxcol.columns.get_level_values(0).unique().astype(int).tolist()
        
        #blank holder
        res_ser = pd.Series(index = lvl0vals)
        
        #loop and calc sums for each flood
        for aep in lvl0vals:
            res_ser[aep] = dxcol.loc[:,(aep,'total')].sum()
        
        #add extras
        if not self.ead_tot is None:
            res_ser['ead_tot'] = self.ead_tot
                
        
        res_ser['dt'] = self.tstep_o.year
        res_ser['sim'] = self.simu_o.ind
        
        
        lindex = '%s.%s'%(self.simu_o.name, self.tstep_o.name)
        
        
        hp.pd.write_fly_df(self.fly_res_fpath,res_ser,  lindex = lindex,
                   first = self.write_dmg_fly_first, tag = 'fdmg totals', 
                   db_f = self.db_f, logger=logger) #write results on the fly
    
        self.write_dmg_fly_first = False
        
        return

    def get_plot_kids(self): #raise kids for plotting the damage summaries
        logger = self.logger.getChild('get_plot_kids')
        #=======================================================================
        # get slice of aad_fmt_df matching the aad cols
        #=======================================================================
        aad_fmt_df = self.session.pars_df_d['dmg_sumry_plot'] #pull teh formater pars from the tab
      
        dmgs_df = self.dmgs_df
        self.data = dmgs_df
        
        boolidx = aad_fmt_df.loc[:,'name'].isin(dmgs_df.columns) #get just those formaters with data in the aad
        
        aad_fmt_df_slice = aad_fmt_df[boolidx] #get this slice3
        
        """
        hp.pd.view_web_df(self.data)
        hp.pd.view_web_df(df)
        hp.pd.view_web_df(aad_fmt_df_slice)
        aad_fmt_df_slice.columns
        """

        #=======================================================================
        # formatter kids setup
        #=======================================================================
        """need to run this every time so the data is updated
        TODO: allow some updating here so we dont have to reduibl deach time
        if self.plotter_kids_dict is None:"""
        self.plotr_d = self.raise_children_df(aad_fmt_df_slice, kid_class = hp.data.Data_o)
            
        logger.debug('finisehd \n')
                 
    def plot_dmgs(self, wtf=None, right_nm = None, xaxis = 'ari', logx = True,
                  ylims = None, #tuple of min/max values for the y-axis
                  ): #plot curve of aad
        """
        see tab 'aad_fmt' to control what is plotted and formatting
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('plot_dmgs')
        if wtf == None: wtf = self.session._write_figs
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if self.dmgs_df is None:
                raise IOError
            

        #=======================================================================
        # setup
        #=======================================================================
        if not ylims is None:
            try:
                ylims = eval(ylims)
            except:
                pass
            
        #get the plot workers
        if self.plotr_d is None: 
            self.get_plot_kids()
            
        kids_d = self.plotr_d
        
        title = '%s-%s-%s EAD-ARI plot on %i objs'%(self.session.tag, self.simu_o.name, self.name, len(self.binv.childmeta_df))
        logger.debug('with \'%s\''%title)
        
        if not self.tstep_o is None:
            title = title + ' for %s'%self.tstep_o.name
        
        #=======================================================================
        # update plotters
        #=======================================================================
        logger.debug('updating plotters with my data')

        #get data
        data_og = self.data.copy() #store this for later
        
        if self.dmgs_df_wtail is None:
            df = self.dmgs_df.copy()
        else:
            df = self.dmgs_df_wtail.copy()
        
        df = df.sort_values(xaxis, ascending=True)
  
        #reformat data
        df.set_index(xaxis, inplace = True)
        
        #re set
        self.data = df
        
        #tell kids to refresh their data from here
        for gid, obj in kids_d.iteritems(): obj.data = obj.loadr_vir()
             
        self.data = data_og #reset the data
        
        #=======================================================================
        # get annotation
        #=======================================================================
        val_str = '$' + "{:,.2f}".format(self.ead_tot/1e6)
        #val_str = "{:,.2f}".format(self.ead_tot)
        """
        txt = 'total aad: $%s \n tail kwargs: \'%s\' and \'%s\' \n'%(val_str, self.ca_ltail, self.ca_rtail) +\
                'binv.cnt = %i, floods.cnt = %i \n'%(self.binv.cnt, len(self.fld_aep_od))"""
         

        txt = 'total EAD = %s'%val_str        
            
                
        #=======================================================================
        #plot the workers
        #=======================================================================
        #twinx
        if not right_nm is None:
            logger.debug('twinning axis with name \'%s\''%right_nm)
            title = title + '_twin'
            # sort children into left/right buckets by name to plot on each axis
            right_pdb_d, left_pdb_d = self.sort_buckets(kids_d, right_nm)
            
            if self.db_f:
                if len (right_pdb_d) <1: raise IOError
            
            #=======================================================================
            # #send for plotting
            #=======================================================================
            'this plots both bundles by their data indexes'
            ax1, ax2 = self.plot_twinx(left_pdb_d, right_pdb_d, 
                                       logx=logx, xlab = xaxis, title=title, annot = txt,
                                       wtf=False)
            'cant figure out why teh annot is plotting twice'
            
            ax2.set_ylim(0, 1) #prob limits
            legon = False
        else:
            logger.debug('single axis')
            
            try:
                del kids_d['prob']
            except:
                pass
            
            pdb = self.get_pdb_dict(kids_d.values())
            
            ax1 = self.plot_bundles(pdb,
                                   logx=logx, xlab = 'ARI', ylab = 'damage ($ 10^6)', title=title, annot = txt,
                                   wtf=False)
            
            legon=True
        
        #hatch
        #=======================================================================
        # post formatting
        #=======================================================================
        #set axis limits
        if xaxis == 'ari': ax1.set_xlim(1, 1000) #aep limits
        elif xaxis == 'prob': ax1.set_xlim(0, .6) 
        
        if not ylims is None:
            ax1.set_ylim(ylims[0], ylims[1])
        

        #ax1.set_ylim(0, ax1.get_ylim()[1]) #$ limits
        
        
        #=======================================================================
        # format y axis labels
        #======================================================= ================
        old_tick_l = ax1.get_yticks() #get teh old labels
        
        # build the new ticks
        l = []
        
        for value in old_tick_l:
            new_v = '$' + "{:,.0f}".format(value/1e6)
            l.append(new_v)
             
        #apply the new labels
        ax1.set_yticklabels(l)
        
        """
        #add thousands comma
        ax1.get_yaxis().set_major_formatter(
            #matplotlib.ticker.FuncFormatter(lambda x, p: '$' + "{:,.2f}".format(x/1e6)))

            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))"""
        
        if xaxis == 'ari':
            ax1.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        

        if wtf: 
            fig = ax1.figure
            savepath_raw = os.path.join(self.outpath,title)
            flag = hp.plot.save_fig(self, fig, savepath_raw=savepath_raw, dpi = self.dpi, legon=legon)
            if not flag: raise IOError 
            

        #plt.close()
        return

class Flood( 
                hp.dyno.Dyno_wrap,
                hp.sim.Sim_o, 
                hp.oop.Parent,  #flood object worker
                hp.oop.Child): 
    
    #===========================================================================
    # program pars
    #===========================================================================
    
    gpwr_f          = False #grid power flag palceholder
    #===========================================================================
    # user defineid pars
    #===========================================================================
    ari             = None

    
    #loaded from flood table
    #area exposure grade. control for areas depth decision algorhithim based on the performance of macro structures (e.g. dykes).
    area_egrd00 = ''
    area_egrd01 = ''
    area_egrd02 = ''
    
    area_egrd00_code = None
    area_egrd01_code = None
    area_egrd02_code = None
    #===========================================================================
    # calculated pars
    #===========================================================================
    hdep_avg        = 0 #average house depth
    #damate properties
    total = 0
    BS = 0
    BC = 0
    MS = 0
    MC = 0
    dmg_gw = 0
    dmg_sw = 0
    
    dmg_df_blank =None
    
    wsl_avg = 0
    

    #===========================================================================
    # data containers
    #===========================================================================
    hdmg_cnt        = 0
    dmg_df = None
    dmg_res_df = None

    #bsmt_egrd counters. see get_begrd_cnt()
    bdry_cnt        = 0
    bwet_cnt        = 0
    bdamp_cnt       = 0


    def __init__(self, parent, *vars, **kwargs):
        logger = mod_logger.getChild('Flood')
        logger.debug('start _init_')
        #=======================================================================
        # #attach custom vars
        #=======================================================================
        self.inherit_parent_ans=set(['mind', 'dmg_types'])
        #=======================================================================
        # initilize cascade
        #=======================================================================
        super(Flood, self).__init__(parent, *vars, **kwargs) #initilzie teh baseclass 
        
        #=======================================================================
        # common setup
        #=======================================================================
        if self.sib_cnt == 0:
            #update the resets
            pass

        #=======================================================================
        # unique setup
        #=======================================================================
        """ handled by the outputr
        self.reset_d.update({'hdmg_cnt':0})"""
        self.ari = int(self.ari)
        self.dmg_res_df = pd.DataFrame() #set as an empty frame for output handling
        
        #=======================================================================
        # setup functions
        #=======================================================================
        self.set_gpwr_f()
        
        logger.debug('set_dmg_df_blank()')
        self.set_dmg_df_blank()
        
        logger.debug('get your water levels from the selected wsl table \n')
        self.set_wsl_frm_tbl()
        
        logger.debug('set_area_egrd()')
        self.set_area_egrd()
        
        logger.debug('get_info_from_binv()')
        df = self.get_info_from_binv() #initial run to set blank frame
        
        self.set_wsl_from_egrd(df)

        
        """ moved into set_wsl_frm_tbl()
        logger.debug('\n')
        self.setup_dmg_df()"""
        
        self.init_dyno()
        
        self.logger.debug('__init___ finished \n')
        
    def set_dmg_df_blank(self):
        
        logger = self.logger.getChild('set_dmg_df_blank')
        
        binv_df = self.model.binv.childmeta_df
        
        colns = OrderedSet(self.model.dmg_df_cols.tolist() + ['wsl', 'area_prot_lvl'])
        'wsl should be redundant'
        
        #get boolean
        self.binvboolcol = binv_df.columns.isin(colns) #store this for get_info_from_binv()
        
        #get teh blank frame
        self.dmg_df_blank = pd.DataFrame(columns = colns, index = binv_df.index) #get the blank frame
        'this still needs the wsl levels attached based on your area exposure grade'
        
        logger.debug('set dmg_df_blank with %s'%(str(self.dmg_df_blank.shape)))
        
        return

    def set_gpwr_f(self): #set your power flag
        
        if self.is_frozen('gpwr_f'): return True#shortcut for frozen
        
        logger = self.logger.getChild('set_gpwr_f')
        
        #=======================================================================
        # get based on aep
        #=======================================================================
        min_aep = int(self.model.gpwr_aep)
        
        if self.ari < min_aep:  gpwr_f = True
        else:                   gpwr_f = False
        
        logger.debug('for min_aep = %i, set gpwr_f = %s'%(min_aep, gpwr_f))
        
        #update handler
        self.handle_upd('gpwr_f', gpwr_f, proxy(self), call_func = 'set_gpwr_f')
        
        return True

    def set_wsl_frm_tbl(self, #build the raw wsl data from the passed flood table
                         flood_tbl_nm = None, #name of flood table to pull raw data from
                         #bid_l=None, 
                         ): 
        """
        here we get the raw values
        these are later modified by teh area_egrd with self.get_wsl_from_egrd()
        #=======================================================================
        # INPUTS
        #=======================================================================
        flood_tbl_df_raw:    raw df of the classic flood table
            columns:`    count, aep, aep, aep, aep....\
            real_columns:    bldg_id, CPID, depth, depth, depth, etc...
            index:    unique arbitrary
            
        wsl_ser: series of wsl for this flood on each bldg_id
        
        #=======================================================================
        # calls
        #=======================================================================
        dynp handles Fdmg.flood_tbl_nm
                    
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_wsl_frm_tbl')
        if flood_tbl_nm is None: flood_tbl_nm = self.model.flood_tbl_nm
        
        #=======================================================================
        # get data
        #=======================================================================
        #pull the raw flood tables
        ftbl_o = self.model.ftblos_d[flood_tbl_nm]
        wsl_d = ftbl_o.wsl_d
        
        df = pd.DataFrame(index = wsl_d.values()[0].index) #blank frame from teh first entry
        

        #=======================================================================
        # loop and apply for each flood type
        #=======================================================================
        for ftype, df1 in wsl_d.iteritems():
            
            #=======================================================================
            # data checks
            #=======================================================================
            if self.db_f:
                if not ftype in ['wet', 'dry', 'damp']: 
                    raise IOError
                df_raw =df1.copy()
                
                if not self.ari in df_raw.columns:
                    logger.error('the flood provided on the \'floods\' tab (\'%s\') does not have a match in the flood table: \n %s'%
                                 (self.ari, self.model.ftblos_d[flood_tbl_nm].filepath))
                    raise IOError
                
            #=======================================================================
            # slice for this flood
            #=======================================================================
            boolcol = df1.columns == self.ari #slice for this aep
                
            #get the series for this 
            wsl_ser = df1.loc[:, boolcol].iloc[:,0].astype(float)
    
            #wsl_ser = wsl_ser.rename(ftype) #rename with the aep
            
            'binv slicing moved to Flood_tbl.clean_data()'
            
            #=======================================================================
            # checks
            #=======================================================================
            if self.db_f:
                if len(wsl_ser) <1: 
                    raise IOError
                
                """ allowing
                #check for nuls
                if np.any(pd.isnull(wsl_ser2)):
                    raise IOError"""
                
                
            #=======================================================================
            # wrap up report and attach
            #======================================================================= 
            df[ftype] = wsl_ser
                       
            logger.debug('from \'%s\' for \'%s\' got wsl_ser %s for aep: %i'
                         %(flood_tbl_nm, ftype, str(wsl_ser.shape), self.ari))
            

        self.wsl_df = df #set this
        
        'notusing dy nps'
        if self.session.state == 'init':
            self.reset_d['wsl_df'] = df.copy()
        

        return True

    def set_area_egrd(self): #pull your area exposure grade from somewhere
        """
        #=======================================================================
        # calls
        #=======================================================================
        self.__init__()
        dynp handles: Fdmg.flood_tbl_nm (just in case we are pulling from there
        """
        #=======================================================================
        # dependency check
        #=======================================================================
        if not self.session.state=='init':
                
            dep_l =  [([self.model], ['set_area_prot_lvl()'])]
            
            if self.deps_is_dated(dep_l, method = 'reque', caller = 'set_area_egrd'):
                return False
            
        
        logger          = self.logger.getChild('set_area_egrd')
        
        #=======================================================================
        # steal egrd from elsewhere table if asked       
        #=======================================================================
        for cnt in range(0,3,1): #loop through each one
            attn = 'area_egrd%02d'%cnt
            
            area_egrd_code = getattr(self, attn + '_code')
            
            if area_egrd_code in ['dry', 'damp', 'wet']: 
                area_egrd = area_egrd_code

            #===================================================================
            # pull from teh flood table
            #===================================================================
            elif area_egrd_code == '*ftbl':
                ftbl_o = self.model.ftblos_d[self.model.flood_tbl_nm] #get the flood tabl object
                
                area_egrd = getattr(ftbl_o, attn) #get from teh table
                
            #===================================================================
            # pull from teh model
            #===================================================================
            elif area_egrd_code == '*model':
                area_egrd = getattr(self.model, attn) #get from teh table
                
            else:
                logger.error('for \'%s\' got unrecognized area_egrd_code: \'%s\''%(attn, area_egrd_code))
                raise IOError

            #===================================================================
            # set these
            #===================================================================
            self.handle_upd(attn, area_egrd, weakref.proxy(self), call_func = 'set_area_egrd')
            'this should triger generating a new wsl set to teh blank_dmg_df'

            logger.debug('set \'%s\' from \'%s\' as \'%s\''
                         %(attn, area_egrd_code,area_egrd))
            
            if self.db_f:
                if not area_egrd in ['dry', 'damp', 'wet']:
                    raise IOError
            
        return True
                
    def set_wsl_from_egrd(self, df = None):  #calculate the wsl based on teh area_egrd
        """
        This is a partial results retrival for non damage function results
        
        TODO: 
        consider checking for depednency on House.area_prot_lvl
        
        #=======================================================================
        # calls
        #=======================================================================
        self.__init__
        
        dynp handles Flood.area_egrd##
        
        
        """
        #=======================================================================
        # check dependencies and frozen
        #=========================================================== ============
        if not self.session.state=='init':
                
            dep_l =  [([self], ['set_area_egrd()', 'set_wsl_frm_tbl()'])]
            
            if self.deps_is_dated(dep_l, method = 'reque', caller = 'set_wsl_from_egrd'):
                return False
                
                
        #=======================================================================
        # defaults
        #=======================================================================
        logger          = self.logger.getChild('set_wsl_from_egrd')
        #if wsl_delta is None: wsl_delta = self.model.wsl_delta
        
        #=======================================================================
        # get data
        #=======================================================================
        if df is None: df = self.get_info_from_binv()
        'need to have updated area_prot_lvls'
        
        #=======================================================================
        # precheck
        #=======================================================================
        if self.db_f:
            if not isinstance(df, pd.DataFrame): raise IOError
            if not len(df) > 0: raise IOError

        #=======================================================================
        # add the wsl for each area_egrd
        #=======================================================================
        for prot_lvl in range(0,3,1): #loop through each one
            #get your  grade fro this prot_lvl
            attn = 'area_egrd%02d'%prot_lvl            
            area_egrd = getattr(self, attn)
            
            #identify the housese for this protection level
            boolidx = df.loc[:,'area_prot_lvl'] == prot_lvl
            
            if boolidx.sum() == 0: continue
            
            #give them the wsl corresponding to this grade
            df.loc[boolidx, 'wsl'] = self.wsl_df.loc[boolidx,area_egrd]
            
            #set a tag for the area_egrd
            if 'area_egrd' in df.columns:
                df.loc[boolidx, 'area_egrd'] = area_egrd
            
            logger.debug('for prot_lvl %i, set %i wsl from \'%s\''%(prot_lvl, boolidx.sum(), area_egrd))
            
        #=======================================================================
        # set this
        #=======================================================================
        self.dmg_df_blank = df
        
        #=======================================================================
        # post check
        #=======================================================================
        logger.debug('set dmg_df_blank with %s'%str(df.shape))
        
        if self.session.state=='init':
            self.reset_d['dmg_df_blank'] = df.copy()
        
            
        if self.db_f:
            if np.any(pd.isnull(df['wsl'])):
                logger.error('got some wsl nulls')
                raise IOError
            
        return True
    
        """
        hp.pd.v(df)
        hp.pd.v(self.dmg_df_blank)
        """
 
    def run_fld(self, **kwargs): #shortcut to collect all the functions for a simulation ru n
        
        self.run_cnt += 1
        
        dmg_df_blank = self.get_info_from_binv()
        
        dmg_df = self.get_dmg_set(dmg_df_blank, **kwargs)
        
        if self.db_f: self.check_dmg_df(dmg_df)
        
        'leaving this here for simplicity'
        self.calc_statres_flood(dmg_df)
        
        return dmg_df

    def get_info_from_binv(self):
    
        #=======================================================================
        # defaults
        #=======================================================================
        logger          = self.logger.getChild('get_info_from_binv')

        binv_df = self.model.binv.childmeta_df 
                   
        #pull static values
        binvboolcol = self.binvboolcol       
        df = self.dmg_df_blank.copy()
        'this should have wsl added to it from set_wsl_from_egrd()'
        
        if self.db_f:
            if not len(binvboolcol) == len(binv_df.columns):
                logger.warning('got length mismatch between binvboolcol (%i) and the binv_df columns (%i)'%
                             (len(binvboolcol), len(binv_df.columns)))
                'pandas will handle this mistmatch.. just ignores the end'
                
        
        #=======================================================================
        # #update with values from teh binv       
        #=======================================================================
        df.update(binv_df.loc[:,binvboolcol], overwrite=True) #update from all the values in teh binv

        logger.debug('retreived %i values from the binv_df on: %s'
                     %(binv_df.loc[:,binvboolcol].count().count(), binv_df.loc[:,binvboolcol].columns.tolist()))
        
        #=======================================================================
        # macro calcs
        #=======================================================================
        if 'hse_depth' in df.columns:
            df['hse_depth'] = df['wsl'] - df['anchor_el']
        
        #groudn water damage flag
        if 'gw_f' in df.columns:
            df.loc[:,'gw_f'] = df['dem_el'] > df['wsl'] #water is below grade
            
        if self.db_f:
            if 'bsmt_egrd' in binv_df.columns:
                raise IOError

        
        return df

    def get_dmg_set(self,  #calcluate the damage for each house
                    dmg_df, #empty frame for filling with damage results
                    #dmg_type_list='all', 
                    #bid_l = None,
                    #wsl_delta = None,
                    dmg_rat_f =None, #includt eh damage ratio in results
                    ):  
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        depth_ser: series of depths (for this flood) with index = bldg_id
        
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger          = self.logger.getChild('get_dmg_set(%s)'%self.get_id())

        if dmg_rat_f is None: dmg_rat_f = self.model.dmg_rat_f
        
        
        hse_od          = self.model.binv.hse_od  #ordred dictionary by bid: hse_dato
        
        """ see get_wsl_from_egrd()
        #=======================================================================
        # build the dmg_df
        #=======================================================================
        bid_ar = self.model.binv.data.loc[:,self.mind].values.astype(np.int) #get everything from teh binv
        dmg_df = pd.DataFrame(index = bid_ar, columns = self.model.dmg_df_cols)"""

        
        #=======================================================================
        # pre checks
        #=======================================================================
        if self.db_f:
            if not isinstance(dmg_df, pd.DataFrame):
                raise IOError
            
            boolidx = dmg_df.index.isin(hse_od.keys())
            if not np.all(boolidx):
                logger.error('some of the bldg_ids in the wsl_ser were not found in the binv: \n    %s'
                             %dmg_df.index[~boolidx])
                raise IOError
            
            #check the damage columns are empty
            boolcol = np.isin(dmg_df.columns, ['MS', 'MC', 'BS', 'BC', 'GS', 'total']) #identify damage columns
            
            if not np.all(pd.isnull(dmg_df.loc[:,boolcol])):
                raise IOError
        
        #=======================================================================
        # frame setup
        #=======================================================================
        #identify columns containing damage results
        dmgbool = np.logical_or(dmg_df.columns.isin(self.model.dmg_types), #damages
                                pd.Series(dmg_df.columns).str.contains('_rat').values) #damage ratios

        
        #=======================================================================
        # get teh damage for each house
        #=======================================================================
        logger.debug('getting damage for %s entries'%(str(dmg_df.shape)))
        """
        to improve performance, we're only looping through those entries with real flood deths (skin_df)
        however, the full results frame is still used (non_real entries should equal zero)
        """
        """generally no memory added during these
        self.session.prof(state='%s.get_dmg_set.loop'%(self.name)) #memory profiling"""
        cnt = 0
        first = True
        for index, row in dmg_df.iterrows(): #loop through each row
            #===================================================================
            # pre-printouts
            #===================================================================
            #self.session.prof(state='%s.get_dmg_set.%i'%(self.name, cnt)) #memory profiling
            cnt +=1
            if cnt%self.session._logstep == 0: logger.info('    (%i/%i)'%(cnt, len(dmg_df)))
            
            #===================================================================
            # retrive info
            #===================================================================
            
            hse_obj = hse_od[index] #get this house object by bldg_id
            hse_obj.floodo = self #let the house know who is flooding it
            logger.debug('on hse \'%s\' \n'%hse_obj.name)

            #===================================================================
            # add damage results
            #===================================================================
            
            if row['hse_depth'] < self.model.hse_skip_depth:
                logger.debug('depth below hse_obj.vuln_el for bldg_id: %i. setting fdmg=0'%index)
                row[dmgbool] = 0.0 #set all damage to zero
                
            #depth significant. calc it
            else:
                
                #runt he house
                logger.debug('running house \n')
                dmg_ser = hse_obj.run_hse(row['wsl'], dmg_rat_f = dmg_rat_f)
                
                row.update(dmg_ser) #add all these entries


            #===================================================================
            # extract extra attributers from teh house
            #===================================================================    
            #find the entries to skip attribute in filling
            if first:
                boolar1 = ~np.isin(row.index, ['total'])
                boolar2 = pd.isnull(row)
                boolar = np.logical_and(boolar1, boolar2)
                first = False
            
            #fill thtese
            for attn, v in row[boolar].iteritems(): 
                row[attn] = getattr(hse_obj, attn)
                
            #===================================================================
            # wrap up
            #===================================================================

            
            dmg_df.loc[index,:] = row #store this row back into the full resulst frame
            

        #=======================================================================
        # macro stats
        #=======================================================================
        #total
        boolcol = dmg_df.columns.isin(self.model.dmg_types)
        dmg_df['total'] = dmg_df.iloc[:,boolcol].sum(axis = 1) #get the sum
        
        #=======================================================================
        # closeout and reporting
        #=======================================================================

        #print out summaries
        if not self.db_f:
            logger.info('finished for %i houses'%(len(dmg_df.index)))
        else:
            totdmg = dmg_df['total'].sum()
        
            totdmg_str = '$' + "{:,.2f}".format(totdmg)
            
            logger.info('got totdmg = %s for %i houses'%(totdmg_str,len(dmg_df.index)))
        
            if np.any(pd.isnull(dmg_df)):
                raise IOError
            for dmg_type in self.model.dmg_types:
                dmg_tot = dmg_df[dmg_type].sum()
                dmg_tot_str = '$' + "{:,.2f}".format(dmg_tot)
                logger.debug('for dmg_type \'%s\' dmg_tot = %s'%(dmg_type, dmg_tot_str))
            
        return dmg_df
    
    
    def check_dmg_df(self, df):
        logger = self.logger.getChild('check_dmg_df')
        
        #=======================================================================
        # check totals
        #=======================================================================
        boolcol = np.isin(df.columns, ['MS', 'MC', 'BS', 'BC', 'GS']) #identify damage columns
        if not round(df['total'].sum(),2) == round(df.loc[:, boolcol].sum().sum(), 2):
            logger.error('total sum did not match sum from damages')
            raise IOError
        
    def calc_statres_flood(self, df): #calculate your statistics
        'running this always'
        logger = self.logger.getChild('calc_statres_flood')
        s = self.session.outpars_d[self.__class__.__name__]
        
        """needed?
        self.outpath =   os.path.join(self.model.outpath, self.name)"""
        
        #=======================================================================
        # total damage
        #=======================================================================
        for dmg_code in self.model.dmg_types + ['total']:
            
            #loop through and see if the user asked for this output
            'e.g. MC, MS, BC, BS, total'
            if dmg_code in s:
                v = df[dmg_code].sum()
                setattr(self, dmg_code, v)
                
                logger.debug('set \'%s\' to %.2f'%(dmg_code, v))
                
        #=======================================================================
        # by flood type
        #=======================================================================
        if 'dmg_sw' in s:
            self.dmg_sw = df.loc[~df['gw_f'], 'total'].sum() #sum all those with surface water
            
        if 'dmg_gw' in s:
            self.dmg_gw = df.loc[df['gw_f'], 'total'].sum() #sum all those with surface water
                            
        
        #=======================================================================
        # number of houses with damage
        #=======================================================================
        if 'hdmg_cnt' in s:

            boolidx = df.loc[:, 'total'] > 0
        
            self.hdmg_cnt = boolidx.sum()
            
        #=======================================================================
        # average house depth
        #=======================================================================
        if 'hdep_avg' in s:
            
            self.hdep_avg = np.mean(df.loc[:,'hse_depth'])
            
        #=======================================================================
        # wsl average
        #=======================================================================
        if 'wsl_avg' in s:
            self.wsl_avg = np.mean(df.loc[:,'wsl'])
            
            
        #=======================================================================
        # basement exposure grade counts
        #=======================================================================      
        'just calcing all if any of them are requested'  
        boolar = np.isin(np.array(['bwet_cnt', 'bdamp_cnt', 'bdry_cnt']),
                         np.array(s))
        
        if np.any(boolar): self.get_begrd_cnt()
        
        #=======================================================================
        # plots
        #=======================================================================
        if 'dmg_res_df' in s:
            self.dmg_res_df = df
        
        """
        hp.pd.v(df)
        """
        
        return
            
    def get_begrd_cnt(self):
        logger = self.logger.getChild('get_begrd_cnt')
        
        df = self.dmg_res_df
        
        #=======================================================================
        # #get egrades
        # try:
        #     ser = df.loc[:,'bsmt_egrd'] #make the slice of interest
        # except:
        #     df.columns.values.tolist()
        #     raise IOError
        #=======================================================================
        
        ser = df.loc[:,'bsmt_egrd'] #make the slice of interest
        
        begrd_l = ser.unique().tolist()
        
        logger.debug('looping through %i bsmt_egrds: %s'%(len(begrd_l), begrd_l))
        for bsmt_egrd in begrd_l:
            att_n = 'b'+bsmt_egrd+'_cnt'
            
            #count the number of occurances
            boolar = ser == bsmt_egrd
            
            setattr(self, att_n, int(boolar.sum()))
            
            logger.debug('setting \'%s\' = %i'%(att_n, boolar.sum()))
        
        logger.debug('finished \n')
                        
        return
                    
    def plot_dmg_pie(self, dmg_sum_ser_raw = None, 
                     exp_str = 1, title = None, wtf=None): #generate a pie chart for the damage
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        dmg_sum_ser:    series of damage values (see calc_summary_ser)
            index: dmg_types
            values: fdmg totals for each type for this flood
            
        exp_main: amoutn to explote structural damage values by
        """
        #=======================================================================
        # set defaults
        #=======================================================================
        logger = self.logger.getChild('plot_dmg_pie')
        if title == None: title = self.session.tag + ' '+self.name+' ' + 'dmgpie_plot'
        if wtf is None: wtf = self.session._write_figs
        
        if dmg_sum_ser_raw == None:  #just calculate
            dmg_sum_ser_raw = self.dmg_res_df[self.dmg_types].sum()
            #dmg_sum_ser_raw = self.calc_summary_ser()
            
        logger.debug('with dmg_sum_ser_raw: \n %s'%dmg_sum_ser_raw)
        #=======================================================================
        # data cleaning
        #=======================================================================
        #drop na
        dmg_sum_ser1 = dmg_sum_ser_raw.dropna()
        #drop zero values
        boolidx = dmg_sum_ser1 == 0
        dmg_sum_ser2 = dmg_sum_ser1[~boolidx]
        
        if np.all(boolidx):
            logger.warning('got zero damages. not pie plot generated')
            return
        
        if boolidx.sum() > 0:
            logger.warning('dmg_pie dropped %s zero totals'%dmg_sum_ser1.index[boolidx].tolist())
        
        dmg_sum_ser = dmg_sum_ser2
        #=======================================================================
        # get data
        #=======================================================================
        #shortcuts
        dmg_types = dmg_sum_ser.index.tolist()
        
        labels = dmg_types
        sizes = dmg_sum_ser.values.tolist()


        #=======================================================================
        # #get properties list from the dfunc tab
        #=======================================================================
        colors = []
        explode_list = []
        wed_lab_list = []
        dfunc_df = self.session.pars_df_d['dfunc']
        
        for dmg_type in dmg_types:
            boolidx = dfunc_df['dmg_type'] == dmg_type #id this dmg_type
            
            #color
            color = dfunc_df.loc[boolidx,'color'].values[0]
            colors.append(color) #add to the list
            
            #explode
            explode = dfunc_df.loc[boolidx,'explode'].values[0]
            explode_list.append(explode) #add to the list
            
            #wedge_lable
            wed_lab = '$' + "{:,.2f}".format(dmg_sum_ser[dmg_type])
            wed_lab_list.append(wed_lab)
            
            
        plt.close()
        fig, ax = plt.subplots()
        
        
        wedges = ax.pie(sizes, explode=explode_list, labels=labels, colors = colors,
               autopct=hp.plot.autopct_dollars(sizes), 
               shadow=True, startangle=90)
        
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        
        ax.set_title(title)
        
        if wtf: #write to file
            filetail = self.session.name + ' '+self.name+' ' + 'dmgpie_plot'
            filename = os.path.join(self.model.outpath, filetail)
            hp.plot.save_fig(self, fig, savepath_raw = filename)
            
        return ax
    
    def plot_dmg_scatter(self, #scatter plot of damage for each house
                         dmg_df_raw=None, yvar = 'hse_depth', xvar = 'total', plot_zeros=True,
                         title=None, wtf=None, ax=None, 
                         linewidth = 0, markersize = 3, marker = 'x',
                          **kwargs): 
        
        """
        for complex figures, axes should be passed and returned
        #=======================================================================
        # INPUTS
        #=======================================================================
        should really leave this for post processing
        plot_zeros: flag to indicate whether entries with x value = 0 should be included
        
        #=======================================================================
        # TODO
        #=======================================================================
        redo this with the plot worker
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('plot_dmg_scatter')
        if title == None: title = self.session.tag + ' '+self.name + ' dmg_scatter_plot'
        if wtf is None: wtf = self.session._write_figs
        
            
        if dmg_df_raw == None: 
            dmg_res_df_raw = self.dmg_res_df #just use the attached one
            
            if not hp.pd.isdf(dmg_res_df_raw): raise IOError
                
        #=======================================================================
        # manipulate data for plotting
        #=======================================================================
        if plot_zeros:
            dmg_df = dmg_res_df_raw
        else:
            #exclude those entries with zero value on the xvar
            boolidx = dmg_res_df_raw[xvar] == 0
            dmg_df = dmg_res_df_raw[~boolidx]
            self.logger.warning('%s values = zero (%i) excluded from plot'%(xvar, boolidx.sum()))
            
        #=======================================================================
        # setup data plot
        #=======================================================================
        x_ar = dmg_df[xvar].values.tolist() #damage
        xlab = 'damage($)' 
        'could make this more dynamic'
        
        if sum(x_ar) <=0:
            logger.warning('got no damage. no plot generated')
            return

        y_ar = dmg_df[yvar].values.tolist() #depth
        

        #=======================================================================
        # SEtup defaults
        #=======================================================================
        if ax == None:
            plt.close('all')
            fig = plt.figure(2)
            fig.set_size_inches(9, 6)
            ax = fig.add_subplot(111)

            ax.set_title(title)
            ax.set_ylabel(yvar + '(m)')
            ax.set_xlabel(xlab)
            
            #set limits
            #ax.set_xlim(min(x_ar), max(x_ar))
            #ax.set_ylim(min(y_ar), max(y_ar))
        else:
            fig = ax.figure
            
        label = self.name + ' ' + xvar
        #=======================================================================
        # send teh data for plotting
        #=======================================================================
        
        pline = ax.plot(x_ar,y_ar, 
                        label = label,
                        linewidth = linewidth, markersize = markersize, marker = marker,
                        **kwargs)
        

        
        #=======================================================================
        # post formatting
        #=======================================================================
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        """

        plt.show()

        
        """

        if wtf: #trigger for saving the fiture
            filetail = title
            filename = os.path.join(self.model.outpath, filetail)
            hp.plot.save_fig(self, fig, savepath_raw = filename, logger=logger)

        
        return pline
    
class Binv(     #class object for a building inventory
                hp.data.Data_wrapper,
                hp.plot.Plot_o,
                hp.sim.Sim_o,
                hp.oop.Parent,
                hp.oop.Child): 
    #===========================================================================
    # program pars
    #===========================================================================
    # legacy index numbers
    legacy_ind_d = {0:'ID',1:'address',2:'CPID',10:'class', 11:'struct_type', 13:'gis_area', \
                    18:'bsmt_f', 19:'ff_height', 20:'xcoord',21:'ycoord', 25:'dem_el'}
    
    #column index where the legacy binv transitions to teh new binv
    legacy_break_ind = 26
    
    #column names expected in the cleaned binv
    exepcted_coln = ['gis_area', 'bsmt_f', 'ff_height',\
                     'dem_el', 'value', 'ayoc', 'B_f_height',\
                     'bkflowv_f','sumpump_f', 'genorat_f', 'hse_type', \
                     'name', 'anchor_el', 'parcel_area']


    
    hse_type_list = ['AA', 'AD', 'BA', 'BC', 'BD', 'CA', 'CC', 'CD'] #classification of building types
    
    
    #===========================================================================
    # user provided
    #===========================================================================
    legacy_binv_f       = True


    #===========================================================================
    # calculated pars
    #===========================================================================

    
    #===========================================================================
    # data holders
    #===========================================================================
    #cnt = 0
    hnew_cnt    = 0
    hAD_cnt     = 0


    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Binv')
        logger.debug('start _init_')
        
        
        """Im explicitly attaching the child datobuilder here 
        dont want to change the syntax of the binv
        
        inspect.isclass(self.kid_class)
        """
        self.inherit_parent_ans=set(['mind', 'legacy_binv_f', 'gis_area_max'])
        
        super(Binv, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
        
        
        #=======================================================================
        # special inheritance
        #=======================================================================
        #self.model = self.parent
        self.kid_class = House
        
        self.reset_d.update({'hnew_cnt':0, 'hAD_cnt':0})
        #=======================================================================
        # checks
        #=======================================================================
        if self.db_f:
            if not self.kid_class == House:
                raise IOError
            
            if not isinstance(self.reset_d, dict):
                raise IOError
        
            if self.model is None:
                raise IOError
            
            if not self.model.name == self.parent.name:
                raise IOError
            
        #=======================================================================
        # special inits
        #=======================================================================
        self.exepcted_coln = set(self.exepcted_coln + [self.mind]) #expect the mind in the column names as well
        
        self.load_data()

        logger.debug('finiished _init_ \n')
        return
        

    def load_data(self): #custom data loader
        #=======================================================================
        # defaults
        #=======================================================================

        logger = self.logger.getChild('load_data')
        #test pars
        if self.session._parlo_f: 
            test_trim_row = self.test_trim_row
        else: test_trim_row = None
        
        #=======================================================================
        # load the file
        #=======================================================================
        self.filepath = self.get_filepath()
        
        logger.debug('from filepath: %s'%self.filepath)
        
        #load from file
        df_raw = hp.pd.load_xls_df(self.filepath, logger=logger, test_trim_row = test_trim_row, 
                header = 0, index_col = None)
                
        #=======================================================================
        # send for cleaning
        #=======================================================================
        df1 = hp.pd.clean_datapars(df_raw, logger = logger)
        """
        hp.pd.v(df3)
        """
        
        #=======================================================================
        # clean per the leagacy binv
        #=======================================================================
        if self.legacy_binv_f:
            df2 = self.legacy_clean_df(df1)
        else:
            df2 = df1
            
        #=======================================================================
        # standard clean   
        #=======================================================================
        df3 = self.clean_inv_df(df2)
        
        
        #=======================================================================
        # macro data manipulations
        #=======================================================================
                
        #add names column
        if not 'name' in df3.columns:
            df3['name'] = 'h' + df3.loc[:, self.mind].astype(np.string_) #format as strings
        
        
        #add anchor el
        if not 'anchor_el' in df3.columns:
            df3['anchor_el'] = df3['dem_el'] + df3['ff_height']
            df3['anchor_el'] = df3['anchor_el'].astype(np.float)
        
        #=======================================================================
        # checking
        #=======================================================================
        if self.db_f: self.check_binv_df(df3)
                
        #=======================================================================
        # wrap up
        #=======================================================================
        self.childmeta_df = df3.copy()
        
        #shortcut lists
        self.bid_l = df3[self.mind].astype(np.int).values.tolist()
        self.hse_types_l = df3['hse_type'].unique().tolist()
        
        

        logger.info('attached binv_df with %s'%str(df3.shape))
        
        return
    """
    hp.pd.v(df3)
    """
            

    
    def legacy_clean_df(self, df_raw): #compile data from legacy (rfda) inventory syntax
        """
        pulling column headers from the dictionary of location keys
        
        creating some new headers as combinations of this
        
        """
        #=======================================================================
        # setup
        #=======================================================================
        logger = self.logger.getChild('legacy_clean_df')
        
        d = self.legacy_ind_d
        
        #=======================================================================
        # split the df into legacy and non
        #=======================================================================
        df_leg_raw = df_raw.iloc[:,0:self.legacy_break_ind]
        df_new = df_raw.iloc[:,self.legacy_break_ind+1:]
        
        #=======================================================================
        # clean the legacy frame
        #=======================================================================

        #change all the column names
        df_leg1 = df_leg_raw.copy()
        
        """ couldnt get this to work
        df_leg1.rename(mapper=d, index = 'column')"""

        
        for colind, coln in enumerate(df_leg_raw.columns):
            if not colind in d.keys():continue

            df_leg1.rename(columns = {coln:d[colind]}, inplace=True)
            
            logger.debug('renamed \'%s\' to \'%s\''%(coln,d[colind] ))
            
        #trim down to these useful columns
        boolcol = df_leg1.columns.isin(d.values()) #identify columns in the translation dictionary
        df_leg2 = df_leg1.loc[:,boolcol]
        
        logger.debug('trimmed legacy binv from %i to %i cols'%(len(df_leg_raw.columns), boolcol.sum()))
        
        #=======================================================================
        # add back the new frame
        #=======================================================================
        df_merge = df_leg2.join(df_new)

        #=======================================================================
        #  house t ype
        #=======================================================================        
        df_merge.loc[:,'hse_type'] =  df_leg2.loc[:,'class'] + df_leg2.loc[:,'struct_type']
        
        logger.debug('cleaned the binv from %s to %s'%(str(df_raw.shape), str(df_merge.shape)))
        
        if self.db_f:
            if not len(df_merge) == len(df_raw):
                raise IOError
            if np.any(pd.isnull(df_merge['hse_type'])):
                raise IOError
        
        return df_merge
        """
        hp.pd.v(df_leg_raw)
        hp.pd.v(df_merge)
        hp.pd.v(df_raw)
        """
        
    def clean_inv_df(self, df_raw): #placeholder for custom cleaning
        logger = self.logger.getChild('clean_inv_df')
        #clean with kill_flags
        'this makes it easy to trim the data'
        df1 = hp.pd.clean_kill_flag(df_raw, logger = logger)
        
        #=======================================================================
        # format boolean columns
        #=======================================================================
        df1 = hp.pd.format_bool_cols(df1, logger = logger)
        
        #=======================================================================
        # #reformat
        #=======================================================================
        # the MIND as integer
        df1.loc[:, self.mind] = df1.loc[:, self.mind].astype(np.int) #reset as an integer
        
        
        #ayoc as an intger
        df1['ayoc'] = df1['ayoc'].astype(np.int)
        
        #df1['hse_type'] = df1['hse_type'].astype(np.string_)
                
        
        #=======================================================================
        # #reindex by a sorted mind (and keep the column)
        #=======================================================================
        df2 = df1.set_index(self.mind, drop=False).sort_index()
        
        #=======================================================================
        # trim to the desired columns
        #=======================================================================
        boolcol = df2.columns.isin(self.exepcted_coln)
        df3 = df2.loc[:,boolcol]
        
        
        return df3
    
    """
    df1.columns.str.strip()
    
    df2.columns[~boolcol]
    hp.pd.v(df2)
    """
        

    def check_binv_df(self, df):  
        logger = self.logger.getChild('check_binv_df')
        'todo: add some template check'
        if not hp.pd.isdf(df):
            raise IOError
        
        if np.any(pd.isnull(df)):
            raise IOError
        
        #=======================================================================
        # check all the expected columns are there
        #=======================================================================
        boolcol = np.isin(list(self.exepcted_coln), df.columns)
        if not np.all(boolcol):
            logger.error('could not find \'%s\' in the binv_df'
                         %np.array(list(self.exepcted_coln))[~boolcol])
            """
            hp.pd.v(df)
            """
            raise IOError
        #=======================================================================
        # check area column
        #=======================================================================
        boolidx = df.loc[:,'gis_area']< self.model.gis_area_min
        if np.any(boolidx):
            logger.error('got %i binv entries with area < 5'%boolidx.sum())
            raise IOError
        
        boolidx = df.loc[:,'gis_area']> self.model.gis_area_max
        if np.any(boolidx):
            logger.error('got %i binv entries with area > %.2f'%(boolidx.sum(), self.model.gis_area_max))
            raise IOError
        
        if 'bsmt_egrd' in df:
            raise IOError
            
        return 
        
#===============================================================================
#     def Xget_childmetadf(self): #custom childmeta builder
#         """
#         this should overwrite hte default function and be called from raise_children
#         Here we add the hierarchy info to the bldg inventory 
#         so the children can be raised
#         """
#         logger = self.logger.getChild('get_childmetadf')
#         df1 = self.data.copy()
#         
#         logger.debug('with data %s'%(str(df1.shape)))
#         #=======================================================================
#         # macro data manipulations
#         #=======================================================================
#                 
#         #add names column
#         if not 'name' in df1.columns:
#             df1['name'] = 'h' + df1.loc[:, self.mind].astype(np.string_) #format as strings
#         
#         
#         #add anchor el
#         if not 'anchor_el' in df1.columns:
#             df1['anchor_el'] = df1['dem_el'] + df1['ff_height']
#             df1['anchor_el'] = df1['anchor_el'].astype(np.float)
#         
#         """
#         see House.set_hse_anchor()
#         anchor_el = self.dem_el + float(self.ff_height) #height + surface elevation
#         """
#         
#         
#         #=======================================================================
#         # wrap up
#         #=======================================================================
#         self.childmeta_df = df1
#         
#         """want to capture some of the edits made by House
#         moved this to after raise_chidlren
#         #add the df tot he rest list
#         self.reset_d['childmeta_df'] = df2.copy()"""
# 
#         'House makes some edits to this so we need to update this copy'
#         
#         """
#         hp.pd.v(df1)
#         hp.pd.v(df2)
#         hp.pd.v(self.childmeta_df)
#         hp.pd.v(self.data)
#         """
#         
#         return
#     
#===============================================================================
    def raise_houses(self): 
        #=======================================================================
        # setup
        #=======================================================================
        start = time.time()
        logger = self.logger.getChild('raise_houses')
        
        
        df = self.childmeta_df #build the childmeta intelligently
        'we could probably just passt the data directly'
        
        if self.db_f:
            if not hp.pd.isdf(df):
                raise IOError
        
        logger.info('executing with data %s'%str(df.shape))
        
        hse_n_d = self.raise_children_df(df, #run teh generic child raiser
                                         kid_class = self.kid_class,
                                         dup_sibs_f = True) 
        """
        House.spc_inherit_anl
        self.kid_class
        """
        
        #=======================================================================
        # add a custom sorted dictionary by name
        #=======================================================================
        #build a normal dictionary of this
        d = dict() 
        for cname, childo in hse_n_d.iteritems(): 
            d[childo.bldg_id] = weakref.proxy(childo)
            
        #bundle this into a sorted ordered dict
        self.hse_od = OrderedDict(sorted(d.items(), key=lambda t: t[0]))
        
        """put this here so the edits made by House are captured"""
        self.reset_d['childmeta_df'] = self.childmeta_df.copy()
        
        logger.debug('calc_binv_stats() \n')
        self.calc_binv_stats()
        
        stop = time.time()
        logger.info('finished with %i hosues in %.4f secs'%(len(d), stop - start))
    
        
        return
            

    
    def set_all_hse_atts(self, attn,  #reset an attribute name/value pair to all houses in the binv
                         attv=None, #single value to apply to each house
                         ser=None, #series to pull data from indexed by the obj_key
                         obj_key = 'dfloc',
                         ): 
        """
        NOTE: oop.attach_att_df() is similar, but doesnt handle the dynamic updating
        udev.set_fhr is also similar
        
        ToDo: consider moving this into dyno
        
        
        """
        logger = self.logger.getChild('set_all_hse_atts')
        #=======================================================================
        # precheck
        #=======================================================================
        if self.db_f:
            if not ser is None:
                if not isinstance(ser, pd.Series):
                    raise IOError
                
            if not len(self.hse_od) > 0:
                raise IOError
            
            if (attv is None) and (ser is None):
                raise IOError #need at least one input

        #=======================================================================
        # loop and add to each house
        #=======================================================================
        logger.debug('dynamically updating %i houses on \'%s\''%(len(self.hse_od), attn))
        
        for k, hse in self.hse_od.iteritems():
            
            if not ser is None:
                attv = ser[getattr(hse, obj_key)]
                
            #haqndle the updates on this house
            hse.handle_upd(attn, attv, proxy(self), call_func = 'set_all_hse_atts') 
            
        return
    
    """
    df = self.childmeta_df
    df.columns
    hp.pd.v(df)
    """

     
    def calc_binv_stats(self): #calculate output stats on the inventory
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        __init__
            raise_children #after raising all the Houses
            
        (Fdmg or Udev).get_restults()

            
        
        #=======================================================================
        # TODO
        #=======================================================================
        fix this so it acts more like a dynp.update with ques triggerd from changes on the HOuse
        
        #=======================================================================
        # TESTING
        #=======================================================================
        hp.pd.v(df)

        df.columns.values.tolist()
        """
        #logger = self.logger.getChild('calc_binv_stats')
        
        s = self.session.outpars_d[self.__class__.__name__]

        
        """using this in some annotations
        if 'cnt' in s:"""
    
        self.cnt = len(self.hse_od) #get the number of houses in the binv
        
        if 'hAD_cnt' in s:
            #house type counts
            boolidx = self.childmeta_df.loc[:, 'hse_type'] == 'AD'
            self.hAD_cnt = boolidx.sum()
            
        if 'hnew_cnt' in s:
        
            #new house counts
            boolidx = self.childmeta_df.loc[:,'ayoc'] > self.session.year0
            self.hnew_cnt = boolidx.sum()
            
        return
        
    def get_bsmt_egrds(self, set_f=False):
        
        logger = self.logger.getChild('get_bsmt_egrds')
        
        df = self.childmeta_df
        
        if not 'bsmt_egrd' in df.columns:
            #self.session.state
            raise IOError
        
        #basement exposure grade
        logger.debug('getting bsmt_egrd stats on %s'%(str(df.shape)))
        
        d = dict()
        
        for grade in ['wet', 'dry', 'damp']: #loop through and count all the finds
            
            #get count
            boolidx = df.loc[:,'bsmt_egrd'] == grade
            
            cnt = boolidx.sum()
            
            d[grade] = cnt
            
            #set as attribute
            if set_f:
                new_an = '%s_cnt'%grade
                
                setattr(self, new_an, cnt)
            
            logger.debug('for bsmt_egrd = \'%s\' found %i'%(grade,cnt))
            
        return d
              
    def write(self): #write the current binv to file
        logger = self.logger.getChild('write')
        
        df = self.childmeta_df
        """
        hp.pd.v(df)
        """
        
        filename = '%s binv'%(self.session.state)
        filehead = self.model.tstep_o.outpath
        
        filepath = os.path.join(filehead, filename)
        
        hp.pd.write_to_file(filepath, df, logger=logger)
                
        return
        
          
class House(
            udev.scripts.House_udev,
            hp.plot.Plot_o, 
            hp.dyno.Dyno_wrap,
            hp.sim.Sim_o,  
            hp.oop.Parent, #building/asset objects 
            hp.oop.Child): 
    
    #===========================================================================
    # program pars
    #==========================================================================
    geocode_list        = ['area', 'per', 'height', 'inta'] #sufficxs of geometry attributes to search for (see set_geo)
    finish_code_list    = ['f', 'u', 't'] #code for finished or unfinished
    

    #===========================================================================
    # debugging
    #===========================================================================
    last_floodo = None
    
    #===========================================================================
    # user provided pars
    #===========================================================================
    dem_el      = None    
    hse_type    = None # Class + Type categorizing the house
    anchor_el   = None # anchor elevation for house relative to datum (generally main floor el)   
    gis_area    = None #foot print area (generally from teh binv)
    bsmt_f   = True
    area_prot_lvl = 0 #level of area protection
    
    B_f_height = None
    
    #defaults passed from model
    """While the ICS for these are typically uniform and broadcast down by the model,
    these need to exist on the House, so we can spatially limit our changes"""
    G_anchor_ht   = None   #default garage anchor height (chosen aribtrarily by IBI (2015)
    joist_space   = None   #space between basement and mainfloor. used to set the 
    

    #===========================================================================
    # calculated pars
    #===========================================================================
    floodo      = None  #flood object flooding the house

    # #geometry placeholders
    #geo_dxcol_blank = None #blank dxcol for houes geometry
    geo_dxcol = None

    
    
    'keeping just this one for reporting and dynp'
    
    
    boh_min_val = None #basement open height minimum value
    #===========================================================================
    # B_f_area = None #basement finished (liveable) area
    # B_f_per  = None #perimeter
    # B_f_inta = None
    # 
    # B_u_area = None
    # B_u_per  = None
    # B_u_inta = None
    # 
    # M_f_area = None
    # M_f_per  = None
    # M_f_inta = None
    #  
    # M_u_area = None #mainfloor non-finisehd area
    # M_u_per  = None 
    # M_u_inta = None
    #    
    # """
    # For garages, the assessment records have the area under
    #     BLDG_TOTAL_NONLIV_AREA_ABOVE and P2.
    #     average = 48m2. 
    # for the legacy rfda dmg_feat_tables, we don't know what the base area was for the garage
    # lets assume 50m2
    # also, these are usually pretty square
    # """
    # G_f_area = None
    # G_f_per  = None
    # G_f_inta = None
    # 
    # G_u_area = None
    # G_u_per  = None
    # G_u_inta = None
    # 
    # #heights
    # """these are interior aeras, Bheight + joist space = B_anchor_ht
    # assumed some typical values from the internet.
    # really shouldnt use the NONE values here.. these are just placeholders"""
    # M_f_height = None #default mainfloor height
    # B_f_height = None
    # G_f_height = None
    # 
    # M_u_height = None
    # B_u_height = None
    # G_u_height = None
    #===========================================================================
    
        
    # #anchoring
    """
    Im keeping anchor heights separate from geometry attributes as these could still apply
    even for static dmg_feats
    """
    
    bsmt_opn_ht   = 0.0   #height of lowest basement opening
    damp_spill_ht = 0.0
    
    vuln_el       = 9999   #starter value

    # personal property protection
    bkflowv_f       = False #flag indicating the presence of a backflow  valve on this property
    sumpump_f       = False
    genorat_f       = False
    
    bsmt_egrd   = ''
    
    #statistics
    BS_ints     = 0.0 #some statistic of the weighted depth/damage of the BS dfunc
    max_dmg     = 0.0 #max damage possible for this house 
    
    #===========================================================================
    # data containers
    #===========================================================================
    dd_df       = None #df results of total depth damage 


    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('House')
        logger.debug('start _init_')
            #=======================================================================
        # attach pre init atts
        #=======================================================================
        #self.model              = self.parent.model #pass the Fdmg model down
        'put this here just to keep the order nice and avoid the unresolved import error'
        
        self.inherit_parent_ans=set(['mind', 'model'])

        #=======================================================================
        # #initilzie teh baseclass
        #=======================================================================
        super(House, self).__init__(*vars, **kwargs) 
        
        if self.db_f:
            if self.model is None: raise IOError
        

        #=======================================================================
        #common setup
        #=======================================================================
        if self.sib_cnt == 0:
            logger.debug("sib_cnt=0. setting atts")
            self.kid_class          = Dfunc
            self.childmeta_df       = self.model.house_childmeta_df #dfunc meta data
            
            self.joist_space        = self.model.joist_space
            self.G_anchor_ht        = self.model.G_anchor_ht
            
        #=======================================================================
        # unique se5tup
        #=======================================================================
        self.bldg_id            = int(getattr(self, self.mind ))
        #self.ayoc               = int(self.ayoc)
        #self.area_prot_lvl      = int(self.area_prot_lvl)
        self.bsmt_f            = hp.basic.str_to_bool(self.bsmt_f, logger=self.logger)

        
        if not 'B' in self.model.place_codes:
            self.bsmt_f = False
        
        'these need to be unique. calculated during init_dyno()'
        self.post_upd_func_s = set([self.calc_statres_hse])
        
        """ahndled by dyno
        self.reset_d['hse_type']      = self.hse_type 
        'using this for type change checking'
        
        self.kid_class
        """
        logger.debug('building the house \n')
        self.build_house()
        logger.debug('raising my dfuncs \n')
        self.raise_dfuncs()
        logger.debug('init_dyno \n')
        self.init_dyno()
        
        #=======================================================================
        # cheking
        #=======================================================================
        if self.db_f: self.check_house()
        
        logger.debug('_init_ finished as %i \n'%self.bldg_id)
        
        return
    
    def check_house(self):
        logger = self.logger.getChild('check_house')
        if not self.model.__repr__() == self.parent.parent.__repr__(): 
            raise IOError
        
        #=======================================================================
        # garage area check
        #=======================================================================
        g_area = self.geo_dxcol.loc['area',('G','u')]
        if g_area > self.gis_area:
            logger.error('got garage area greater than foot print for the house!')
            
            """if we use the legacy areas for the garage curves this will often be the case
            raise IOError"""
        
        return
    
    def build_house(self): #buidl yourself from the building inventory
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        binv.raise_children()
            spawn_child()
        
        """
        logger = self.logger.getChild('build_house')
    
        #=======================================================================
        # custom loader functions
        #=======================================================================
        #self.set_binv_legacy_atts() #compile data from legacy (rfda) inventory syntax
        logger.debug('\n')
        self.set_geo_dxcol() #calculate the geometry (defaults) of each floor
        logger.debug('\n')
        self.set_hse_anchor()
                        
        """ a bit redundant, but we need to set the bsmt egrade regardless for reporting consistency
        'these should be accessible regardless of dfeats as they only influence the depth calc'"""
        self.set_bsmt_egrd()
        
        if self.bsmt_f:
            logger.debug('\n')
            self.set_bsmt_opn_ht()
            logger.debug('set_damp_spill_ht() \n')
            self.set_damp_spill_ht()
            
            
        #=======================================================================
        # value
        #=======================================================================
        'need a better way to do this'
        self.cont_val = self.value * self.model.cont_val_scale
            
        if self.db_f:    
            
            if self.gis_area < self.model.gis_area_min:
                raise IOError
            if self.gis_area > self.model.gis_area_max: raise IOError
                        
        logger.debug('finished as %s \n'%self.hse_type)

        
    def raise_dfuncs(self): #build dictionary with damage functions for each dmg_type
        """
        2018 06 05: This function isnt setup very well
        
        called by spawn_child and passing childmeta_df (from dfunc tab. see above)
        this allows each dfunc object to be called form the dictionary by dmg_type
        
        dfunc_df is sent as the childmeta_df (attached during __init__)
        #=======================================================================
        # INPUTS
        #=======================================================================
        dfunc_df:    df with headers:

        these are typically assigned from the 'dfunc' tab on the pars.xls
        #=======================================================================
        # TESTING
        #=======================================================================
        hp.pd.v(childmeta_df)
        """
        #=======================================================================
        # #defautls
        #=======================================================================
        logger = self.logger.getChild('raise_dfuncs')
        logger.debug('starting')
        
        #self.kids_d = dict() #reset this just incase
        
        df = self.childmeta_df
        'this is a slice from the dfunc tab made by Fdmg.load_pars_dfunc'
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not hp.pd.isdf(df): raise IOError
            if len(df) == 0:  raise IOError
            if not self.kid_class == Dfunc:
                raise IOError
            if len(self.kids_d) > 0: raise IOError
        #=======================================================================
        # compile for each damage type
        #=======================================================================
        self.dfunc_d = self.raise_children_df(df,
                               kid_class = self.kid_class,
                               dup_sibs_f = True)
                    
        #=======================================================================
        # closeout and wrap up
        #=======================================================================
        logger.debug('built %i dfunc children: %s'%(len(self.dfunc_d), self.dfunc_d.keys()))
        
        return 
    


    def set_hse_anchor(self):
        'pulled this out so updates can be made to dem_el'
        if self.is_frozen('anchor_el'): return True
        
        anchor_el = self.dem_el + float(self.ff_height) #height + surface elevation
        
        #set the update
        self.handle_upd('anchor_el', anchor_el, proxy(self), call_func = 'set_hse_anchor')
        
        return True
            
    def set_bsmt_opn_ht(self):
        """
        bsmt_open_ht is used by dfuncs with bsmt_e_grd == 'damp' and damp_func_code == 'spill' 
            for low water floods
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if not self.bsmt_f: return True
        
        #=======================================================================
        # check dependencies and frozen
        #=========================================================== ============
        if not self.session.state=='init':

            if self.is_frozen('bsmt_opn_ht'): return True
            
            dep_l =  [([self], ['set_hse_anchor()', 'set_geo_dxcol()'])]
            
            if self.deps_is_dated(dep_l, method = 'reque', caller = 'set_bsmt_opn_ht'):
                return False
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_bsmt_opn_ht')
        
        #=======================================================================
        # from user provided minimum
        #=======================================================================
        if self.model.bsmt_opn_ht_code.startswith('*min'):
            #first time calcs
            if self.boh_min_val is None:
                'this means we are non dynamic'
                s_raw = self.model.bsmt_opn_ht_code
                s = re.sub('\)', '',s_raw[5:])
                self.boh_min_val = float(s) #pull the number out of the brackets
                
            min_val = self.boh_min_val

            # get the basement anchor el
            B_f_height = float(self.geo_dxcol.loc['height',('B','t')]) #pull from frame
        
            bsmt_anchor_el = self.anchor_el - B_f_height - self.joist_space#basement curve
        

            #get the distance to grade
            bsmt_to_dem = self.dem_el - bsmt_anchor_el
                
            #take the min of all three
            bsmt_opn_ht = min(B_f_height, bsmt_to_dem, min_val)
            
            if self.db_f:
                #detailed output
                boolar = np.array([B_f_height, bsmt_to_dem, min_val]) == bsmt_opn_ht
                
                selected = np.array(['B_f_height', 'bsmt_to_dem', 'min_val'])[boolar]
                
                logger.debug('got bsmt_opn_ht = %.2f from \'%s\''%(bsmt_opn_ht, selected[0]))
                
            else:
                logger.debug('got bsmt_opn_ht = %.2f ')
            
        #=======================================================================
        # from user provided float
        #=======================================================================
        else:
            bsmt_opn_ht = float(self.model.bsmt_opn_ht_code)
        
        #=======================================================================
        # wrap up
        #=======================================================================
        self.handle_upd('bsmt_opn_ht', bsmt_opn_ht, proxy(self), call_func = 'set_bsmt_opn_ht')
        
        if self.db_f:
            if not bsmt_opn_ht > 0:
                raise IOError
        
        return True
    
    def set_damp_spill_ht(self):

        damp_spill_ht = self.bsmt_opn_ht / 2.0
        
        self.handle_upd('damp_spill_ht', damp_spill_ht, proxy(self), call_func = 'set_damp_spill_ht')   

        return True
              
    
    def set_bsmt_egrd(self): #calculate the basement exposure grade
        """
        bkflowv_f    sumpump_f    genorat_f

        There is also a globabl flag to indicate whether bsmt_egrd should be considered or not
        
        for the implementation of the bsmt_egrd in determining damages, see Dfunc.get_dmg_wsl()
        
        #=======================================================================
        # CALLS
        #=======================================================================
        this is now called during every get_dmgs_wsls()... as gpwr_f is a function of the Flood object
        
        consider only calling w
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if self.is_frozen('bsmt_egrd'):return

        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_bsmt_egrd')
        
        if self.bsmt_f:
            #=======================================================================
            # from plpms
            #=======================================================================
            if self.model.bsmt_egrd_code == 'plpm':
                cond = 'plpm'
                #=======================================================================
                # get the grid power state
                #=======================================================================
                if self.session.state == 'init':
                    gpwr_f = self.model.gpwr_f
                    cond = cond + '.init'
                else:
                    gpwr_f = self.floodo.gpwr_f
                    cond = '%s.%s'%(cond, self.floodo.name)
            
                #=======================================================================
                # grid power is on
                #=======================================================================
                if gpwr_f:
                    cond = cond + '.on'
                    if self.bkflowv_f and self.sumpump_f:
                        bsmt_egrd = 'dry'
                        
                    elif self.bkflowv_f or self.sumpump_f:
                        bsmt_egrd = 'damp'
                        
                    else: 
                        bsmt_egrd = 'wet'
                    
                #=======================================================================
                # grid power is off
                #=======================================================================
                else:
                    cond = cond + '.off'
                    if self.bkflowv_f and self.sumpump_f and self.genorat_f:
                        bsmt_egrd = 'dry'
                        
                    elif self.bkflowv_f or (self.sumpump_f and self.genorat_f):
                        bsmt_egrd = 'damp'
                        
                    else: bsmt_egrd = 'wet'
                
                self.gpwr_f = gpwr_f #set this
                logger.debug('set bsmt_egrd = %s (from \'%s\') with grid_power_f = %s'%(bsmt_egrd,self.bsmt_egrd, gpwr_f))
                
            #=======================================================================
            # ignore bsmt_egrd
            #=======================================================================
            elif self.model.bsmt_egrd_code == 'none':
                cond = 'none'
                bsmt_egrd = 'wet'
            
            #=======================================================================
            # allow the user to override all
            #=======================================================================
            elif self.model.bsmt_egrd_code in ['wet', 'damp', 'dry']:
                cond = 'global'
                bsmt_egrd = self.model.bsmt_egrd_code
            
            else:
                raise IOError
            
        else:
            cond = 'nobsmt'
            bsmt_egrd = 'nobsmt'
        
        #=======================================================================
        # wrap up
        #=======================================================================
        self.bsmt_egrd = bsmt_egrd
        
        """report/collect on the flood
        self.parent.childmeta_df.loc[self.dfloc,'bsmt_egrd'] = bsmt_egrd"""
        
        """
        if self.db_f:
            if not self.session.state == 'init':
                
                #===============================================================
                # write the beg histor y
                #===============================================================
                if self.model.beg_hist_df is None:                
                    self.model.beg_hist_df.loc[self.dfloc, self.floodo.name] = '%s_%s'%(bsmt_egrd, cond)"""


        return cond

        
    def set_geo_dxcol(self): #calculate the geometry of each floor based on the geo_build_code
        """
        builds a dxcol with all the geometry attributes of this house

        
        called by load_data when self.session.wdfeats_f = True
        
        #=======================================================================
        # KEY VARS
        #=======================================================================
        geo_build_code: code to indicate what geometry to use for the house. see the dfunc tab
            'defaults': see House.get_default_geo()
            'from_self': expect all geo atts from the binv.
            'any': take what you can from the binv, everything else use defaults.
            'legacy': use gis area for everything
            
        gbc_override: used to override the geo_build_code
        
        geo_dxcol: house geometry
        
        #=======================================================================
        # UDPATES
        #=======================================================================
        when a specific geometry attribute of the house is updated (i.e. B_f_height)
        this dxcol needs to be rebuilt
        and all the dfuncs need to run build_dd_ar()
                    

        """
        logger = self.logger.getChild('set_geo_dxcol')
        
        if self.is_frozen('geo_dxcol', logger=logger): 
            return True
        
        pars_dxcol = self.session.pars_df_d['hse_geo'] #pull the pars frame
        
        #=======================================================================
        # get default geometry for this house
        #=======================================================================
        self.defa = self.gis_area #default area
        
        
        if self.defa <=0:
            logger.error('got negative area = %.2f'%self.defa)
            raise IOError
        
        self.defp = 4*math.sqrt(self.defa)

        #=======================================================================
        # setup the geo_dxcol
        #=======================================================================
            
        dxcol = self.model.geo_dxcol_blank.copy() #get a copy of the blank one\
        
        'I need to place the reference herer so that geometry attributes have access to each other'
        #self.geo_dxcol = dxcol
            
        place_codes     = dxcol.columns.get_level_values(0).unique().tolist()
        #finish_codes    = dxcol.columns.get_level_values(1).unique().tolist()
        #geo_codes       = dxcol.index
        
        
                
        logger.debug("for hse_type \'%s\' from geo_dxcol_blank %s filling:"%(self.hse_type, str(dxcol.shape)))
        #=======================================================================
        # #loop through each place code and compile the appropriate geometry
        #=======================================================================
        for place_code in place_codes:
            geo_df = dxcol[place_code] #geometry for just this place           
            pars_df = pars_dxcol[place_code]
            
            #logger.debug('filling geo_df for place_code: \'%s\' '%(place_code))        
            #===================================================================
            # #loop through and build the geometry by each geocode
            #===================================================================
            for geo_code, row in geo_df.iterrows():
                
                for finish_code, value in row.iteritems():
                    
                    #===========================================================
                    # total column
                    #===========================================================
                    if finish_code == 't':
                        uval = dxcol.loc[geo_code, (place_code, 'u')]
                        fval = dxcol.loc[geo_code, (place_code, 'f')]
                        
                        if self.db_f:
                            if np.any(pd.isnull([uval, fval])):
                                raise IOError
                        
                        
                        if geo_code == 'height': #for height, take the maximum
                            att_val = max(uval, fval)
                                                        
                        else: #for other geometry, take the total
                            att_val = uval + fval
                    
                    #===========================================================
                    # finish/unfinished                       
                    #===========================================================
                    else:
                        #get the user passed par for this
                        gbc = pars_df.loc[geo_code, finish_code]
                        
                        try:gbc = float(gbc)
                        except: pass

                        #===========================================================
                        # #assemble per the geo_build_code
                        #===========================================================
                        #user specified code
                        if isinstance(gbc, basestring):
                            gbc = str(gbc)
                            if gbc == '*binv':
                                att_name = place_code +'_'+finish_code+'_'+ geo_code #get the att name for this
                                att_val = getattr(self, att_name) #get this attribute from self
                                
                            elif gbc == '*geo':
                                att_val = self.calc_secondary_geo(place_code, finish_code, geo_code, dxcol=dxcol) #calculate the default value
                                
                            elif gbc.startswith('*tab'):
                                #get the pars
                                tabn = re.sub('\)',"",gbc[5:]) #remove the end parentheisis
                                df = self.session.pars_df_d[tabn]
                                
                                att_name = place_code +'_'+finish_code+'_'+ geo_code #get the att name for this
                                
                                att_val = self.get_geo_from_other(df, att_name)
                                
                            else:
                                att_val = getattr(self, gbc)
                            
                        #user speciifed value
                        elif isinstance(gbc, float): #just use the default value provided in the pars
                            att_val = gbc
                            
                        else: raise IOError
                        
                        logger.debug('set %s.%s.%s = %.2f with gbc \'%s\''%(place_code,finish_code,geo_code, att_val, gbc))
                    
                    #===========================================================
                    # value checks
                    #===========================================================
                    if self.db_f:
                        if not isinstance(att_val, float):
                            raise IOError
                        if pd.isnull(att_val): 
                            raise IOError
                        if att_val < 0: 
                            raise IOError
                        if att_val is None:
                            raise IOError

                        
                    #===========================================================
                    # set the value
                    #===========================================================
                    dxcol.loc[geo_code, (place_code, finish_code)] = att_val
                    
                    #row[finish_code] = att_val #update the ser
                    #logger.debug('set \'%s\' as \'%s\''%(att_name, att_val))
                  
        #=======================================================================
        # special attribute setting 
        #=======================================================================
        'need this as an attribute for reporting'
        B_f_height = dxcol.loc['height', ('B', 'f')]
        #===============================================================
        # POST
        #===============================================================
        #logger.debug('built house_geo_dxcol %s'%str(dxcol.shape))
        
        self.handle_upd('geo_dxcol', dxcol, weakref.proxy(self), call_func = 'set_geo_dxcol')
        self.handle_upd('B_f_height', B_f_height, weakref.proxy(self), call_func = 'set_geo_dxcol')
                        
        return True

    
    def calc_secondary_geo(self,  #aset the default geometry for this attribute
                           place_code, finish_code, geo_code,
                           dxcol = None): 
        
        logger = self.logger.getChild('get_default_geo')
        
        #=======================================================================
        # get primary geometrty from frame
        #=======================================================================
        if dxcol is None: dxcol = self.geo_dxcol
        
        area = dxcol.loc['area',(place_code, finish_code)]
        height = dxcol.loc['height',(place_code, finish_code)]
        
        #=======================================================================
        # calculate the geometris
        #=======================================================================
    
        if geo_code == 'inta':
            per = dxcol.loc['per',(place_code, finish_code)]
            
            att_value = float(area + height * per)
        
        elif geo_code == 'per':
            
            per = 4*math.sqrt(area)
            att_value = float(per)
            
            
        else: raise IOError
        
        logger.debug(" for \'%s\' found %.2f"%(geo_code, att_value))
        
        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            for v in [area, height, per, att_value]:
                if not isinstance(v, float): raise IOError
                if pd.isnull(v): 
                    raise IOError
                
                if not v >= 0: raise IOError
        
        
        return att_value
    

    def get_geo_from_other(self, df_raw, attn_search): #set the garage area 
        """
        we need this here to replicate the scaling done by the legacy curves on teh garage dmg_feats
        
        assuming column 1 is the cross refereence data
        """
        logger = self.logger.getChild('get_geo_from_other')


        #=======================================================================
        # find the cross reference row
        #=======================================================================
        cross_attn = df_raw.columns[0]
        cross_v = getattr(self, cross_attn)  #get our value for this
        
        boolidx = df_raw.iloc[:,0] == cross_v #locate our cross reference
        
        
        #=======================================================================
        # find the search column
        #=======================================================================
        boolcol = df_raw.columns == attn_search
        
        value_fnd = df_raw.loc[boolidx, boolcol].iloc[0,0] #just take the first
        
        if self.db_f:
            if not boolidx.sum() == 1:
                raise IOError
            if not boolidx.sum() == 1:
                raise IOError
        

        return value_fnd
 
 
    def run_hse(self, *vargs, **kwargs):
        'TODO: compile the total dfunc and use that instead?'
        logger = self.logger.getChild('run_hse')
        self.run_cnt += 1
        
        #=======================================================================
        # precheck
        #=======================================================================
        """todo: check that floods are increasing
        if self.db_f:
            if self.last_floodo is None:
                pass"""

        
        #=======================================================================
        # basement egrade reset check
        #=======================================================================
        if self.model.bsmt_egrd_code == 'plpm':
            if self.run_cnt ==1:
                cond = self.set_bsmt_egrd()
            
            elif not self.gpwr_f == self.floodo.gpwr_f:
                cond = self.set_bsmt_egrd()
                
            else:
                cond = 'nochng'
                logger.debug('no change in gpwr_f. keeping bsmt egrd = %s'%self.bsmt_egrd)
        else:
            cond = 'no_plpm'
                
        #===============================================================
        # write the beg histor y
        #===============================================================
        if not self.model.beg_hist_df is None:                
            self.model.beg_hist_df.loc[self.dfloc, (self.floodo.ari, 'bsmt_egrd')] = self.bsmt_egrd
            self.model.beg_hist_df.loc[self.dfloc, (self.floodo.ari, 'cond')] = cond
            

        
        logger.debug('returning get_dmgs_wsls  \n')
        
        results = self.get_dmgs_wsls(*vargs, **kwargs)
        

                
                
        
        self.floodo = None #clear this
        

          
        return results

    def get_dmgs_wsls(self,  #get damage at this depth from each Dfunc
                    wsl, 
                    dmg_rat_f = False, #flat to include damage ratios in the outputs
                    #res_ser=None, 
                    #dmg_type_list=None,
                    ): 
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        res_ser: shortcut so that damage are added to this series
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('get_dmgs_wsls')
        
        #=======================================================================
        # calculate damages  by type
        #=======================================================================
        id_str = self.get_id()
                
        #=======================================================================
        # fast calc
        #=======================================================================
        if not dmg_rat_f:
            dmg_ser = pd.Series(name = self.name, index = self.dfunc_d.keys())
            
            """
            logger.debug('\'%s\' at wsl= %.4f anchor_el = %.4f for %i dfuncs bsmt_egrd \'%s\'\n'
                     %(id_str, wsl, self.anchor_el, len(dmg_ser), self.bsmt_egrd))"""

            for dmg_type, dfunc in self.kids_d.iteritems():
    
                logger.debug('getting damages for \'%s\' \n'%dmg_type)
        
                #get the damge
                _, dmg_ser[dmg_type], _ = dfunc.run_dfunc(wsl)
                
                dfunc.get_results() #store these outputs if told
                

        #=======================================================================
        # full calc
        #=======================================================================
        else:
            raise IOError #check this
            dmg_df = pd.DataFrame(index = self.dfunc_d.keys(), columns = ['depth', 'dmg', 'dmg_raw'])
            dmg_ser = pd.Series()
            
            logger.debug('\'%s\' at wsl= %.4f anchor_el = %.4f for %i dfuncs bsmt_egrd \'%s\'\n'
                     %(id_str, wsl, self.anchor_el, len(dmg_df), self.bsmt_egrd))
            
            for indx, row in dmg_df.iterrows():
                dfunc = self.kids_d[indx]
                
                row['depth'], row['dmg'], row['dmg_raw'] = dfunc.run_dfunc(wsl)
                
                dfunc.get_results() #store these outputs if told
                
                #enter into series
                dmg_ser[indx] = row['dmg']
                dmg_ser['%s_rat'%indx] = row['dmg_raw']
                
        #=======================================================================
        # wrap up
        #=======================================================================
        logger.debug('at %s finished with %i dfuncs queried and res_ser: \n %s \n'
                     %(self.model.tstep_o.name, len(self.kids_d), dmg_ser.values.tolist()))

        
        if self.db_f:
            #check dfeat validity
            if 'BS' in self.kids_d.keys():
                dfunc = self.kids_d['BS']
                d = dfunc.kids_d
                
                for k, v in d.iteritems():
                    if not v.hse_type == self.hse_type:
                        logger.error('%s.%s hse_type \'%s\' does not match mine \'%s\''
                                     %(v.parent.name, v.name, v.hse_type, self.hse_type))
                        raise IOError
            

        return dmg_ser
    

                
    def raise_total_dfunc(self, #compile the total dd_df and raise it as a child
                          dmg_codes = None, place_codes = None): 
        """ this is mostly used for debugging and comparing of curves form differnet methods
        
        #=======================================================================
        # todo
        #=======================================================================
        allow totaling by 
        possible performance improvement;
            compile the total for all objects, then have Flood.get_dmg_set only run the totals
            
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('raise_total_dfunc')
        tot_name = self.get_tot_name(dmg_codes)
        
        if dmg_codes is None:  dmg_codes = self.model.dmg_codes
        if place_codes is None: place_codes = self.model.place_codes
        
        #=======================================================================
        # get the metadata for the child
        #=======================================================================
        df_raw = self.session.pars_df_d['dfunc'] #start with the raw tab data
        
        #search by placecode
        boolidx1 = df_raw['place_code'] == 'total' #identify all the entries except total
        
        #search by dmg_code where all strings in the list are a match
        boolidx2 = hp.pd.search_str_fr_list(df_raw['dmg_code'], dmg_codes, all_any='any') #find 
        
        if boolidx2.sum() <1:
            logger.warning('unable to find a match in the dfunc tab for %s. using default'%tot_name)
            boolidx2 = pd.Series(index = boolidx2.index, dtype = np.bool) #all true 


        'todo: add some logic for only finding one of the damage codes'
        
        #get this slice
        boolidx = np.logical_and(boolidx1, boolidx2)
        
        if not boolidx.sum() == 1: 
            logger.error('childmeta search boolidx.sum() = %i'%boolidx.sum())
            raise IOError
        
        att_ser = df_raw[boolidx].iloc[0]

        'need ot add the name here as were not using the childname override'
                
        logger.debug('for place_code: \'total\' and dmg_code: \'%s\' found child meta from dfunc_df'%(dmg_codes))
        #=======================================================================
        # raise the child
        #=======================================================================
        #set the name        
        child = self.spawn_child(att_ser = att_ser, childname = tot_name)
        
        #=======================================================================
        # #do custom edits for total
        #=======================================================================
        child.anchor_el = self.anchor_el
        
        #set the dd_ar
        dd_df = self.get_total_dd_df(dmg_codes, place_codes)
        depths = dd_df['depth'].values - child.anchor_el #convert back to no datum
        
        child.dd_ar     = np.array([depths, dd_df['damage'].values])
        
        #add this to thedictionary
        self.kids_d[child.name] = child
        
        logger.debug('copied and edited a child for %s'%child.name)
        
        return child
    
    def get_total_dd_df(self, dmg_codes, place_codes): #get the total dd_df (across all dmg_types)
        logger = self.logger.getChild('get_total_dd_df')
        
        #=======================================================================
        # compile al lthe depth_damage entries
        #=======================================================================
        df_full = pd.DataFrame(columns = ['depth', 'damage_cum', 'source'])
        
        # loop through and fill the df
        cnt = 0
        for datoname, dato in self.kids_d.iteritems():
            if not dato.dmg_code in dmg_codes: continue #skip this one
            if not dato.place_code in place_codes: continue
            
            cnt+=1
            #===================================================================
            # get the adjusted dd
            #===================================================================
            df_dato = pd.DataFrame() #blank frame
            
            df_dato['depth'] =  dato.dd_ar[0]+ dato.anchor_el  #adjust the dd to the datum
            df_dato['damage_cum'] = dato.dd_ar[1]
            """the native format of the dmg_ar is cumulative damages
            to sum these, we need to back compute to incremental
            """
            df_dato['damage_inc'] = hp.pd.get_incremental(df_dato['damage_cum'], logger=logger)
            df_dato['source'] = datoname
            
            #append these to the full
            df_full = df_full.append(df_dato, ignore_index=True)
            
        logger.debug('compiled all dd entries %s from %i dfuncs with dmg_clodes: %s'
                     %(str(df_full.shape), cnt, dmg_codes))
        
        df_full = df_full.sort_values('depth').reset_index(drop=True)
        
        #=======================================================================
        # harmonize this into a dd_ar
        #=======================================================================
        #get depths
        
        depths_list = df_full['depth'].sort_values().unique().tolist()
        
        #get starter frame
        dd_df = pd.DataFrame(columns = ['depth', 'damage'])
        dd_df['depth'] = depths_list #add in the depths
        
        for index, row in dd_df.iterrows(): #sort through and sum by depth
            
            boolidx = df_full['depth'] <= row['depth'] #identify all those entries in the full 
            
            row['damage'] = df_full.loc[boolidx, 'damage_inc'].sum() #add these as the sum
            
            dd_df.iloc[index,:] = row #update the master
            
        logger.debug('harmonized and compiled dd_df %s'%str(dd_df.shape))
        
        self.dd_df = dd_df
        
        return dd_df
      
    def get_tot_name(self, dmg_codes): #return the equilvanet tot name
        'not sure whats going on here'
        new_str = 'total_'
        for dmg_code in dmg_codes: new_str = new_str + dmg_code
        
        return new_str
    
    def calc_statres_hse(self): #calculate statistics for the house (outside of a run)
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        this is always called with mypost_update() executing each command in self.post_upd_func_s()
        
        mypost_update() is called:
            init_dyno()    #first call before setting the OG values
            session.post_update() #called at the end of all the update loops
            
        
        """
        logger = self.logger.getChild('calc_statres_hse')
        s = self.session.outpars_d[self.__class__.__name__]
        #=======================================================================
        # BS_ints
        #=======================================================================
        if 'BS_ints' in s:
            'I dont like this as it requires updating the child as well'
            """rfda curves also have this stat
            if self.dfunc_type == 'dfeats':"""
            
            #updat eht ekid
            if not self.kids_d['BS'].calc_intg_stat(): raise IOError
            
            self.BS_ints = self.kids_d['BS'].intg_stat
            
            """this is handled by set_og_vals()
            if self.session.state == 'init':
                self.reset_d['BS_ints'] = self.BS_ints"""
                
            logger.debug('set BS_ints as %.4f'%self.BS_ints)
                
        if 'vuln_el' in s:
            self.set_vuln_el()
            
        
        if 'max_dmg' in s:
            self.max_dmg = self.get_max_dmg()
            self.parent.childmeta_df.loc[self.dfloc, 'max_dmg'] = self.max_dmg #set into the binv_df

        return True
    

    
    def set_vuln_el(self): #calcualte the minimum vulnerability elevation
        """
        #=======================================================================
        # CALLS
        #=======================================================================

        TODO: consider including some logic for bsmt_egrade and spill type
        """
        #=======================================================================
        # check frozen and dependenceis
        #=======================================================================
        logger = self.logger.getChild('set_vuln_el')
        
        """this is a stat, not a dynamic par
        if self.is_frozen('vuln_el', logger=logger): return True"""
        
        vuln_el = 99999 #starter value
        
        for dmg_type, dfunc in self.kids_d.iteritems():
            vuln_el = min(dfunc.anchor_el, vuln_el) #update with new minimum
            

        logger.debug('set vuln_el = %.2f from %i dfuncs'%(vuln_el, len(self.kids_d)))
        
        self.vuln_el = vuln_el
            
        return True
 
    def get_max_dmg(self): #calculate the maximum damage for this house
        #logger = self.logger.getChild('get_max_dmg')
        
        
        ser = pd.Series(index = self.kids_d.keys())
        
        #=======================================================================
        # collect from each dfunc
        #=======================================================================
        for dmg_type, dfunc in self.kids_d.iteritems():
            ser[dmg_type] = dfunc.dd_ar[1].max()
            
        return ser.sum()

        
        
    def plot_dd_ars(self,   #plot each dfunc on a single axis
                    datum='house', place_codes = None, dmg_codes = None, plot_tot = False, 
                    annot=True, wtf=None, title=None, legon=False,
                    ax=None, 
                    transparent = True, #flag to indicate whether the figure should have a transparent background
                    **kwargs):
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        datum: code to indicate what datum to plot the depth series of each dd_ar
            None: raw depths (all start at zero)
            real: depths relative to the project datum
            house: depths relative to the hse_obj anchor (generally Main = 0)
            
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('plot_dd_ars')
        if wtf==None:           wtf= self.session._write_figs
        if dmg_codes is None:   dmg_codes = self.model.dmg_codes
        if place_codes is None: place_codes = self.model.place_codes
            
        if title is None:
            title = 'plot_dd_ars on %s for %s and %s'%(self.name, dmg_codes, place_codes)
            if plot_tot: title = title + 'and T'
        
        'this should let the first plotter setup the axis '

        logger.debug('for \n    dmg_codes: %s \n    place_codes: %s'%(dmg_codes, place_codes))
        #=======================================================================
        # plot the dfuncs that fit the criteria
        #=======================================================================
        dfunc_nl = [] #list of dfunc names fitting criteria
        for datoname, dato in self.dfunc_d.iteritems():
            if not dato.dmg_code in dmg_codes: continue
            if not dato.place_code in place_codes: continue
            ax = dato.plot_dd_ar(ax=ax, datum = datum, wtf=False, title = title, **kwargs)
            
            dfunc_nl.append(dato.name)
            
        #=======================================================================
        # add the total plot
        #=======================================================================
        if plot_tot:
            #get the dato
            tot_name = self.get_tot_name(dmg_codes)
            if not tot_name in self.kids_d.keys(): #build it
                'name searches should still work'
                
                tot_dato = self.raise_total_dfunc(dmg_codes, place_codes)
            else:
                tot_dato = self.kids_d[tot_name]
            
            #plot the dato
            ax = tot_dato.plot_dd_ar(ax=ax, datum = datum, wtf=False, title = title, **kwargs)
            
        #=======================================================================
        # add annotation
        #=======================================================================
        if not annot is None:
            if annot:
                """WARNING: not all attributes are generated for the differnt dfunc types
                """               
                B_f_height = float(self.geo_dxcol.loc['height',('B','f')]) #pull from frame
                
                
                annot_str = 'hse_type = %s\n'%self.hse_type +\
                            '    gis_area = %.2f m2\n'%self.gis_area +\
                            '    anchor_el = %.2f \n'%self.anchor_el +\
                            '    dem_el = %.2f\n'%self.dem_el +\
                            '    B_f_height = %.2f\n'%B_f_height +\
                            '    bsmt_egrd = %s\n'%self.bsmt_egrd +\
                            '    AYOC = %i\n \n'%self.ayoc
                            
                #add info for each dfunc
                
                for dname in dfunc_nl:
                    dfunc = self.dfunc_d[dname]
                    annot_str = annot_str + annot_builder(dfunc)
    
            else: annot_str = annot
            #=======================================================================
            # Add text string 'annot' to lower left of plot
            #=======================================================================
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            
            x_text = xmin + (xmax - xmin)*.7 # 1/10 to the right of the left axis
            y_text = ymin + (ymax - ymin)*.01 #1/10 above the bottom axis
            anno_obj = ax.text(x_text, y_text, annot_str)
                
        #=======================================================================
        # save figure
        #=======================================================================
        if wtf: 
            """
            self.outpath
            """
            fig = ax.figure
            flag = hp.plot.save_fig(self, fig, dpi = self.dpi, legon=legon, transparent = transparent)
            if not flag: raise IOError 
            
        logger.debug('finished as %s'%title)
        
        return ax
    
    def write_all_dd_dfs(self, tailpath = None): #write all tehchildrens dd_dfs
        
        if tailpath is None: tailpath = os.path.join(self.outpath, self.name)
        
        if not os.path.exists(tailpath): os.makedirs(tailpath)
        
        for gid, childo in self.kids_d.iteritems():
            
            if not childo.dfunc_type == 'dfeats': continue #skip this one\
            
            filename = os.path.join(tailpath, childo.name + ' dd_df.csv')
            
            childo.recompile_dd_df(outpath = filename)
            
class Dfunc(
            hp.plot.Plot_o,
            hp.dyno.Dyno_wrap,
            hp.sim.Sim_o, #damage function of a speciic type. to be attached to a house
            hp.oop.Parent,
            hp.oop.Child): 
    '''
    #===========================================================================
    # architecture
    #===========================================================================
    rfda per house predicts 4 damage types (MS, MC, BS, BC)\
    
    do we want these damage types comtained under one Dfunc class object? or separate?
    
        lets keep them separate. any combining can be handled in the House class
        
    #===========================================================================
    # main vars
    #===========================================================================
    dd_ar:    main damage array (np.array([depth_list, dmg_list])) data
        using np.array for efficiency
        this is compiled based on dfunc_type see:
            legacy: get_ddar_rfda 
            abmri: get_ddar_dfeats (this requires some intermittent steps)

    '''
    #===========================================================================
    # program pars
    #===========================================================================
    
    """post_cmd_str_l      = ['build_dfunc']
    
    # object handling overrides
    load_data_f         = True
    raise_kids_f        = True #called explicilly in load_data()"""
    #raise_in_spawn_f    = True #load all the children before moving on to the next sibling
    db_f                = False
    
    """
    #===========================================================================
    # #shadow kids
    #===========================================================================
    see note under Dfeats
    """
    reset_shdw_kids_f = False #flag to install the shadow_kids_d during reset 
    shdw_kids_d   = None #placeholder for the shadow kids
    
    kid_cnt         = 0 #number of kids you have
    
    #===========================================================================
    # passed pars from user
    #===========================================================================
    place_code      = None 
    dmg_code        = None 
    dfunc_type      =''
    bsmt_egrd_code = ''
    anchor_ht_code = None 
    geo_build_code = None 
    rat_attn       = '*none' #attribute name  to scale by for relative damage functions

    #===========================================================================
    # calculation pars
    #===========================================================================
    dd_ar           = None #2d array of depth (dd_ar[0])vs total damage (dd_ar[1]) values
    dmg_type        = None  #type of damage predicted by this function
    
    anchor_el       = 0.0 #height from project datum to the start of the dd_ar (depth = 0)

    
    #headers to keep in the dyn_dmg_df
    dd_df_cols = ['name', 'base_price', 'depth', 'calc_price']
    
    depth_allow_max = 10 #maximum depth to allow without raising an error with dg_f = True. 
    '10m seems reasonable for a 12ft basement and 1000 yr flood' 
    
    tag             = None #type of dfeats curve
    

    
    intg_stat       = None #placeholder for this stat
     

    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Dfunc')
        logger.debug('start _init_')
        #=======================================================================
        # update program handlers
        #=======================================================================
        self.inherit_parent_ans=set(['mind', 'model'])
        
        
        super(Dfunc, self).__init__(*vars, **kwargs) #initilzie teh baseclass   
        
        #=======================================================================
        #common setup
        #=======================================================================

        if self.sib_cnt == 0:
            logger.debug('sib_cnt = 0. setting complex atts')
            self.kid_class      = Dmg_feat #mannually pass/attach this
            self.hse_o          = self.parent 
            'this should be a proxy'
        
                    
        #=======================================================================
        # #unique
        #=======================================================================
        'for now, only using this on BS curves'
        if self.name == 'BS':
            self.post_upd_func_s = set([self.calc_statres_dfunc])
        
        #misc
        self.label = self.name + ' (%s) (%s)'%(self.dfunc_type, self.units)
        
        """ keep as string
        #relative curves
        if self.rat_attn == '*none':
            self.rat_attn = None"""
            
        
            
        if not self.place_code == 'total':
            #loaders
            logger.debug('build_dfunc \n')
            self.build_dfunc()
            logger.debug('init_dyno \n')
            self.init_dyno()
        
        #=======================================================================
        # checks
        #=======================================================================
        if self.db_f: 
            logger.debug("checking myself \n")
            self.check_dfunc()
            
            if hasattr(self, 'kids_sd'):
                raise IOError

        self.logger.debug('finished _init_ as \'%s\' \n'%(self.name))
        
        return
    
    def check_dfunc(self):
        logger = self.logger.getChild('check_dfunc')
        logger.debug('checking')
        
        """not using the dyno_d any more
        if not self.gid in self.session.dyno_d.keys():
            raise IOError"""
            
        if not self.place_code == 'B':
            pass
            
        if (self.place_code == 'G') & (self.dfunc_type == 'rfda'):
            raise IOError
        
        
        if self.dfunc_type == 'rfda':
            if not self.rat_attn == 'self.parent.gis_area':
                logger.error('for RFDA, expected \'gis_area\' for rat_attn')
                raise IOError
            
        elif self.dfunc_type == 'dfeats':
            if not self.rat_attn =='*none':
                logger.error('expected \'*none\' for rat_attn on dfeats')
                raise IOError
        
        if not self.rat_attn =='*none':
            
            try:
                _ = eval(self.rat_attn)
            except:
                logger.error('failed to execute passed \'%s\''%self.rat_attn)
                raise IOError
            
        #=======================================================================
        # total checks
        #=======================================================================
        if self.place_code == 'total':
            if self.anchor_ht_code == '*hse':
                raise IOError #hse not allowed for total curve
            
        return

    def build_dfunc(self):  #execute all the commands to build the dfunc from scratch
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        _init_
        handles
        
        """
        'todo: move these commands elsewhere'
        id_str = self.get_id()
        logger = self.logger.getChild('build_dfunc(%s)'%id_str)
    
        """leaving this to more specific functions
        #=======================================================================
        # dependency check
        #=======================================================================
        if not self.session.state == 'init':
            dep_p = [([self.parent],['set_geo_dxcol()'] )] #dependency paring
            if self.deps_is_dated(dep_p, method = 'force', caller = 'build_dfunc'):
                raise IOError #because we are forcing this should alwys return FALSE"""
            
        'need to clear this so the children will update'
        self.del_upd_cmd(cmd_str = 'build_dfunc()')
        self.del_upd_cmd(cmd_str = 'recompile_ddar()')
                

        #=======================================================================
        # custom loader funcs
        #=======================================================================
        logger.debug('set_dfunc_anchor() \n')
        res1 = self.set_dfunc_anchor() #calculate my anchor elevation
        
        logger.debug('build_dd_ar() \n')
        res2 = self.build_dd_ar()

        """ moved this into build_dd_ar 
        self.constrain_dd_ar()"""
        

        
        #logger.debug('\n')
        
        if self.session.state == 'init':
            if self.dfunc_type == 'dfeats':
                'add this here so the children have a chance to fill it out during load'
                self.reset_d['childmeta_df'] = self.childmeta_df.copy()
        else:
            pass
            """some comands (constrain_dd_ar) we want to leave in the que
            self.halt_update()"""
            
            """cleared this at the beginning
            if len(self.upd_cmd_od) > 0:
                self.del_upd_cmd(cmd_str = 'recompile_ddar()')"""

        #=======================================================================
        # post checks
        #=======================================================================

        if self.db_f:
            if len(self.upd_cmd_od) > 0:
                logger.warning('still have updates queud: \n %s'%self.upd_cmd_od.keys())

        logger.debug('finished \n')
        return True #never want to re-que this
        

               
            
    def build_dd_ar(self): #buidl the damage curve from codes passed on the 'dfunc' tab
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        build_dfunc
        
        (this could be used by some handles...but not a great case)

        """
        #=======================================================================
        # dependencies
        #=======================================================================
        """leaving these to the type specific dd_ar builders
        
        #state = self.session.state != 'update'
        _ = self.depend_outdated(search_key_l = ['set_geo_dxcol()'], #see if the parent has these
                             force_upd = True, #force the parent to update if found
                             halt = False) #clear all of my updates (except this func)"""
                             

        
        
        logger = self.logger.getChild('build_dd_ar')
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if self.dfunc_type == 'rfda':
                """switched to hse_geo tab
                if not self.geo_build_code == 'defaults':
                    logger.error('dfunc_type=rfda only uses gis_area. therefore geo_build_code must = defaults') 
                    raise IOError"""
                
                if not self.anchor_ht_code == '*rfda':
                    logger.debug('dfunc_type=rfda got  anchor_ht_code != rfda_par.')
                    'as were keeping the contents rfda, need to allow cross anchoring types'
                    
            elif self.dfunc_type == 'dfeats':                
                if self.dmg_code == 'C':
                    logger.error('Contents damage not setup for dfunc_type = dfeats')
                    raise IOError
                
            elif self.dfunc_type == 'depdmg':
                pass #not impoxing any restrictions?
                
                
            else: raise IOError
        #=======================================================================
        # get raw curve data
        #=======================================================================
        if (self.place_code=='B') and (not self.parent.bsmt_f):
            logger.debug('this building doesnt have a basement. dummy Dfunc')
            
            self.dummy_f = True
            dd_ar = np.array()
                        
        elif self.dfunc_type == 'rfda':#leagacy
            logger.debug('dfunc_type = rfda. building')
            
            dd_ar = self.get_ddar_rfda() #build the dfunc from this house type
                
            self.kids_d =  wdict()#empty placeholder
            
        elif self.dfunc_type == 'dfeats':
            logger.debug('dfunc_type = dfeats. raising children')
       
            #grow all the damage features
            self.raise_dfeats() 
            
            #compile the damage array
            dd_ar = self.get_ddar_dfeats()
            
        elif self.dfunc_type == 'depdmg':
            logger.debug('dfunc_type = depdmg. building array')
            
            dd_ar = self.get_ddar_depdmg()
            
            self.kids_d =  wdict()#empty placeholder
            
        else: raise IOError
        
        #=======================================================================
        # wrap up
        #=======================================================================
        'constrain will set another copy onto this'
        self.dd_ar = dd_ar
        #=======================================================================
        # this seems overly complicated...
        # if not self.anchor_el is None: 
        #     """ set anchor is called by load_data after this"""
        #     logger.debug('for session state \'%s\' running constrain_dd_ar'%(self.session.state))
        #=======================================================================
        #=======================================================================
        # constrain_dd_ar
        #=======================================================================
        """even thourgh we may receive multiple ques, this should be called everytime. 
            build_dfunc() will clear the que"""

        res = self.constrain_dd_ar()
        
        """cosntrain_dd_ar will execute this
        self.handle_upd('dd_ar', dd_ar, proxy(self), call_func = 'build_dd_ar')
        'this will que constrain_dd_ar for non init runs'"""
        
        #=======================================================================
        # get stats
        #=======================================================================
        


        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            if 'build_dd_ar()' in self.upd_cmd_od.keys(): raise IOError
            if res:
                if 'constrain_dd_ar()' in self.upd_cmd_od.keys(): raise IOError
            
            """
            see note. not a strong case for queuing this command directly (with handles)
            'because we are not using the update handler, just calling this'
            self.del_upd_cmd(cmd_str = 'build_dd_ar') #delete yourself from the update command list"""
            
        
        logger.debug('finished for dfunc_type \'%s\' and dd_ar %s \n'%(self.dfunc_type,str(self.dd_ar.shape)))
        
        return True
    
    def get_ddar_rfda(self): #build a specific curve from rfda classic
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        raw_dcurve_df: raw df from the standard rfda damage curve file
            Ive left the sorting/cleaning to here'
            may be slightly more efficient (although more confusing) to clean this in the session
            
        #=======================================================================
        # OUTPUTS
        #=======================================================================
        dd_ar: depth damage per m2 
            NOTE: this is different than the dd_ar for the dyn_ddars
            reasoning for this is I want the parent calls to all be in the run loop
                (rather than multiplying the rfda $/m2 by the parents m2 during _init_)
                
        #=======================================================================
        # TODO:
        #=======================================================================
        consider precompiling all of these and making pulls to a shadow set instead
            
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('get_ddar_rfda')
        
        self.dfunc_type     = 'rfda' #set the dfunc type
        dmg_type            = self.dmg_type
        hse_type            = self.hse_o.hse_type
        raw_dcurve_df       = self.model.kids_d['rfda_curve'].data
        'need this goofy reference as the fdmg_o has not fully loaded'
        
        if self.db_f:
            if not hp.pd.isdf(raw_dcurve_df):
                raise IOError
            
            """for new houses we should run this mid session
            if not self.session.state == 'init': 
                raise IOError"""
        
        logger.debug('for dmg_type = %s, hse_type = %s and raw_dcurve_df %s'%(dmg_type, hse_type, str(raw_dcurve_df.shape)))
        #=======================================================================
        # prechecks
        #=======================================================================
        #if dmg_type is None: raise IOError
        #=======================================================================
        # get the raw data
        #=======================================================================
        #clean the data
        df1 =   raw_dcurve_df.dropna(how = 'all', axis='index') #drop rows where ALL values ar na
        df2 =   df1.dropna(how = 'any', axis='columns') #drop columns where ANY values are na
        
        #find the rows for this hse_type
        boolidx = df2.iloc[:,0].astype(str).str.contains(hse_type) #
        df3 = df2[boolidx]
        
        #narrow down to this dmg_type
        boolidx = df3.iloc[:,-1].astype(str).str.contains(dmg_type)
        df4 = df3[boolidx]
        
        dcurve_raw_list = df4.iloc[0,:].values.tolist() #where both are true
        
        #checks
        if len(dcurve_raw_list) == 0: raise IOError
        #=======================================================================
        # for this row, extract teh damage curve
        #=======================================================================
        depth_list = []
        dmg_list = []
        for index, entry in enumerate(dcurve_raw_list): #loop through each entry
            'the syntax of these curves is very strange'
            #===================================================================
            # logic for non depth/damage entries
            #===================================================================
            if index <=1:                   continue #skip the first 2
            if not hp.basic.isnum(entry):   continue #skip non number
            #===================================================================
            # logic to sort depth from damage based on even/odd
            #===================================================================
            if index%2 == 0:    depth_list.append(float(entry))
            else:               dmg_list.append(float(entry))
                
        
        """ thsi even/odd index selectio may not work for non house type damage curves        
        """

        #=======================================================================
        # Build array
        #=======================================================================
        dd_ar = np.sort(np.array([depth_list, dmg_list]), axis=1)

        
        """ moved this to make parent reference more accessible
        dd_ar[1] = dd_ar1[1] * self.parent.gis_area"""
        
        #checks
        if self.db_f:
            logger.debug('got \n depth_list: %s \n dmg_list: %s'%(depth_list, dmg_list))
            
            if not len(depth_list) == len(dmg_list): #check length
                self.logger.error('depth/fdmg lists do not match')
                """ these should both be 11
                [0.0, 0.1, 0.30000000000000004, 0.6000000000000001, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
                """
                raise IOError
            
            if not len(depth_list) == 11: 
                raise IOError
            
            if not dd_ar.shape[0] == 2:
                self.logger.warning('got unexpected shape on damage array: %s'%str(dd_ar.shape))
                'getting 3.208 at the end of the depth list somehow'
                raise IOError
        
        #=======================================================================
        # closeout
        #=======================================================================
        #self.dd_ar = dd_ar
        
        
        logger.debug('built damage array from rfda for hse_type \'%s\' and dmg_type \'%s\' as %s'
                          %(hse_type, dmg_type, str(dd_ar.shape)))

        return dd_ar
    
    def get_ddar_depdmg(self): #build the dd_ar from standard format depth damage tables
        logger = self.logger.getChild('get_ddar_depdmg')
        
        #=======================================================================
        # get your data from the session
        #=======================================================================
        df = self.model.dfunc_raw_d[self.name]
        
        dd_ar = np.sort(df.values, axis=1)
        
        logger.debug('build dd_ar from passed file for with %s'%(str(dd_ar.shape)))
        
        return dd_ar
        
        
    
    def get_ddar_dfeats(self): #build the dd_ar from the dmg_feats
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        build_dd_ar (for dfeats)
        recompile_ddar (called by handles)
        
        never directly called by handles
        #=======================================================================
        # OUTPUTS
        #=======================================================================
        dd_ar: depth/damage (total)
            NOTE: This is different then the dd_ar for rfda curves
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('get_ddar_dfeats')
        
        dd_df = self.childmeta_df #ge tthe dynamic depth damage frame 
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not hp.pd.isdf(dd_df): 
                raise IOError
            
            if np.any(pd.isnull(dd_df['calc_price'])):
                logger.error('got %i null entries on teh dd_df'%pd.isnull(dd_df['calc_price']).sum())
                raise IOError
            """
            hp.pd.v(dd_df.drop(columns=['price_calc_str', 'desc', 'unit']))
            dd_df.columns
            """
            
            for k, v in self.kids_d.iteritems():
                if not v.hse_type == self.parent.hse_type:
                    logger.error('dfeat \'%s\' hse_type \'%s\' does not match \'%s\''
                                 %(k, v.hse_type, self.parent.hse_type))
                    raise IOError

            #=======================================================================
            # dependencies/frozen
            #=======================================================================
            'only using this as a check'
            #if not self.session.state == 'init':
            dep_p = [([self.parent], ['set_geo_dxcol()']),\
                     (self.kids_d.values(), ['eval_price_calc_str()', 'set_new_depth()'])]#,\
            
                    #the dfeats que this'
                    #([self],['recompile_ddar()'])]
            
            if self.deps_is_dated(dep_p, caller = 'get_ddar_dfeats'):
                'this func is never called directly by the handles'
                raise IOError #because we are forcing this should alwys return FALSE
            
            #===================================================================
            # frame checks
            #===================================================================
            if dd_df['depth'].min() < 0: 
                raise IOError

        #=======================================================================
        # setup
        #=======================================================================
        logger.debug('compiling dd_ar from dd_df %s'%str(dd_df.shape))
        #get list of depths
        depth_list = dd_df['depth'].astype(np.float).sort_values().unique().tolist()
         
        #=======================================================================
        # calc dmg_list for these depths
        #=======================================================================
        dmg_list = []
        for depth in depth_list:
            #find all the depths less then this
            boolidx = dd_df['depth'] <= depth
            dd_df_slice = dd_df[boolidx] #get this slice
            
            #check this
            #if len(dd_df_slice) <1: raise IOError
            
            #calc the damage for this slice
            dmg = dd_df_slice['calc_price'].sum()
            
            """these are calculated by each Dmg_feat.get_calc_price
             and entered into the dyn_dmg_df"""

            dmg_list.append(dmg)

        #=======================================================================
        # constrain the depths
        #=======================================================================
        """ moved this
        dd_ar = self.constrain_dd_ar(depth_list, dmg_list)"""
         
        #=======================================================================
        # closeout
        #=======================================================================
        
        dd_ar = np.sort(np.array([depth_list, dmg_list]), axis=1)
        
        """moved this to build_dd_ar() and recompile_dd_ar()
        self.handle_upd('dd_ar', dd_ar, proxy(self), call_func = 'get_ddar_dfeats')"""
        'this will add constrain_dd_ar to the queue, but generally this should remove itself shortly'
                
        logger.debug('built dd_ar %s'%str(dd_ar.shape))
        
        return dd_ar


    def set_childmetadf(self): #get teh childmetadata for the dyn dmg_feats
        logger = self.logger.getChild('set_childmetadf')
        
        #=======================================================================
        # setup
        #=======================================================================
        #pull the data for this curve from globals
        dyn_dc_dato = self.model.kids_d['dfeat_tbl']
                
        if not self.hse_o.hse_type in dyn_dc_dato.df_dict.keys():
            logger.error('my house type (%s) isnt in teh dfeat_tbl'%self.hse_o.hse_type)
            raise IOError
        
        df_raw = dyn_dc_dato.df_dict[self.hse_o.hse_type] #for just this house type
        
        logger.debug('got data from file for %s with %s'%(self.hse_o.hse_type, str(df_raw.shape)))
        
        #=======================================================================
        # clean the raw
        #=======================================================================
        df = df_raw.drop(columns = ['desc', 'unit'])
        """
        hp.pd.v(df)
        """
        
        #get slice for just this floor
        boolidx = df['place_code'] == self.place_code
        df_slice = df[boolidx].reset_index(drop=True)
        
        """being done elsewhere
        df_slice['depth'] = df_slice['depth_dflt'] #duplicate this column
        
        df_slice.loc[:,'calc_price'] = np.nan #add this as a blank column"""
        
        self.childmeta_df = df_slice
        
        #=======================================================================
        # attach base geometry
        #=======================================================================
        'todo: add check that all these are the same'

        #=======================================================================
        # close out/wrap up
        #=======================================================================
        """ not ready to store this yet... wait till teh children load and insert their initial values
        placed in build_dfunc() 
        self.reset_d['childmeta_df'] = self.childmeta_df.copy() #add this for reseting"""

        logger.debug('finisehd with %s'%str(df_slice.shape))      
        
        if self.db_f:
            if not 'calc_price' in self.childmeta_df.columns.tolist():
                raise IOError
    
        return 
    
    def raise_dfeats(self): #raise the dmg_feats children

        #=======================================================================
        # defautls
        #=======================================================================
        if self.db_f: start = time.time()
        logger = self.logger.getChild('raise_dfeats')
        
        id_str = self.get_id()
        
        self.tag = self.hse_o.hse_type + self.place_code
        
        """see below
        #=======================================================================
        # dependency check
        #======================================================================="""
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not self.dmg_code == 'S':    
                logger.debug('got unexpected dmg_code == %s'%self.dmg_code)
                raise IOError
            #if not hp.pd.isdf(df): raise IOError
            if not self.dfunc_type == 'dfeats':raise IOError
            if hasattr(self, 'kids_sd'): raise IOError
            
            if not self.session.state == 'init': 
                if len(self.kids_d) == 0: 
                    raise IOError
                
                #list of old kid gids
                old_kgid_s = set()
                for k, v in self.kids_d.iteritems(): 
                    old_kgid_s.update([v.gid])
                del v #need to relase the last one from this space

                
        """
        'only bothering if we are outside of the update scan'
        state = self.session.state != 'update' #a boolean of whether the session is upating
        _ = self.depend_outdated(search_key_l = ['set_geo_dxcol()'], #see if the parent has these
                             force_upd = state, #force the parent to update if found
                             halt = False) #clear all of my updates (except this func)"""
            
        #=======================================================================
        # dynamic vuln. mid simulation changes
        #=======================================================================

        if not self.session.state == 'init': 
            #=======================================================================
            # dependecy check
            #=======================================================================
            dep_p = [([self.parent],['set_geo_dxcol()'] )] #dependency paring
            if self.deps_is_dated(dep_p, method = 'force', caller = 'raise_dfeats'):
                raise IOError #because we are forcing this should alwys return FALSE

            #===================================================================
            # clear out old kids
            #===================================================================
            logger.debug('at \'%s\' session.state != init. killing kids \n'%id_str)
            
            self.kill_kids() # run kill() on all your children
            
            #add this reseting function
            """just adding this to everyone
            self.reset_func_od[self.reset_dfunc]='Dfunc'"""
            self.reset_shdw_kids_f = True
        

            
        #=======================================================================
        # post kill checks
        #=======================================================================
        if self.db_f:
            if not self.session.state == 'init': 
                if len(self.kids_d) > 0:
                    gc.collect()
                    if len(self.kids_d) > 0:
                        '1 child seems to hang'
                        logger.warning('sill have %i children left: %s'%(len(self.kids_d), self.kids_d.keys()))
                        raise IOError

                if not self.session.dyn_vuln_f: raise IOError
                if not isinstance(self.shdw_kids_d, dict): raise IOError
        

        #=======================================================================
        # set the childmeta data
        #=======================================================================
        logger.debug('at \'%s\' running set_childmetadf \n'%id_str)
        self.set_childmetadf()
        'this container is always needed for holding all the meta data and curve building'
            
        #=======================================================================
        # initizlie the dfeats
        #=======================================================================
        if self.tag in self.model.dfeats_d:
            #===================================================================
            # CLONE
            #===================================================================
            logger.debug("found my dfeats \'%s\' in the preloads"%self.tag)
            d_pull = self.model.dfeats_d[self.tag] #get the base dfeats
            d = hp.oop.deepcopy_objs(d_pull, container = dict, logger=logger) #make a copy for yourself
            
            #birth the clone
            for name, dfeat in d.iteritems(): 
                dfeat.init_clone(self)
                dfeat.inherit_logr(parent = self)

            """ init_clone does this   
            self.kids_d = wdict(d)"""
            
        else:
            #===================================================================
            # SPAWN your own
            #===================================================================
            df =  self.childmeta_df
            
            logger.debug('raising children from df %s\n'%str(df.shape))
            #=======================================================================
            # load the children
            #=======================================================================
            'probably a better way to deal witht hese falgs'
            d = self.raise_children_df(df, 
                                       kid_class = self.kid_class,
                                       dup_sibs_f = True) #adding the kwarg speeds it up a bit
        
            
        #=======================================================================
        # Activate the dfeats
        #=======================================================================
        logger.debug("setup_dfeat() on %i dfeats \n"%len(d))
        for name, dfeat in d.iteritems():
            dfeat.birth_dfeat(self) #tell the dfeat your the parent and inherit
        #=======================================================================
        # reset handling
        #=======================================================================
        
        #save teh recompile state
        if self.session.state == 'init':
            if self.session.dyn_vuln_f:
                
                #add the shadow kids and their handler
                logger.debug('deep copying over to shdw_kids_d')
                self.shdw_kids_d = hp.oop.deepcopy_objs(d, container = dict, logger=logger)
                self.reset_func_od[self.reset_dfunc]='Dfunc'
                """
                self.shdw_kids_d = self.session.shadow_objs(d)"""
                'this should create a shadow copy of the children. see ntoe in my header'
        else:
            pass
            """never called directly by the handler"""

        #=======================================================================
        # post checking
        #=======================================================================
        if self.db_f:

            self.check_family(d)
            
            for k, v in d.iteritems():
                if not v.hse_type == self.parent.hse_type:
                    raise IOError
                
            if not self.session.state == 'init': 
                #make sure all the old kids are dead
                book = self.session.family_d['Dmg_feat'] #get the book containing all these
                for gid in old_kgid_s:
                    if gid in book.keys(): 
                        raise IOError
                    
            #check the childmeta_df
            if np.any(pd.isnull(self.childmeta_df)):
                raise IOError
            """
            hp.pd.v(self.childmeta_df.sort_values('calc_price'))
            """
            
            stop = time.time()
            logger.debug('in %.4f secs finished on %i Dmg_feats: %s \n'%(stop - start, len(d), d.keys()))
        else:
            logger.debug('finished on %i Dmg_feats: %s \n'%(len(d), d.keys()))
            
        return


    
    def set_dfunc_anchor(self): #calculate my anchor_el from place code
        """
        This is a special type of geometry that needs to be determined regardles of whether the curves are dynanmic
        therefore kept seperate from teh geo calcs
        """       
        logger = self.logger.getChild('set_dfunc_anchor')
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not isinstance(self.parent.geo_dxcol, pd.DataFrame): raise IOError
            if not len(self.parent.geo_dxcol) > 0: raise IOError
            if self.place_code == 'total': return
            
        
            
        
        if not self.session.state=='init':
            #=======================================================================
            # check dependencies and frozen
            #=========================================================== ============
            if self.is_frozen('anchor_el', logger = logger): return True
            
            dep_l =  [([self.parent], ['set_hse_anchor()', 'set_geo_dxcol()'])]
            
            if self.deps_is_dated(dep_l, method = 'reque', caller = 'set_dfunc_anchor'):
                return False

        
        #=======================================================================
        # shortcut assignments
        #=======================================================================
        #pc      = self.place_code
        pa_el   = self.parent.anchor_el
        #id_str  = self.get_id()
        #===================================================================
        # #get the anchor el based on the anchor_ht_code
        #===================================================================
        #=======================================================================
        # logger.debug('\'%s\' for place_code \'%s\' parent.anchor_el = %.4f and anchor_ht_code \'%s\''
        #              %(id_str, pc, pa_el, self.anchor_ht_code))
        #=======================================================================
        #=======================================================================
        # #use legacy anchor heights
        #=======================================================================
        if self.anchor_ht_code == '*rfda_pars': 
            ' we could exclude this from updates as it shouldnt change'
            if self.place_code == 'B': 
                try: 
                    rfda_anch_ht = self.model.fdmgo_d['rfda_pars'].floor_ht #use the floor height from the par file
            
                except:
                    logger.error('failed to load rfda_pars')
                    raise IOError

                anchor_el = pa_el - rfda_anch_ht #basement curve
                'rfda_anch_ht is defined positive down'
                
            #main floor
            elif self.place_code == 'M':  
                anchor_el = pa_el #main floor, curve datum = house datum

            #garage
            elif self.place_code == 'G': 
                anchor_el = pa_el -.6 #use default for garage
                'the garage anchor was hard coded into the rfda curves'
                            
            else: raise IOError

        #=======================================================================
        # #pull elevations from parent
        #=======================================================================
        elif self.anchor_ht_code == '*hse': 
            
            #basemeents
            if self.place_code == 'B':
                jh = self.parent.joist_space
                
                B_f_height = float(self.parent.geo_dxcol.loc['height',('B','f')]) #pull from frame
                
                anchor_el = pa_el - B_f_height - jh#basement curve
                logger.debug('binv.B got B_f_height = %.4f, jh = %.4f'%(B_f_height, jh))
            #main floor
            elif self.place_code == 'M': 
                anchor_el = self.parent.anchor_el #main floor, curve datum = house datum
            
            #Garage
            elif self.place_code == 'G': 
                anchor_el = pa_el + self.parent.G_anchor_ht #use default for garage
                'parents anchor heights are defined positive up'
            
            else: raise IOError
            
        #=======================================================================
        # straight from dem
        #=======================================================================
        elif self.anchor_ht_code =='*dem':
            anchor_el = self.parent.dem_el
        
        #=======================================================================
        # striaght from parent
        #=======================================================================
        elif self.anchor_ht_code == '*parent':
            anchor_el = self.parent.anchor_el 
            
        else: raise IOError
            


        
        #=======================================================================
        # wrap up
        #=======================================================================
        logger.debug('for anchor_ht_code: \'%s\' and place_code \'%s\' found %.4f'
                     %(self.anchor_ht_code, self.place_code, anchor_el))
        
        #updates
        self.handle_upd('anchor_el', anchor_el, proxy(self), call_func = 'set_dfunc_anchor')
        self.anchor_ht = anchor_el - pa_el
        
        
        """done by teh handler
        #update the dd_ar
        'this needs to happen after the anchor_el update has been set'
        #logger.debug('running constrain_dd_ar \n')
        self.constrain_dd_ar()"""
        
        return True
    
    def constrain_dd_ar(self, tol = 0.01): #adjust the depth damage arrays to fit within the anchor
        """
        TODO: switch to sets
        """

        logger = self.logger.getChild('constrain_dd_ar')

            
            
        #=======================================================================
        # dependency checks
        #=======================================================================
        if not self.session.state == 'init':
            dep_p = [([self.parent], ['set_geo_dxcol()', 'set_hse_anchor()']),\
                     ([self], ['set_dfunc_anchor', 'build_dd_ar', 'recompile_ddar()']),\
                     (self.kids_d.values(), ['set_new_depth()'])]#,\
            
                    #the dfeats que this'
                    #([self],['recompile_ddar()'])]
            
            if self.deps_is_dated(dep_p, caller='constrain_dd_ar'):
                return False

            
        #=======================================================================
        # defaults
        #=======================================================================
        depth_list  = self.dd_ar[0].tolist()
        dmg_list    = self.dd_ar[1].tolist()
        
        max_d = max(depth_list)
        #logger.debug('with place_code = \'%s\' and max_d = %.2f'%(self.place_code, max_d))
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if self.anchor_el is None: raise IOError
            if min(depth_list) < 0: 
                raise IOError
            if min(dmg_list) < 0: raise IOError
            
        #=======================================================================
        # get the expected height
        #=======================================================================
        if self.place_code == 'B':
            height = self.parent.anchor_el - self.anchor_el
            'this allows for multiple anchor_ht_code methods. means the set_anchor has to be updated'
        else:
            height = self.parent.geo_dxcol.loc['height',(self.place_code,'t')]
            
        
        logger.debug('for place_code = \'%s\' got height = %.2f'%(self.place_code, height))
        
        if self.db_f:
            if not height > 0: 
                logger.error('got non-positive height = %.2f'%height)
                raise IOError
        #=======================================================================
        # collapse based on relative
        #=======================================================================
        #too short. raise it up
        if max_d <= height - tol: #check if the curve extents to teh ceiling
            'add a dummy value up'
            depth_list.append(height)
            dmg_list.append(max(dmg_list)) #add a max value
            
            logger.debug('dd_ar too short (%.2f). added dummy entry for height= %.2f'%(max_d, height))
            
        #too high. collapse
        elif max_d > height+tol:
            'change the last height depth value to match the height'
            ar = np.array(depth_list) 
            boolind =   ar> height #identify all of th evalues greater than the height
            
            ar[boolind] = height #set all these to the height
            
            depth_list = ar.tolist() #convert back to  list
            
            logger.debug('max(depth %.4f)> height (%.4f) collapsing %i'
                           %(max_d, height, boolind.sum()))
            
        else:
            logger.debug('dd_ar (%.2f) within tolerance (%.2f) of height (%.2f). no mods made'%(max_d, tol, height))

            
        if not min(depth_list) == 0: #add a zero here
            depth_list.append(0.0)
            dmg_list.append(0.0)
            logger.debug('added zero entry')
            
        #bundle this        
        dd_ar = np.sort(np.array([depth_list, dmg_list]), axis=1)

        #=======================================================================
        # wrap up
        #=======================================================================
        #update handling
        """
        ive just locked this... dont need to send to the update handler
        actually... we want the upd_cmd to be cleared of this function
        """
        logger.debug('finished with dd_ar[0]: %s \n'%dd_ar[0].tolist())
               
        self.handle_upd('dd_ar', dd_ar, proxy(self), call_func = 'constrain_dd_ar')
        'this shouldnt que any updates. removes yourself from the que'
        
        """mvoed to post updating
        self.calc_statres_dfunc()"""
        
        if self.db_f:
            if not min(self.dd_ar[0].tolist()) == 0:
                raise IOError
        

        return True
    
    def recompile_ddar(self,  #loop through all the children and recomplie the dyn_dmg_df
                        childn_l = None, outpath = None): 
        
        """
        this loops through each dmg_feat, and updates the dyn_dmg_df with price and depth
        then we recompile the depth damage array
        
        to improve efficiency, when were only changing a few dmg_feats, 
            pass a subset of child names
            
        
        Why not just call build_dfunc??
            This kills all the dfeat children
        
        #=======================================================================
        # CALLS
        #=======================================================================
        this should be called by the dynp handles for any changes to:
            house.geometry (geo_dxcol)
            Dmg_feat price/depth
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        logger = self.logger.getChild('recompile_ddar')
        'this gets called by teh dyno handles regardles of the dfunc type'
        if self.dfunc_type == 'dfeats': 
            #=======================================================================
            # check dependneciies
            #=======================================================================
            #dependency pairing
            dep_p = [([self.parent], ['set_geo_dxcol()']),\
                     (self.kids_d.values(), ['eval_price_calc_str', 'set_new_depth'])]
            
            if self.deps_is_dated(dep_p, caller='recompile_ddar'):
                return False
        
            #=======================================================================
            # prechecks
            #=======================================================================
            if self.db_f:
                
                if self.session.state == 'init': raise IOError
                
                #check your kids
                for k, v in self.kids_d.iteritems():
                    if not v.parent.__repr__() == self.__repr__(): raise IOError
                    if 'eval_price_calc_str' in v.upd_cmd_od.keys(): raise IOError
                    
                if not self.reset_dfunc in self.reset_func_od.keys(): raise IOError

            
            #=======================================================================
            # rebuild the depth damage array
            #=======================================================================
            id_str = self.get_id()
            logger.debug('at %s get_ddar_dfeats() \n'%id_str)
            self.dd_ar = self.get_ddar_dfeats() #recomplie the full curve
            'contrain sets a new copy'
        
            """this is added to the que by changes to dd_ar
            logger.debug('at %s constrain_dd_ar()\n'%id_str)
            self.constrain_dd_ar() """
            'thsi assumes the anchor_el is still accurate'
            
            'need to remove this here so that contrain_dd_ars dependencies arent tripped'
            self.del_upd_cmd(cmd_str = 'recompile_ddar()')
            
            res = self.constrain_dd_ar()
            #===================================================================
            # updates
            #===================================================================
            """letting constrain_dd_ar deal with the handles
            self.handle_upd('dd_ar', dd_ar, proxy(self), call_func = 'recompile_ddar')"""
        
            #=======================================================================
            # wrap up
            #=======================================================================
            """ this should only be run by upd_cmd (which executes del_upd_cmd
            'not using the update handler so we need to do the delete manually'
            self.del_upd_cmd() """
            
            self.reset_shdw_kids_f = True #flag the children for reseting
            'because we dont reset our children, this just swaps them all out for the shadow set'
            'todo: only swap changed ones?'
        #=======================================================================
        # post checks
        #=======================================================================

        if self.db_f:
            'this shouldnt be called during __init__. see raise_chidlren()'
            
            #=======================================================================
            # writiing
            #=======================================================================
            if not outpath is None:
                if outpath == True:
                    filename = os.path.join(self.outpath, 'dd_df.csv')
                else:
                    filename = outpath
                    
                hp.pd.write_to_file(filename, self.dd_df, logger=logger)
        
        return True
    

    
#===============================================================================
#     def upd_dd_df(self, #update a single dmg_feats entry in the dd_df
#                   dfeat_n,  #d[dfeat.name] = dfeat
#                   method = 'update'): 
#         
#         """
#         #=======================================================================
#         # TODO
#         #=======================================================================
#         we need to fix the handles so they execute in the callers names space
#             allows sending of parent.upd_dd_df(self.name)
#         """
#         raise IOError
#         #=======================================================================
#         # defaults
#         #=======================================================================
#         logger = self.logger.getChild('upd_dd_df')
#                 
#         df = self.childmeta_df.copy()
# 
#         
#         if self.db_f:
#             if not dfeat_n in self.kids_d.keys(): raise IOError
#             #if len(dfeat_nd) == 0: raise IOError
#             if len(df) == 0: raise IOError
#                 
#         
#         #=======================================================================
#         # loop and update
#         #=======================================================================
#         
#         #=======================================================================
#         # logger.debug('udpating with \'%s\' my dd_df %s for %i dfeats \n'%(method, str(df.shape), len(dfeat_nd)))
#         # for dname, dfeato in dfeat_nd.iteritems():
#         #=======================================================================
#         dfeato = self.kids_d[dfeat_n]
#         
#         logger.debug('\'%s\' on \'%s\' with calc_price \'%.2f\' and depth \'%.2f\''
#                      %(method, dfeato.name, dfeato.calc_price, dfeato.depth))
#         
#         #===================================================================
#         # add handling
#         #===================================================================
#         if method == 'add':
#             logger.debug('method = new. adding row for \'%s\''%dfeato.name)
#             
#             #add an empty row to the end
#             ser = pd.Series(index = df.columns)
#             ser['name'] = dfeato.name
#             df = df.append(ser, ignore_index = True)
#         
# 
#         #===================================================================
#         # # #locate this row in teh df
#         #===================================================================
#         boolidx = df['name'] == dfeato.name
#     
#         if self.db_f:
#         #check
#             #if dfeato.upd_f: raise IOError
#             if not boolidx.sum() == 1:
#                 if boolidx.sum() == 0: logger.error('could not find \'%s\' in the dd_df'%dfeato.name)
#                 else: logger.error('found multiple matches in the dd_df for \'%s\''%dfeato.name)
#                 """
#                 dd_df['name'].values.tolist()
#                 """
#                 raise IOError
#     
#         #=======================================================================
#         # make the update
#         #=======================================================================
#         if method == 'delete':
#             df = df[~boolidx] #take all rows except the located one
#         else:
#             #update tis row
#             """ This assumes that any updates have already happend to the child"""
#             df.loc[boolidx, 'calc_price'] = dfeato.calc_price
#             df.loc[boolidx, 'depth'] = dfeato.depth
#         
#         self.childmeta_df = df #update this
#         
#         self.que_upd_skinny('recompile_ddar()', 'childmeta_df',proxy(self), 'upd_dd_df') #que teh recompiler
#         'generally this is already in the que'
#         
#         logger.debug('udpated childmeta_df with %s'%str(df.shape))
#                
#         return True
#     
#             
#===============================================================================
    def run_dfunc(self, *vars, **kwargs):
        self.run_cnt += 1
        
        if self.db_f:
            #===================================================================
            # cjecls
            #===================================================================
            logger = self.logger.getChild('run_dfunc')

            if not 'dd_ar' in self.reset_d.keys():
                raise IOError
            
            if self.dfunc_type == 'dfeats':
                if not len(self.kids_d) > 0: 
                    raise IOError
                
            if len(self.upd_cmd_od) > 0:
                raise IOError
            
            
            #msg
            id_str = self.get_id()
            logger.debug('for %s returning get_dmg_wsl \n'%id_str)
            

        return self.get_dmg_wsl(*vars, **kwargs)
    
    def get_dmg_wsl(self, wsl): #main damage call. get the damage from the wsl
        #logger = self.logger.getChild('get_dmg_wsl')
                    
        #convert this wsl to depth
        depth    = self.get_depth(wsl)
        
        #shortcut
        if depth < 0: 
            return depth, 0.0, 0.0
           
        #get dmg from this depth
        dmg, dmg_raw       = self.get_dmg(depth)

        #=======================================================================
        # reporting
        #=======================================================================
       #=======================================================================
        # msg = 'depth = %.2f (%s) fdmg = %.2f (%s)'%(depth, condition_dep, dmg, condition_dmg)        
        # logger.debug(msg)
        #=======================================================================
        
        return depth, dmg, dmg_raw
    
    def get_depth(self, wsl):
        """
        This converts wsl to depth based on teh wsl and the bsmt_egrd
        this modified depth is sent to self.get_dmg_depth for interpolation from teh data array

        """       
        logger = self.logger.getChild('get_depth')
        depth_raw = wsl - self.anchor_el #get relative water level (depth)
        
        #=======================================================================
        # #sanity check
        #=======================================================================
        if self.db_f:
            if depth_raw > self.depth_allow_max:
                logger.error('\n got depth_raw (%.2f) > depth_allow_max (%.2f) for state: %s '
                             %(depth_raw, self.depth_allow_max, self.model.state))
                
                #get your height
                df      = self.hse_o.geo_dxcol[self.place_code] #make a slice for just this place
                height  = df.loc['height', 'f'] #        standing height
        

                logger.error('\n with anchor_el: %.2f wsl: %.2f, place_code: \'%s\' and height = %.2f'%
                             (self.anchor_el, wsl, self.place_code, height))
                
                raise IOError
            

        #=======================================================================
        # short cuts
        #=======================================================================
            
        if depth_raw < 0: #shortcut for water is too low to generate damage
            logger.debug('depth_raw (%.2f) < anchor_El (%.2f).skipping'%(depth_raw, self.anchor_el))
            return depth_raw
        
        #=======================================================================
        # set depth by floor
        #=======================================================================       
        if not self.dmg_type.startswith('B'): 
            depth = depth_raw
            condition = 'non-basement'
            
        else:  #basement
            #check if the flood depth is above the main floor
            if wsl > self.parent.anchor_el: 
                depth = depth_raw
                condition = 'high wsl'
                
            else: #apply the exposure grade
                if self.parent.bsmt_egrd == 'wet':
                    depth = depth_raw
                    condition = 'bsmt_egrd = WET'
                    
                elif self.parent.bsmt_egrd == 'damp':
                    
                    if self.model.damp_func_code == 'spill':
                        if depth_raw < self.parent.damp_spill_ht:
                            depth = 0
                            condition = 'bsmt_egrd = DAMP. spill. DRY. damp_spill_ht = %.2f'%self.parent.damp_spill_ht
                        else:
                            depth = depth_raw
                            condition = 'bsmt_egrd = DAMP. spill. WET damp_spill_ht = %.2f'%self.parent.damp_spill_ht
                        
                    elif self.model.damp_func_code == 'seep':
                        depth = depth_raw*0.5 #take half
                        condition = 'bsmt_egrd = DAMP. seep'
                        
                    else: raise IOError

                elif self.parent.bsmt_egrd == 'dry':
                    if depth_raw > self.parent.bsmt_opn_ht:
                        depth = depth_raw
                        condition = 'bsmt_egrd = DRY. above bsmt_opn_ht = %.2f'%self.parent.bsmt_opn_ht
                    else:
                        depth = 0
                        condition = 'bsmt_egrd = DRY. below bsmt_opn_ht = %.2f'%self.parent.bsmt_opn_ht
                    
                else:
                    logger.error('got unexpected code for bsmt_egrd = %s'%self.parent.bsmt_egrd )
                    raise IOError
                
                
        logger.debug('found depth = %.4f with \'%s\' from anchor_el = %.4f'%(depth, condition, self.anchor_el))
                
        return depth
                
    def get_dmg(self, depth): #basic depth from fdmg
        """
        for depth manipulations, see 'self.get_dmg_wsl'
        this dd_ar should be precompiled (based on dfunc_type) by one of the compiler functions
        """
        logger = self.logger.getChild('get_dmg')
        
        depth_list = self.dd_ar[0] #first row
        dmg_list = self.dd_ar[1]
        
        if self.db_f:
            if not min(depth_list)==0:
                logger.error('min(depth_list) (%.4f)!=0'%min(depth_list))
                'constrain_dd_ar should have been run to fix this'
                raise IOError
            
            #===================================================================
            # self.parent.hse_type
            # if self.parent.ayoc > 2000.0:
            #     if self.parent.fhz == 1.0:
            #         if self.name == 'BS':
            #      
            #===================================================================
            
            """
            df = self.childmeta_df
            hp.pd.v(df)
            """        
            
        
        #check for depth outside bounds
        if depth < min(depth_list):
            dmg_raw = 0 #below curve
            condition = 'below'
            
        elif depth > max(depth_list):
            dmg_raw = max(dmg_list) #above curve
            condition = 'above'
        else:
            dmg_raw = np.interp(depth, depth_list, dmg_list)
            condition = 'interp'
            
        #=======================================================================
        # scale with ratios
        #=======================================================================
        if not self.rat_attn =='*none':
            scale = eval(self.rat_attn)
        else:
            scale = 1.0
            
        dmg= dmg_raw * scale
            
        logger.debug('for \'%s\' type dmg = %.4f with \'%s\' from dd_ar %s min(%.4f) max(%.4f) scale(%.2f)'
                     %(self.dfunc_type, dmg, condition, str(self.dd_ar.shape),  min(depth_list),  max(depth_list), scale))
        
        return dmg, dmg_raw
    
    def reset_dfunc(self, *vars, **kwargs): #reset routine
        """
        keeping this command inthe func_od to make it easier for children to flag for a reset
        del self.reset_func_od['reset_dfunc']
        """
        logger = self.logger.getChild('reset_dfunc(%s)'%self.get_id())
        if self.reset_shdw_kids_f:
            
            
            logger.debug('reset_shdw_kids_f=TRUE. swapping kids \n')
            
            if self.db_f:
                if not isinstance(self.shdw_kids_d, dict):
                    raise IOError
            
                if not len(self.kids_d) > 0: raise IOError
                
                gen1_gid_l = self.kids_d.keys()
            
            self.swap_kids(self.shdw_kids_d)
            'swaps kids_d with shdw_kids_d'
            """
            self.session.family_d.keys()
            book = self.session.family_d['Dmg_feat']
            
            len(book)
            
            self.parent.hse_type
            
            self.kids_d.keys()
            """
            
            self.reset_shdw_kids_f = False #turn teh flag off
            
            if self.db_f:
                if not isinstance(self.kids_d, wdict): raise IOError
                if not len(self.kids_d) == len(self.shdw_kids_d): raise IOError
                
                #check that the old kid is not in teh family_d
                for gid in gen1_gid_l:
                    if gid in self.session.family_d['Dmg_feat'].keys():
                        raise IOError
                    
        else:
            pass
            #logger.warning("reset_shdw_kids_f=FALSE")
            
        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            for k, v in self.kids_d.iteritems():
                if not v.hse_type == self.parent.hse_type: raise IOError
                
            self.check_family()
                
            

        return 
    
    def calc_statres_dfunc(self): #clacluated a statistic for the dd_ar
        #logger = self.logger.getChild('calc_statres_dfunc')
        s = self.session.outpars_d[self.__class__.__name__]
        #=======================================================================
        # BS_ints
        #=======================================================================
        if 'intg_stat' in s: 
            res = self.calc_intg_stat()
            
        if 'kid_cnt' in s:
            self.kid_cnt = len(self.kids_d)
            

        
        """ using handler
        self.parent.BS_ints = intg_stat
        just setting this directly for now as we're only doing this on one child
        #=======================================================================
        # updates
        #=======================================================================
        
        self.parent.que_upd_skinny('calc_statres_hse()', 'intg_stat', proxy(self), 'calc_statres_dfunc')"""
        
        return True
    
    def calc_intg_stat(self, dx = 0.001):
        
        if not self.name == 'BS': return False#only care about the basement
        logger = self.logger.getChild('calc_intg_stat')
        
        depth_list  = self.dd_ar[0].tolist()
        dmg_list    = self.dd_ar[1].tolist()
            
        intg_stat = scipy.integrate.trapz(dmg_list, depth_list, dx = dx)
        
        logger.debug('got intg_state = %.4f'%intg_stat)
        
        self.intg_stat = intg_stat
            
        return True
                
    def plot_dd_ar(self, #plot the depth damage array
                   datum = None, ax=None, wtf=None, title=None, annot=None, **kwargs):
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('plot_dd_ar')
        if wtf == None: wtf = self.session._write_figs
        if title is None: title = self.parent.name +' ' + self.name + ' depth-damage plot'
        
        #=======================================================================
        # depth data setup 
        #=======================================================================
        
        #plot formatting
        depth_dato = copy.copy(self) #start with a copy of self
        
        
        #data transforms
        if datum is None:  #plot raw values
            delta = 0.0
            depth_dato.units = 'depth raw(m)'
            
        elif datum == 'real': #plot relative to elevation/anchor
            delta = self.anchor_el
            depth_dato.units = 'elevation (m)'

        elif datum == 'house': #plot relative to the house anchor
            delta = self.anchor_el - self.parent.anchor_el
            depth_dato.units = 'depth (m)'
            
        depth_ar = self.dd_ar[0] + delta
        depth_dato.label = depth_dato.units
        
        logger.debug('added %.2f to all depth values for datum: \'%s\''%(delta, datum))

        #=======================================================================
        # damage data setup 
        #=======================================================================
        dmg_ar = self.dd_ar[1] 
        
        #scale up for rfda
        if self.dfunc_type == 'rfda': dmg_ar= dmg_ar * self.parent.gis_area
        
        #annotation
        if not annot is None:
            if annot == True:
                annot = 'hse_type = %s\n'%self.hse_o.hse_type +\
                                'dfunc_type: %s\n'%self.dfunc_type
                
                raise IOError
                'this may be broken'
                if hasattr(self, 'f_area'):
                                annot = annot +\
                                'f_area = %.2f m2\n'%self.f_area +\
                                'f_per = %.2f m\n'%self.f_per +\
                                'f_inta = %.2f m2\n'%self.f_inta +\
                                'base_area = %.2f m2\n'%self.base_area +\
                                'base_per = %.2f m\n'%self.base_per +\
                                'base_inta = %.2f m2\n'%self.base_inta 
                else:
                    annot = annot +\
                    'gis_area = %.2f m2\n'%self.parent.gis_area

        #=======================================================================
        # send for  plotting
        #=======================================================================
        """
        dep_dato: dependent data object (generally y)
        indp_dato: indepdent data object (generally x)
        flip: flag to indicate whether to apply plot formatters from the y or the x name list 
        """
        ax = self.model.plot(self, indp_dato = depth_dato, dep_ar = dmg_ar, indp_ar = depth_ar,
                       annot = annot, flip = True, 
                       ax=ax, wtf=False, title=title, **kwargs)
        
        #=======================================================================
        # post formatting
        #=======================================================================
        #add thousands comma
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
            
        #=======================================================================
        # wrap up
        #=======================================================================
        if wtf: 
            fig = ax.figure
            flag = hp.plot.save_fig(self, fig, dpi = self.dpi)
            if not flag: raise IOError 
                            
        
        logger.debug('finished as %s'%title)
        
        return ax
        
    def write_dd_ar(self, filepath = None): #save teh dd_ar to file
        logger = self.logger.getChild('write_dd_ar')
        
        if filepath is None: 
            filename = self.parent.name + ' ' + self.name +'dd_ar.csv'
            filepath = os.path.join(self.outpath, filename)
        
        np.savetxt(filepath, self.dd_ar, delimiter = ',')
        
        logger.info('saved dd_ar to %s'%filepath)
        
        return filepath
        
class Dmg_feat( #single damage feature of a complex damage function
                hp.dyno.Dyno_wrap,
                hp.sim.Sim_o,  
                hp.oop.Child):

    #===========================================================================
    # program pars
    #===========================================================================
    # object handling overrides
    """
    raise_kids_f        = True
    db_f                = False
    post_cmd_str_l      = ['build_dfeat']"""
    """not worth it
    spc_inherit_ans     = set(['hse_o', 'place_code'])""" 
    
    """
    # dynp overrides
    run_upd_f = False #teh parent uses a complied damage function during run.""" 
    
    
    # Simulation object overrides
    perm_f              = False #not a permanent object
    
    """
    #===========================================================================
    # OBJECT PERMANENCE (shadow handling)
    #===========================================================================
    because we are temporary, we need a way to reset after each sim (if modified)
    this must completely replace our parents old kids with the new kids
    
    create a deepcopy of the original kids_d during _init_
    integrate shadow_kids into all simulation containers
        update all selections
    
    
    #===========================================================================
    # replacement flagging
    #===========================================================================
    all dynp changes should trigger self.flag_parent_shdw_kids()
    this notifes the parent to reset ALL children
        todo: consider using a container instead, and only replacing select children
        
    #===========================================================================
    # replacement handling
    #===========================================================================
    during Dfunc.reset_simo() (custom version) the old kids are relaced with the shadow
    
    #===========================================================================
    # container updating
    #===========================================================================
    Parent.kids_d and kids_sd: 
        generic_inherit()
    
    Selectors update at given intervals
        todo: add check on interval logic with perm_f=False
        
    Outputrs
    
    Dynp

    """

    #===========================================================================
    #user provided pars
    #===========================================================================
    hse_type = None
    place_code = None
    dmg_code = None
    cat_code = None
    base_area = None
    base_per = None
    base_height = None
    base_inta = None
    raw_index = None
    depth_dflt = None
    desc = None
    quantity = None
    unit = None
    unit_price = None
    base_price = None
    price_calc_str = None
    
    tag = None #concanation of hse_type + place_code
    #===========================================================================
    # calculated pars
    #===========================================================================
    #geometry placeholders
    'use geometry directly from parent'
    
    calc_price = None
    hse_o = None

    #===========================================================================
    # data containers
    #===========================================================================
    
    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Dmg_feat')
        logger.debug('start _init_')
        self.inherit_parent_ans=set(['mind', 'model'])
        
        super(Dmg_feat, self).__init__(*vars, **kwargs) #initilzie teh baseclass  
        

        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not isinstance(self.gid, basestring):
                raise IOError
            
            if self.place_code is None:
                raise IOError
        
        #=======================================================================
        # setup funcs
        #=======================================================================
        """ 
        dont want to activate this until the dfeat is on its proper parent
        NO. need this separate
        just bringing this all back
        logger.debug('build_dfeat \n')
        self.build_dfeat() """
        
        """ needs to be unique
        if self.sib_cnt == 0:
            logger.debug('getting dyno kids \n')"""
            
        'as we are non-permanent, this doesnt really do much'
        self.init_dyno()
                
        if self.db_f:
            logger.debug('check_dynh \n')
            self.check_dynh()
            
        logger.debug('initilized as \'%s\''%self.name)

        return
    
    def birth_dfeat(self, parent): #setup the dfeat under this parent
        """here we use a special inheritance 
        as we are spawning the dfeat with a different object than the real parent
        
        see datos.Dfeat_tbl.raise_all_dfeats()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        #logger = self.logger.getChild('birth_dfeat')
        
        """handled by init_clone
        self.parent = weakref.proxy(parent)"""
        self.state = 'birth'
        #=======================================================================
        # clear all your updates
        #=======================================================================
        self.halt_update()
        
        #=======================================================================
        # inherit all the attributes from your parent
        #=======================================================================
        self.hse_o = self.parent.parent

        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if self.parent is None: raise IOError

            if not self.hse_o == self.parent.parent: raise IOError
            
            if not self.hse_type == self.hse_o.hse_type:
                """
                self.hse_o.hse_type
                self.hse_type
                """
                raise IOError
            
            if not self.mind == self.model.mind: raise IOError
            
            if not self.model == self.parent.parent.model: raise IOError
            
            if self.hse_o.geo_dxcol is None:
                raise IOError
            
            if not self.parent.__class__.__name__ == 'Dfunc':
                raise IOError
        
        #=======================================================================
        # build the dfeat
        #=======================================================================
        #logger.debug("build_dfeat()")
        self.build_dfeat()
        #logger.debug('finished \n')
        self.state = 'ready'
        return
    
    def build_dfeat(self):
        'need this as standalone for  dyno handle calls'
        """fien to run on an outdated parent
        if self.depend_outdated(halt=True): return"""
        logger = self.logger.getChild('build_dfeat')
        
        #=======================================================================
        # check dependencies
        #=======================================================================
        if self.depend_outdated(    depend = self.hse_o,
                                    search_key_l = ['set_geo_dxcol()'], #see if the parent has these
                                    reque = False,
                                    force_upd = False, #force the parent to update if found
                                    halt = False): #clear all of my updates (except this func)
            
            raise IOError #should not be built if the parent is this outdated
        
        #=======================================================================
        # garage area override
        #=======================================================================
                
        #logger.debug('eval_price_calc \n')
        self.eval_price_calc_str()
        """ to speed things up, just pulling out the useful bits
        logger.debug('init_dyno \n')
        self.init_dyno()"""
        #logger.debug('finished \n')
        
        self.halt_update()
        """using halt command
        self.upd_cmd_od = OrderedDict() #empty the dictionary""" 
        
        return
                      

             
    def eval_price_calc_str(self): #get the calc price by evaluating the string in teh table
        """
        evalutes the price_calc_str provided on the dyn_dmg_tble colum for this bldg type
        #=======================================================================
        # CALLS
        #=======================================================================
        load_data()
        update.run()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('eval_price_calc_str')
        
        #=======================================================================
        # check dependneciies
        #=======================================================================
        #dependency pairing
        dep_p = [([self.hse_o], ['set_geo_dxcol()']),\
                 ([self.parent], ['build_dfunc()'])]
        
        if self.deps_is_dated(dep_p, caller='eval_price_calc_str()'):
            return False
            
            
        
        dxcol = self.hse_o.geo_dxcol #get a reference to the houses geometry dxcol
        df = dxcol[self.place_code] #make a slice for just this place
        
        #=======================================================================
        # #make all the variables local
        #=======================================================================
        """ so the price_calc_str has the right vars to work with"""
        height      = df.loc['height', 'f'] #        standing height
        
        t_area      = df.loc['area', 't'] #         total area for this floor
        f_area      = df.loc['area', 'f']#    finished area for this floor
        u_area      = df.loc['area', 'u'] #   unifnished area for this floor
        
        t_per       = df.loc['per', 't']  # total perimeter for this floor
        f_per       = df.loc['per', 'f']  # finished area perimenter
        u_per       = df.loc['per', 'u'] # unfinisehd perimeter
        
        f_inta      = df.loc['inta', 'f']
        u_inta      = df.loc['inta', 'u']
        

        base_area   = float(self.base_area)
        base_per    = float(self.base_per)
        base_inta   = float(self.base_inta)
        
        #attributes passed form the dfeat tabler
        base_price  = float(self.base_price)
        unit_price  = float(self.unit_price)
        quantity    = int(self.quantity)
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f: 
            if pd.isnull(t_area): raise IOError
            if t_area is None: raise IOError
            if t_area < 5: 
                raise IOError
        
        """
        hp.oop.log_all_attributes(self, logger = logger)
        """
        #=======================================================================
        # execute hte price calc string
        #=======================================================================
        try:        
            calc_price = eval(self.price_calc_str)
        except:
            logger.error('unable to evaluate price_calc_str: \'%s\''%self.price_calc_str)
            raise IOError
        
        #=======================================================================
        # postchecks
        #=======================================================================
        if self.db_f: 
            logger.debug('calc_price = %.2f on eval: %s'%(calc_price, self.price_calc_str))
        
            if pd.isnull(calc_price): 
                raise IOError
            if calc_price == 0: 
                logger.debug('got zero price')
            elif not calc_price >= 0: 
                logger.warning('not calc_price (%.2f) > 0 with \'%s\''%(calc_price, self.price_calc_str))
                raise IOError
            
            if not self.dfloc in self.parent.childmeta_df.index.tolist():
                logger.error('could not find my dfloc \'%s\' in my parents \'%s\' index'%(self.dfloc, self.parent.name))
                raise IOError
            
            if not self.name in self.parent.childmeta_df.loc[:,'name'].values.tolist():
                raise IOError
            
            if not self.parent.parent == self.hse_o:
                raise IOError
            
            if not self.hse_type == self.hse_o.hse_type:
                logger.error('hse_type mismatch')
                raise IOError
            
            if not self.session.state == 'init':
                if self.session.dyn_vuln_f:
                    if not self.parent.reset_dfunc in self.parent.reset_func_od.keys(): 
                        raise IOError
            
            """
            self.hse_o.kids_d.keys()
            self.parent.kids_d.keys()
            self.parent.name
            """
            
            
            """dont use reset_d on non-permanents
            if not self.session.state == 'init':
                if not 'calc_price' in self.reset_d.keys(): raise IOError"""
            
        #=======================================================================
        # update handling
        #=======================================================================
        #set the att
        self.calc_price = calc_price
        
        #update parents df
        self.parent.childmeta_df.loc[self.dfloc, 'calc_price'] = calc_price
        
        if not self.session.state == 'init':
            #flag parent
            'this is redundant as recomplie_dyn() should also flag this'
            self.parent.reset_shdw_kids_f = True
            
             
            
            if not self.state == 'birth':
                'dont want to recompile during the original compiling!'
                self.parent.que_upd_skinny('recompile_ddar()', 'calc_price',proxy(self),'eval_price_calc_str')
                
                'during birth we call a full halt'
                self.del_upd_cmd(cmd_str = 'eval_price_calc_str')

        return True
    
    def depth_check(self, 
                    depth = None, tol = 0.1, dep_check = True): #check that you are with the right parent
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('depth_check')

        if depth is None: depth = self.depth
        swap = False
        #=======================================================================
        # check dependneciies
        #=======================================================================
        if dep_check: #letting callers skip thsi for speed
            if self.is_frozen('depth', logger = logger): raise IOError
            
            #dependency pairing
            dep_l =  [([self.hse_o], ['set_hse_anchor()', 'set_geo_dxcol()']),\
                      ([self.parent], ['set_dfunc_anchor()', 'build_dfunc()'])]
            
            if self.deps_is_dated(dep_l, caller='depth_check()'):
                return False
        
                
        #=======================================================================
        # get floro height
        #=======================================================================
        if self.place_code == 'B':
            height = self.hse_o.anchor_el - self.parent.anchor_el
            'this allows for multiple anchor_ht_code methods. means the set_anchor has to be updated'
        else:
            height = self.hse_o.geo_dxcol.loc['height',(self.place_code,'f')]
            
        #=======================================================================
        # check depth
        #=======================================================================
        if depth > height: #too big for my floor
            newd = depth - height #take all the depth above the height 
            logger.debug('weve outgrowh our parent by %.2f'%newd)
            
            if self.place_code == 'B': 
                self.change_dfunc('M') #check and change parents 
                swap = True
            else:
                logger.debug('already on teh main floor... nothing to do')
                
        else: #still fit in my floor
            newd = depth
            
        #=======================================================================
        # handle new depth
        #=======================================================================
        if (newd > self.depth + tol) or (newd < self.depth - tol) or swap:
            logger.debug('with height=%.2f found significant depth change (%.2f to %.2f)'%(height, self.depth, newd))
            self.handle_upd('depth', newd, proxy(self), call_func = 'depth_check()') #change my attribute
            self.parent.reset_shdw_kids_f = True
        else:
            logger.debug('insignificant depth change')

            
        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            if not self.place_code == self.parent.place_code: raise IOError
            if not self.hse_type == self.hse_o.hse_type: raise IOError
            
            if newd < 0: 
                raise IOError
        
        return True
            
    def set_new_depth(self, new_el_raw, #elevate the selected dfeats to the new elevation
                      min_max = 'max', tol = 0.1): 
        """
        need a special function for modifying dfeat depths from elevations
        this converts elevations to depths using some logic and the dfunc's anchor_el
        #=======================================================================
        # INPUTS
        #=======================================================================
        new_el_raw:    new elevation for this feature
        min_max: logic to set new elevation relative to old elevation
        
        
        #=======================================================================
        # FUTURE THOUHGTS
        #=======================================================================
        perhaps we should have combined all dfuncs into 1?
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_new_depth')
        """decided to be more explicit
        if self.depend_outdated(): return"""
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not self.parent.parent == self.hse_o:
                raise IOError
            if not self.hse_o.hse_type == self.hse_type: 
                logger.error('my hse_type \'%s\' doesnt match my houses (%s) \'%s\''
                             %(self.hse_type, self.hse_o.name, self.hse_o.hse_type))
                raise IOError
            
        #=======================================================================
        # frozen and dependneces
        #=======================================================================
        if self.is_frozen('depth', logger = logger): return
        
        dep_l =  [([self.hse_o], ['set_hse_anchor()', 'set_geo_dxcol()']),\
                  ([self.parent], ['set_dfunc_anchor()', 'build_dfunc()'])]
        
        if self.deps_is_dated(dep_l, method = 'force', caller = 'set_new_depth'):
            raise IOError
        """
        problem with this shortcut is the parents dependencies may also be out of date
        if 'set_dfunc_anchor()' in self.parent.upd_cmd_od.keys():
            self.parent.set_dfunc_anchor()
            'this is the only dependency I care about. avoids ahving to add all the dfeats'"""
            

        old_el = self.parent.anchor_el + self.depth #should be the updated anchor)el
        #=======================================================================
        # get the new elevation
        #=======================================================================

        # raw elevation
        if min_max is None:
            logger.debug('using raw new_el %.2f'%new_el_raw)
            new_el = new_el_raw
            
        # logical elevation
        else:
            logger.debug('with new_el = %.2f using logic from old_el = %.2f and min_max code \'%s\''
                         %(new_el_raw, old_el, min_max))
            #===================================================================
            # by min/max flag
            #===================================================================
            if min_max == 'min': 
                new_el = min(new_el_raw, old_el)
            elif min_max == 'max': 
                new_el = max(new_el_raw, old_el)
            else: raise IOError
            
        #=======================================================================
        # depth handling
        #=======================================================================
        #shortcut out for non change
        'because we are setting the depth directly with this func, if there is no change, no need to check'
        if (old_el< new_el +tol) & (old_el > new_el - tol):
            logger.debug('old_el = new_el (%.2f). doing nothing'%new_el)
        else:
            #===================================================================
            # significant change
            #===================================================================
            
            new_depth = new_el - self.parent.anchor_el
            
            logger.debug('from parent.anchor_el = %.2f and new_el_raw = %.2f got new depth = %.2f'
                         %(self.parent.anchor_el, new_el, new_depth))
            
            #=======================================================================
            # send this for checking/handling
            #=======================================================================
            if not self.depth_check(depth = new_depth, tol = tol, dep_check = False): 
                raise IOError #we have the same dependencies so there is no reason this should fail

        #=======================================================================
        # updates
        #=======================================================================
        'depth_check sets the value'
        self.del_upd_cmd('set_new_depth') #no need to ru nthis check
        #=======================================================================
        # post check
        #=======================================================================
        if self.db_f:
            if self.depth < 0 : raise IOError
            
            if not self.parent.reset_dfunc in self.parent.reset_func_od.keys(): raise IOError
        
        #logger.debug('finished \n')
        
        return True
#===============================================================================
#     
#     
#         ""
#         #=======================================================================
#         # basement handling
#         #=======================================================================
#         'only moving things out of the basement. main floor can just collapse'
#         if self.parent.place_code == 'B':
#             
#             h = self.parent.parent.anchor_el - self.parent.anchor_el #get basement height
# 
#             #===============================================================
#             # too high for the basement
#             #===============================================================
#             if new_depth > h: #too high for the basement
#                 
#                 new_depth2 = new_depth - h #drop the depth down as weve moved up a floor
#                 
#                 'as were using delta anchors for height, this accounts for joist space'
#                 
#                 logger.debug('elevated depth (%.2f) > h(%.2f): switching parents dropping new depth = %.2f (from %.2f)'
#                              %(new_depth, h, new_depth2, self.depth))
#                 
#                 
#                 """NO! need to change parents first
#                 self.handle_upd('depth', new_depth2, proxy(self), call_func = 'set_new_depth') #change my attribute
#                 'this should update the parent_df. need this before I swap parents'"""
#                 
#                 self.change_dfunc('M') #check and change parents 
#                 
#             #===============================================================
#             # I still fit in the basement
#             #===============================================================
#             else: 
#                 logger.debug('d (%.2f) < h (%.2f): no parent swap'%(new_depth, h))
#                 new_depth2 = new_depth
#                 
#         #=======================================================================
#         # updates
#         #=======================================================================
#         self.handle_upd('depth', new_depth2, proxy(self), call_func = 'set_new_depth') #change my attribute
#         self.del_upd_cmd('depth_check') #no need to ru nthis check
#         'updates the childmeta_df and ques recompile_ddar() (flags for shadow kids)'
#         """handler does this
#         'todo: set so we are only updating this one child'
#         self.parent.que_upd_skinny('recompile_ddar()', 'depth',proxy(self),'set_new_depth') #notify the parent"""
#         
#         """
#         TRUE, but we may not run an update loop before a reset
#         handled by the recompile_ddar"""
#         self.parent.reset_shdw_kids_f = True
#         
#         #=======================================================================
#         # post check
#         #=======================================================================
#         if self.db_f:
#             if new_depth2 < 0 : raise IOError
#             
#             if not self.parent.reset_dfunc in self.parent.reset_func_od.keys(): raise IOError
#         
#         logger.debug('finished \n')
#         
#         return True
#===============================================================================
    
    def change_dfunc(self, new_pc): #move from one Dfunc to another
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('change_dfunc')
        parent_old = self.parent
        
        
        #=======================================================================
        # get the new parent
        #=======================================================================
        #get the new parents name
        'assumes the damage code/type stays the same... only changing place code'
        new_p_n = new_pc + self.parent.dmg_code 
        
        try:
            parent_new = self.hse_o.dfunc_d[new_p_n]
        except:
            if not new_p_n in self.hse_o.dfunc_d.keys():
                logger.error('passed dfunc name \'%s\' not loaded'%new_pc)
            
            raise IOError
        
        logger.debug('moving from \'%s\' to \'%s\''%(parent_old.name, parent_new.name))
        
        self.place_code = new_pc #set the new place_code
        #=======================================================================
        # swap out
        #=======================================================================
        """handled by update_childmeta() now
        parent_old.upd_dd_df(self.name, method = 'delete') #remove yourself from teh dd_df"""
        
        self.session.parent_swap(self, parent_new) #inherit, add to family lib, initilze logger, set new gid

        
        #=======================================================================
        # calcluate/apply changes
        #=======================================================================
        """price_calc wont change by just moving floors
        none of the birth_dfeat() cmds are necesary for a floor change
        #update your price calc
        self.eval_price_calc_str()"""
        
        """ handled by updaet_childmeta()
        parent_new.upd_dd_df(self.name, method = 'add') #remove yourself from teh dd_df"""
        #=======================================================================
        # updates
        #=======================================================================
        """this is covered by set_new_depth handle_upd
        parent_new.reset_shdw_kids_f = True #parent needs to swap out all its kids"""
        
        parent_old.reset_shdw_kids_f = True #tell your old parent to reset everyone next time
        
        #que the update commands on the parents
        'this is probably redundant '
        parent_old.que_upd_skinny('recompile_ddar()', 'depth',proxy(self),'change_dfunc')
        """ hte new parent should receive the update command from set_new_depth()
        """

        if self.db_f:
            
            if self.name in parent_old.childmeta_df.loc[:,'name'].tolist(): raise IOError
            if not self.name in parent_new.childmeta_df.loc[:,'name'].tolist(): raise IOError
            if not self.dfloc in parent_new.childmeta_df.index.tolist(): raise IOError
            
            if not self.parent.__repr__() == parent_new.__repr__():
                raise IOError
            
            if not self.parent.hse_o == self.hse_o: raise IOError
            
            if not self.hse_o.hse_type == self.hse_type: 
                raise IOError
            
            if not parent_old.reset_dfunc in parent_old.reset_func_od.keys(): raise IOError
            
            if not parent_new.reset_dfunc in parent_new.reset_func_od.keys(): raise IOError
            
            """
            self.parent.name
            df = parent_new.childmeta_df
            
            """
        return
    

def annot_builder(dfunc): #special annot builder helper

    annot_str = '%s_type: %s\n'%(dfunc.name, dfunc.dfunc_type) +\
                '    anchor_ht_code: %s\n'%dfunc.anchor_ht_code +\
                '    anchor_el: %.2f\n'%dfunc.anchor_el +\
                '    maxdd = %.2f m, $%.2f\n'%(max(dfunc.dd_ar[0]), max(dfunc.dd_ar[1]))
                
    return annot_str

        
  
        
        
        
        
    
