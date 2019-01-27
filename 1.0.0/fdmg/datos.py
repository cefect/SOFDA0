'''
Created on Aug 28, 2018

@author: cef

object handlers for the fdmg tab
'''

#===============================================================================
# # IMPORT STANDARD MODS -------------------------------------------------------
#===============================================================================
import logging, re #os, sys, imp, time, re, math, copy, inspect


import pandas as pd
import numpy as np

#import scipy.integrate


#from collections import OrderedDict

#===============================================================================
#  IMPORT CUSTOM MODS ---------------------------------------------------------
#===============================================================================

import hp.pd


import hp.data


mod_logger = logging.getLogger(__name__)
mod_logger.debug('initilized')

class Fdmgo_data_wrap(object): #generic methods for Fdmg data objects
    
    def clean_binv_data(self, df_raw): #generic cleaning for binv style data
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('clean_binv_data')
        
        binv_df = self.parent.kids_d['binv'].childmeta_df
        
        if self.db_f:
            if not isinstance(binv_df, pd.DataFrame):
                raise IOError
            if not len(binv_df)  > 0:
                raise IOError
        
            
        #try and drop the FID column
        df = df_raw.drop('FID', axis=1, errors = 'ignore')
        df = df.drop('rank', axis=1, errors = 'ignore')
        

        #switch the index to the mind
        df.loc[:,self.mind] = df.loc[:,self.mind].astype(int)  #change the type
        df2 = df.set_index(self.mind).sort_index()
        

        #===================================================================
        # slice for the binv
        #===================================================================       
        boolind = np.isin(df2.index, binv_df.index)
        df3 = df2[boolind]
        
        if self.db_f:
            if not boolind.sum() == len(binv_df):
                boolind2 = np.isin(binv_df.index, df2.index)
                logger.error('failed to find %i entries specified in the binv: \n %s'
                             %(len(binv_df) - boolind.sum(),binv_df.index[~boolind2].values.tolist()))
                raise IOError #check data trimming?
        
        logger.debug('dropped %i (of %i) not found in teh binv to get %s'%
                     (len(df2) - len(df3), len(df2), str(df3.shape)))

        
        
        
        return df3

    def check_binv_data(self, df):
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('check_binv_data')
        
        binv_df = self.parent.kids_d['binv'].childmeta_df
        
        #null check
        if np.any(pd.isnull(df)):
            logger.error('got %i nulls'%pd.isnull(df).sum().sum())
            logger.debug('\n %s'%df[pd.isnull(df).sum(axis=1)==1])
            raise IOError
        

        
        #length check
        if not len(df) == len(binv_df):
            logger.error('my data length (%i) does not match the binv length (%i)'%(len(df), len(binv_df)))
            raise IOError
        
        #check for index match
        if not np.all(df.index == binv_df.index):
            raise IOError
        
    def apply_on_binv(self, #apply the passed key data to the binv
                      data_attn, hse_attn, 
                      coln = None
                      ): 
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('apply_on_binv')
        
        if coln is None: coln = hse_attn #assume this is how the data is labeld
        
        df = getattr(self, data_attn)
        binv = self.model.binv
        
        if self.db_f:
            if not isinstance(df, pd.DataFrame):
                raise IOError
            
            if not coln in df.columns:
                raise IOError
        
        ser = df.loc[:, coln]
        
        """
        type(ser)
        """
        logger.debug('from \'%s\' with %s applied to \'%s\''%
                     (data_attn, str(df.shape), hse_attn))
        #=======================================================================
        # mid session dynamic update to the objects
        #=======================================================================
        if not self.session.state == 'init':
            #=======================================================================
            # tell teh binv to update its houses
            #=======================================================================
            binv.set_all_hse_atts(hse_attn, ser = ser)
            
        #=======================================================================
        # pre run just update the binv_df
        #=======================================================================
        else:

            if self.db_f:
                binv_df = binv.childmeta_df.copy()
                if not np.all(binv_df.index == ser.index):
                    raise IOError
            
            binv.childmeta_df.loc[:,hse_attn] = ser
            logger.debug('merged %i entries for \'%s\' onto the binv_df %s'
                         %(len(ser), hse_attn, str(binv.childmeta_df.shape)))
            
        return

        
class Rfda_curve_data(#class object for rfda legacy pars
            hp.data.Data_wrapper,
            hp.oop.Child): 
    'made this a class for easy tracking/nesting of parameters'
    def __init__(self, *vars, **kwargs):
        super(Rfda_curve_data, self).__init__(*vars, **kwargs) #initilzie teh baseclass   
        
        self.load_data()
        
        self.logger.debug('fdmg.Rfda_curve_data initilized')
        
    def load_data(self): #load legacy pars from the df
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('load_data')
        #test pars
        if self.session._parlo_f: test_trim_row = self.test_trim_row
        else: test_trim_row = None
        
        self.filepath = self.get_filepath()
        
        #load from file
        df_raw = hp.pd.load_xls_df(self.filepath, logger=logger, test_trim_row = test_trim_row, 
                header = 0, index_col = None)
                
        
        self.data = df_raw 
        logger.debug('attached rfda_curve with %s'%str(self.data.shape))
                        
class Dfeat_tbl( #holder/generator fo all the dmg_feats
                 hp.data.Data_wrapper,
                hp.sim.Sim_o,
                hp.oop.Parent,
                hp.oop.Child): 
    """
    holder/generator fo all the dmg_feats
    """
    #===========================================================================
    # progran pars
    #===========================================================================
    extra_hse_types = ['AD'] #always load these house types

    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Dfeat_tbl')
        logger.debug('start _init_ ')
        
        self.inherit_parent_ans=set(['mind'])
        
        super(Dfeat_tbl, self).__init__(*vars, **kwargs) #initilzie teh baseclass   
        
        #=======================================================================
        # properties
        #=======================================================================
        import fdmg.scripts
        self.kid_class      = fdmg.scripts.Dmg_feat #mannually pass/attach this
        
        if self.session.wdfeats_f: #only bother if we're using dfeats
            logger.debug('load_data() \n')
            self.load_data()
                    
        self.logger.debug('fdmg.Dfeat_tbl initilized') 
        
        if self.db_f:
            if self.model is None:
                raise IOError
        
    def load_data(self):
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('load_data')
        #test pars
        if self.session._parlo_f: test_trim_row = self.test_trim_row
        else: test_trim_row = None
        
        self.filepath = self.get_filepath()
        
        #load from file
        df_dict = hp.pd.load_xls_df(self.filepath, logger=logger, test_trim_row = test_trim_row,
                                        skiprows = [1],header = 0, index_col = None, sheetname=None)
                
        
        'wrong function?'
        
        'todo: add some template check'
        
        for tabname, df_raw in df_dict.iteritems():
            #=======================================================================
            # send for cleaning
            #=======================================================================
            df_clean = self.clean_df(df_raw)
            
            #rewrite this
            df_dict[tabname] = df_clean
            
            logger.debug('loaded dynamic danage curve table for %s with %s'%(tabname, str(df_clean.shape)))
        
        self.df_dict = df_dict 
        self.data = None

        #=======================================================================
        # wrap up
        #=======================================================================
        if self.session._write_ins: _ = hp.basic.copy_file(self.filepath,self.session.inscopy_path)
        
        logger.debug('attached df_dict with %i entries'%len(df_dict))
  
    def clean_df(self, df_raw): #custom cleaner
        logger = self.logger.getChild('clean_df')
        
        df1 = self.generic_clean_df(df_raw)
        
        df2 = df1.dropna(how = 'all', axis='columns') #drop columns where ANY values are na
        
        #drop the 'note' column from the frame
        df3 = df2.drop('note', axis=1, errors='ignore')
        
        
        #=======================================================================
        # exclude small dfeats
        #=======================================================================
        if self.model.dfeat_xclud_price > 0:
            boolidx = df3['base_price'] <= self.model.dfeat_xclud_price
            df4 = df3.loc[~boolidx,:] #trim to just these
            
            if boolidx.sum() > 0:
                logger.warning('trimmed %i (of %i) dfeats below %.2f '%(boolidx.sum(), len(df3), self.model.dfeat_xclud_price))
                
                """
                hp.pd.v(df4.sort_values('base_price'))
                hp.pd.v(df3.sort_values('base_price'))
                """
        else: 
            df4 = df3
        
        
        'todo: drop any columns where name == np.nan'
        
        df_clean = df4
        
        hp.pd.cleaner_report(df_raw, df_clean, logger = logger)
        
        #=======================================================================
        # #post formatters
        #=======================================================================
        df_clean.loc[:,'depth'] = df_clean['depth_dflt'].values #duplicate this column
        """ This is throwing the SettingWithCopy warning.
        Tried for 20mins to figure this out, but couldnt find any chained indexing.
        """
        
        df_clean.loc[:,'calc_price'] = np.nan #add this as a blank column
        
        return df_clean
    """
    df_clean._is_view
    df_clean.values.base
    hp.pd.v(df_clean)
    self.name
    """
    
        
    
    def raise_all_dfeats(self):
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('raise_all_dfeats')
        
        d = self.df_dict
        
        dfeats_d = dict() #container of containers
        
        #=======================================================================
        # get the list of house types provided in teh binv
        #=======================================================================
        hse_types_l = set(self.parent.kids_d['binv'].hse_types_l) #pull from binv
        hse_types_l.update(self.extra_hse_types) #add the extras
        """
        self.parent.kids_d.keys()
        """
        
        #=======================================================================
        # load all the dfeats for this
        #=======================================================================
        logger.debug('on %i house types: %s \n'%(len(d), d.keys()))
        
        for hse_type, df in d.iteritems():
            if not hse_type in hse_types_l:
                logger.debug(' hse_type = \'%s\' not found in the binv. skipping'%hse_type)
                continue
            
            #get all teh place codes
            place_codes_l = df['place_code'].unique().tolist()
            
            #===================================================================
            # raise children on each of these
            #===================================================================
            logger.debug('building set for hse_type = \'%s\' with %i place_codes \n'%(hse_type, len(place_codes_l)))
            cnt = 0
            for place_code in place_codes_l:
                tag = hse_type + place_code #unique tag
                
                #get this metadata
                boolidx = df['place_code'] == place_code
                df_slice = df[boolidx].reset_index(drop=True)
                'need thsi so the dfloc aligns'
                """
                hp.pd.v(df_slice)
                df_slice.columns
                import pandas as pd
                pd.reset_option('all')
                pd.get_option("display.max_rows")
                pd.get_option("display.max_columns")
                pd.set_option("display.max_columns", 6)
                
                d.keys()
                
                d['AD']

                """
                
                #spawn this subset
                logger.debug('for \'%s\' raising children from %s'%(tag, str(df_slice.shape)))
                

                #run teh generic child raiser for all dfeats of this type
                'raise these as shadow children'
                dfeats_d[tag] = self.raise_children_df( df_slice, 
                                                        kid_class = self.kid_class,
                                                        dup_sibs_f = True,
                                                        shadow = True) 
                
                logger.debug('finished with %i dfeats on tag \'%s\''%(len(dfeats_d[tag]), tag))
                
                cnt += len(dfeats_d[tag])
                
            logger.debug('finish loop \'%s\' with %i'%(hse_type, len(dfeats_d)))
            
        logger.debug("finished with %i dfeats in %i sets raised: %s \n"%(cnt, len(dfeats_d), dfeats_d.keys()))
        

        return dfeats_d
        
class Flood_tbl(     #flood table worker
                     hp.data.Data_wrapper,
                     hp.oop.Child,
                     Fdmgo_data_wrap): 
    
    #===========================================================================
    # program
    #===========================================================================
    expected_tabn = ['wet', 'dry', 'aprot']
    #===========================================================================
    # from user
    #===========================================================================
    na_value = None #extra value to consider as null
    
    min_chk = 800 #minimum value to pass through checking
    max_chk = 2000 #maximumv alue to allow through checking
    
    wetnull_code = 'take_wet'
    
    wetdry_tol = 0.1
    
    damp_build_code = 'average'
    
    #area exposure grade
    area_egrd00 = None
    area_egrd01 = None
    area_egrd02 = None

    
    #===========================================================================
    # calculated
    #===========================================================================
    
    aprot_df = None
    
    
    
    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Flood_tbl')
        logger.debug('start _init_')
        super(Flood_tbl, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
        

        #=======================================================================
        # custom atts
        #=======================================================================
        self.mind = self.parent.mind
        self.model = self.parent
        
        self.wsl_d = dict()

        
        logger.debug('load_data() \n')
        self.load_data() #execute the standard data loader
        
        self.treat_wetnull()
        
        self.wetdry_logic_fix()
        
        self.build_damp()
        
        if self.db_f:  self.check_data()
        
        """ NO! only want the named flood table to set this
        self.set_area_prot_lvl()"""
        
        logger.debug('finish _init_ \n')
        
        return
    
    def load_data(self):
        logger = self.logger.getChild('load_data')
        
        self.filepath = self.get_filepath()
        
        d = self.loadr_real(self.filepath, multi = True)
        
        #=======================================================================
        # sort and attach
        #=======================================================================
        for k, v in d.iteritems():
            logger.debug('sending \'%s\' for cleaning'%k)
            df1 = self.clean_binv_data(v)
            if k in ['dry', 'wet']: 
                df2 = self.wsl_clean(df1)

                self.wsl_d[k] = df2
            elif k == 'aprot': 
                self.aprot_df = df1.astype(np.int)
                
            else: 
                logger.error('got unexpected tab name \'%s\''%k)
                raise IOError
        
        return
    
    def wsl_clean(self, df_raw):
        logger = self.logger.getChild('wsl_clean')
        #===================================================================
        # headers
        #===================================================================
        #reformat columns
        try:
            df_raw.columns = df_raw.columns.astype(int) #reformat the aeps as ints
        except:
            logger.error('failed to recast columns as int: \n %s'%(df_raw.columns))
            raise IOError
        
        
        #sort the columns
        df2 = df_raw.reindex(columns = sorted(df_raw.columns))

        
        #reformat values
        df2 = df2.astype(float)
        
        
        #=======================================================================
        # clean the user provided null
        #=======================================================================
        if not self.na_value is None:
            boolar = df2.values == self.na_value
            df2[boolar] = np.nan
            
            logger.warning('for set %i user identified values to null with \'%s\''%
                           (boolar.sum().sum(), self.na_value))
        
        """not working for some reason
        hp.pd.cleaner_report(df_raw, df2)"""
        
        logger.debug('cleaned to %s'%str(df2.shape))
        
        return df2
        
    def treat_wetnull(self): #apply the wetnull_code algorhitim to the dry
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('treat_wetnull')
                
        dfwet = self.wsl_d['wet']
        dfdry = self.wsl_d['dry']
        
        dfwet_raw = dfwet.copy()
        dfdry_raw = dfdry.copy()
        
        #=======================================================================
        # precheck
        #=======================================================================
        if self.db_f:
            if np.any(pd.isnull(dfdry)):
                logger.error('got %i null values for dfdry'%pd.isnull(dfdry).sum().sum()) 
                logger.debug('%s'%pd.isnull(dfdry).sum(axis=0))
                logger.debug('%s'%pd.isnull(dfdry).sum(axis=1))
                raise IOError #need all values for the dry scenario
        
        
        #=======================================================================
        # take _wet
        #=======================================================================
        if self.wetnull_code == 'take_dry':
            
            #identify location of null values in the dry frame
            boolar = pd.isnull(dfwet.values)
            
            #replace all the null values with the value from dfwet
            dfwet = dfwet.where(~boolar, other=dfdry)
            
            logger.info('set %i values from the wet flood to the dry flood'%boolar.sum())
            
        else: raise IOError
        
        
        #=======================================================================
        # reset into dict
        #=======================================================================
        self.wsl_d['wet'] = dfwet
        self.wsl_d['dry'] = dfdry
        
        return 
 
    def wetdry_logic_fix(self, tol = None): #fix the dry depths
        'makes sure our dry depths are at elast as high as our wet depths'
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('wetdry_logic_fix')
        
        if tol is None: tol = self.wetdry_tol
        
        dfwet = self.wsl_d['wet']
        dfdry = self.wsl_d['dry']
        
        #=======================================================================
        # find logical inconsistencies
        #=======================================================================

        delta = dfwet - dfdry 
        """
        hp.pd.v(delta)
        """
        
        boolar = delta < 0 #identify all inconsistencies
        
        if not np.any(boolar):
            logger.debug('no inconsistencies found. skipping')
            """
            hp.pd.v(boolar)
            hp.pd.v(delta)
            """
            return
        
        #=======================================================================
        # check we are within tolerance
        #=======================================================================
        delta2 = delta.where(boolar, other=0) #replace all positive deltas with zero
        boolar1 = abs(delta2) > tol #find where these are out of tolerance
        
        if np.any(boolar1):
            logger.error('found %i entries out of tolerance = %.2f'%(boolar1.sum().sum(), tol))
            raise IOError
        
        """
        hp.pd.v(abs(delta2))
        hp.pd.v(delta[boolar])
        hp.pd.v(boolar)
        
        hp.pd.v(delta)
        hp.pd.v(delta2)
        """
        #=======================================================================
        # replace bad dry values with the wet values
        #=======================================================================
        logger.warning('resolving %i dry > wet depth inconsistencies on the wet'%boolar.sum().sum())
        
        
        dfwet1 = dfwet.where(~boolar, other=dfdry) #set these
        
        self.wsl_d['wet'] = dfwet1 #reset this
        
        if self.db_f:
            delta2 = dfwet1 - dfdry
            
            if np.any(delta2 < 0):
                raise IOError
            
        
        return
        
    def check_data(self):
        logger = self.logger.getChild('check_data')
        
        #combine all your data for quicker checking
        d = self.wsl_d.copy()
        d['aprot'] = self.aprot_df
        #=======================================================================
        # check the tabs
        #=======================================================================
        if not np.all(np.isin(self.expected_tabn, d.keys())):
            logger.error("expected tabs %sin the flood table"%self.expected_tabn)
            raise IOError
        
        #=======================================================================
        # check the data
        #=======================================================================\
        shape = None
        cols = None

        for k, df in d.iteritems():
            logger.debug('checking \'%s\''%k)
            self.check_binv_data(df)
            
            #===================================================================
            # check wsl
            #===================================================================
            if k in ['wet', 'dry', 'damp']:
            
                if np.any(df.dropna().values > float(self.max_chk)):
                    raise IOError
                
                if np.any(df.dropna().values < float(self.min_chk)):
                    raise IOError
                
                #check the shape
                
                
                if shape is None: #set the first to check later
                    shape = df.shape
                    
                else: #check against last time
                    if not shape == df.shape:
                        logger.error('got shape mistmatch on wsl tabs')
                        raise IOError
                    """
                    hp.pd.v(df)
                    """
                    
                #check the columns match from tab to tab
                cols = df.columns.tolist()
                
                if not cols is None:
                    if not cols == df.columns.tolist():
                        raise IOError
                    
                
                #check that the depths vary with the aep
                col_last = None
                for coln, col in df.iteritems():
                    
                    #set first
                    if col_last is None:
                        col_last = col
                        coln_last = coln
                        continue
                    
                    #aep should increase monotonically
                    if not coln > coln_last:
                        raise IOError
                    
                    #see that all of our depths are greater than last years
                    boolar = col <= col_last
                    
                    if np.any(boolar):
                        logger.warning('for \'%s\' found %i (of %i) entries from \'%s\' <= \'%s\''
                                     %(k, boolar.sum().sum(), boolar.count().sum(), coln, coln_last))
                        
                        logger.debug('\n %s'%df.loc[boolar, (coln, coln_last)])
                        
                        
                        """
                        structural protections can prevent flood waters from rising for more extreme floods
                        
                        for small floods, wet== dry
                        """
                    coln_last = coln
                        

                    
                logger.debug('finished check on wsl data for \"%s\''%k)
             
            #===================================================================
            # check aprot   
            #===================================================================
            else:
                if not 'area_prot_lvl' in df.columns:
                    logger.error('expected area_prot_lvl as a colmn name on tab \'%s\''%k)
                    raise IOError
                
                
                
                
        #=======================================================================
        # wet dry logic
        #=======================================================================
        dfwet = d['wet']
        dfdry = d['dry']
        
        #look for bad values where the dry depth is higher than the wet depth
        boolar = dfdry > dfwet
        
        if np.any(boolar):
            logger.error('got %i (of %i) dry depths greater than wet depths'%(boolar.sum(), dfdry.count().sum()))
            """
            delta = dfdry - dfwet
            hp.pd.v(delta)
            hp.pd.v(dfdry)
            hp.pd.v(dfwet)
            hp.pd.v
            """
            raise IOError
        
        boolar = d['damp'] > dfwet
        
        if np.any(boolar):
            raise IOError
        
        boolar = dfdry > d['damp']
        
        if np.any(boolar):
            raise IOError
        
            
        return
    
    def build_damp(self): #build the damp levels
        logger = self.logger.getChild('build_damp')
        
        dfwet = self.wsl_d['wet']
        dfdry = self.wsl_d['dry']
        
        if self.damp_build_code == 'average':
            delta = dfwet - dfdry
            """
            hp.pd.v(delta)
            
            hp.pd.v(delta/0.5)
            """
            
            dfdamp = dfdry + delta/2.0
            
        elif self.damp_build_code.startswith('random'):
            
            #===================================================================
            # randmoly set some of these to the wet value
            #===================================================================
            #pull the ratio out of the kwarg 
            str1 = self.damp_build_code[7:]
            frac = float(re.sub('\)',"",str1)) #fraction of damp that should come from wet
            
            #generate random selection
            rand100 = np.random.randint(0,high=101, size=dfdry.shape, dtype=np.int) #randomly 0 - 100
            randbool = rand100 <= frac*100 #select those less then the passed fraction
            
            #set based on this selection
            dfdamp = dfdry.where(~randbool, other = dfwet) #replace with wet
            
            logger.info('generated dfdamp by setting %i (of %i) entries from wet onto dry'%(randbool.sum().sum(), dfdamp.count().count()))
            
            if self.db_f:
                if not np.all(dfdamp[randbool] == dfwet[randbool]):
                    raise IOError
                
                if not np.all(dfdamp[~randbool]==dfdry[~randbool]):
                    raise IOError
                       
            
            
        else:
            raise IOError
        
        logger.info('with damp_build_code = \'%s\', generated damp wsl with %s'%(self.damp_build_code, str(dfdamp.shape)))
        
        self.wsl_d['damp'] = dfdamp
        
        if self.db_f:
            deltaw = dfwet - dfdamp
            
            if np.any(deltaw<0):
                boolar = deltaw<0
                logger.error('got %i (of %i) damp levels greater than wet levels'%(boolar.sum().sum(), dfwet.count().sum()))
                """
                hp.pd.v(deltaw)
                hp.pd.v(delta)
                hp.pd.v(dfwet)
                hp.pd.v(dfdamp)
                
                hp.pd.v(dfdry)
                """
                raise IOError
            
            deltad = dfdamp - dfdry
            
            if np.any(deltad <0):
                raise IOError
        
        return
    

    
    """
    dfpull = self.model.binv.childmeta_df
    dfpull.columns
    binv_df.columns
    hp.pd.v(df3)
    """
    
    
class Fhr_tbl(    hp.data.Data_wrapper,
                     hp.oop.Child,
                     Fdmgo_data_wrap): #flood table worker
    
    
    
    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Fhr_tbl')
        logger.debug('start _init_')
        super(Fhr_tbl, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
        

        #=======================================================================
        # custom atts
        #=======================================================================
        self.mind = self.parent.mind
        self.model = self.parent
        

        logger.debug('load_data() \n')
        self.load_data() #execute the standard data loader
        

        

        
        
        logger.debug('finish _init_ \n')
        
        return
    
    def load_data(self):
        logger = self.logger.getChild('load_data')
        
        self.filepath = self.get_filepath()
        
        d = self.loadr_real(self.filepath, multi = True)
        
        #=======================================================================
        # clean and assign
        #======================================================================='
        """
        d.keys()
        """
        self.d = dict()

        for k, v in d.iteritems():
            logger.debug('on \'%s\' with %s'%(k, str(v.shape)))
            #clean
            df = hp.pd.clean_datapars(v)
            df1 = self.clean_binv_data(df)
            

            
            #check the data
            self.check_data(df1)
            
            #data formatting
            df1['fhz'] = df1['fhz'].astype(np.int)
            df1['bfe']  = df1['bfe'].astype(np.float)
            
            #add this cleaned frame into the collection
            self.d[k] = df1
            
            logger.debug('loaded adn added \'%s\' with %s'%(k, str(df1.shape)))
            
        logger.info('finished with %i fhrs loaded: %s'%(len(self.d), self.d.keys()))
            

            
        return
    
    def check_data(self, df):
        
        self.check_binv_data(df)

        
        #=======================================================================
        # check columns
        #=======================================================================
        if not np.all(np.isin(['fhz', 'bfe'], df.columns)):
            raise IOError
        
        return

        
        
                

                
        
        
            
            
        
    
        
        
          