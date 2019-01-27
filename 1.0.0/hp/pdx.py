'''
Created on Oct 9, 2018

@author: cef

helper commands for multindex frames
'''


import logging, copy, os, time
import numpy as np
import pandas as pd
import xlrd #this is here to test the optional dependency

#import matplotlib.pyplot as plt
from collections import OrderedDict

#===============================================================================
# other helpers
#===============================================================================
import hp.pd
import hp.basic


mod_logger = logging.getLogger(__name__) #creates a child logger of the root


def set_level_value(
        mdex_raw, #mdex to mainpulate
                    level, #level to adjust
                    pos, #position within the level
                    new_val,  #new value to set
                    unique = True, # consistent multindex where the levels repeat each time
                    logger=mod_logger):
    

    logger = logger.getChild('set_level_value')
    
    if unique:
        """
        This is for consistent multindex where the levels repeat each time
        """
        #get the original set of values
        lvl_vals_og = mdex_raw.unique(level=level).tolist()
        
        #make the change
        lvl_vals_n = copy.copy(lvl_vals_og)
        lvl_vals_n[pos] = new_val
        
        #modify the mdex
        mdex = mdex_raw.set_levels(lvl_vals_n, level=level)
    else:
        """
        for non-homogenous mdexes
        """
        
        mdex_ar = mdex_raw.values
        
        tup_og = mdex_ar[pos]
        
        #make the change
        tup_new = list(tup_og)
        tup_new[level] = new_val 
        
        #re insert
        mdex_ar[pos] = tuple(tup_new)
        
        #rebuild the mdex
        mdex = pd.MultiIndex.from_tuples(mdex_ar)
        

        
    
    logger.debug('finished setting \'%s\''%new_val)
    
    return mdex
