'''
Created on May 17, 2018

@author: cef

hp functions for probabilistic calculations
'''
# Import Python LIbraries 
import os
import Tkinter
import tkFileDialog
from datetime import datetime
import logging
#import pandas as pd
import random
#import numpy as np
import shutil
import logging.config





mod_logger = logging.getLogger(__name__)


def Weighted_Choice(choices): #random selecting from a ictionary of {choice, weight}
    """
    TESTING
        import ABM
        choices =  ABM.debugEchoice()
    """ 
    total = sum(w for c, w in choices.iteritems()) #get teh total value for all the choices
    
    if total != 1: #check that the total choice is 1
        msg = 'The total value for all the choices equals %.2f not one'%total
        print msg, logger.critical(msg)
    
    seed = random.uniform(0, total) #pick a seed between teh total of all choices and zero
    upto = 0 #start at zero
    for c, w in choices.iteritems(): #icnrement through each choice
        if upto + w >= seed: #pick this choice if the seed falls within the increment
            return c
        else:
            upto += w #add this weight to the increment
    assert False, "Shouldn't get here"