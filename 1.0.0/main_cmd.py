'''
Created on Aug 16, 2018

@author: cef

module for running ABMRI simulations from command line kwargs

primarily for python server runs
'''

#===============================================================================
# venv check
#===============================================================================
try: 
    import numpy

except:
    print ('failed to import numpy. venv not activated?')
    raise IOError

import os, argparse

if __name__ == '__main__':
    
    print('triggerd as \'%s\' woth getcwd: \'%s\' \n'%(__name__, os.getcwd()))
    #===========================================================================
    # setup argument parser
    #===========================================================================
    """
    defaults provided here will override those in main.py
    
    """
    parser = argparse.ArgumentParser()
    
    #postiional (required) arguments
    """ made this optional
    parser.add_argument("parspath",             help='full filename for control file (.xls). \'gui\' use gui (default)', default = 'gui')"""
    
    #===========================================================================
    # #optional (kwarg) arguments    
    #===========================================================================
    parser.add_argument("-parspath",'-cf',      
                        help='full filename for control file (.xls). \'gui\' use gui (default)', default = 'gui')
    
    parser.add_argument("-work_dir",'-wd',      
                        help='working directory. \'auto\'= use current . \'gui\' =use gui. home (default)= users home dir', default = 'home')
    parser.add_argument("-outpath",'-out',            
                        help='output directory \'auto\'=generate timestaped folder in work_dir\_outs. \'gui\'\n', default = 'auto')
    
    parser.add_argument("-dynp_hnd_file",'-dphf',            
                        help='directory for dynp handle file. \'auto\'(default) = use the packaged file', default = 'auto')
    
    parser.add_argument("-_log_cfg_fn","-lcfg", 
                        help='logger configuration file', default="logger_skinny.conf")
    parser.add_argument("-lg_lvl", "-ll",       
                        help='logging level of the sim logger', default='INFO')
    parser.add_argument("-_lg_stp","-ls",       
                        help='for key iterations, interval on which to log\n', type=int, default=100)

    
    parser.add_argument("-_write_data","-nowd", 
                        help='flag to write outputs', action="store_false") #defaults to True
    parser.add_argument("-_write_figs","-nowf",   
                        help='flag to write generated pyplots', action="store_false") #defaults to False
    parser.add_argument("-_write_ins","-wi",    
                        help='flag to copy inputs to the outputs folder\n', action="store_true")
    

    parser.add_argument("-_dbgmstr","-db",      
                        help='\'all\', \'any\', \'none\' (default) control which objects are db flagged', default='none')
    
    parser.add_argument("-_parlo_f","-part",       
                        help='partial data loading flag\n', action="store_true") #defaults to false
    
    parser.add_argument("-_prof_mem","-prof",     
                        help='0, 1, 2 memory profile level', type=int, default=0) 
    
    parser.add_argument("-force_open","-fo",       
                        help='force open the outputs folder after execution\n', action="store_true") #defaults to false


    
    args = parser.parse_args()
    
    print('parser got %i vars: \n'%(len(vars(args)))) #print all the parsed arguments in dictionary form

    for k, v in vars(args).iteritems():
        print('%s = %s'%(k, v))
        
    #===========================================================================
    # run the module
    #===========================================================================
    print('running sim.main with parsed vars \n \n')
    
    import main
    
    main.run(**vars(args))
    
    print('\n \n finished console run \n \n')
    

