'''
main launching script for SOFDA model execution.

#===============================================================================
# Calling this script
#=============================================================================
via windows command line (see main_cmd)
via IDE/interactive/debugging (run this script)

#===============================================================================
# Variable handling philosophy
#===============================================================================
vars that define the model
    should be wholely contained in the pars file
    
vars that determine file handling and setup
    this script
    
vars that control debugging
    this script (with some in the pars file where necessary)
    

'''

__version__ = '1.0.0'


import logging, logging.config, os, sys, time, datetime
start = time.time() #start the clock

#module directory
mod_dir = os.path.dirname(__file__)

#===============================================================================
# #highest level debuggers
#===============================================================================
_prof_time = False

#===============================================================================
# define the main function
#===============================================================================
#===============================================================================
# from memory_profiler import profile
# 
# @profile
#===============================================================================
def run(    
            #main directory and control file setup
            parspath            = u'C:/LocalStore/03_TOOLS/SOFDA/_ins/_sample/SOFDA_sample.xls', 
            #control file name.    'gui' = use gui (default).
            work_dir            = 'home', #control for working directory. 'auto'=one directory up from this file.
            outpath             = 'auto', #control for the outpath. 'auto': generate outpath in the working directory; 'gui' let user select.
            
            dynp_hnd_file       = 'auto', #control for locating the dynp handle file
            
            #logger handling
            _log_cfg_fn         = 'logger.conf', #filetail of root logging configuration file (relative to mod_dir)
            lg_lvl              = None, #logging level of the sim logger. None = DEBUG
            _lg_stp             = 20, #for key iterations, interval on which to log
            
            #file write handling
            _write_data         = True, #write outputs to file
            _write_figs         = True, #flag whether to write the generated pyplots to file
            _write_ins          = True, #flag whether to writ eht einputs to file
            
            #debug handling
            _dbgmstr            = 'all', #master debug code ('any', 'all (see 'obj_test' tab)', 'none'). 
            _parlo_f            = True,  #flag to run in test mode (partial ddata loading)
            
            #profiling
            _prof_time          = _prof_time, #run while profile the program stats
            _prof_mem           = 0, #0, 1, 2, 3 memory profile level
            
            force_open          = False #force open the outputs folder
                                        ):

    
    #===============================================================================
    # Working directory ------------------------------------------------------
    #===============================================================================
    import hp.gui
    #assign
    if work_dir == 'pauto': 
        'for python runs. doesnt work for freezing'
        work_dir = os.path.dirname(os.path.dirname(__file__)) #default to 1 directories up
    
    elif work_dir == 'auto':
        work_dir = os.getcwd()
        
    elif work_dir == 'gui':
        work_dir = hp.gui.get_dir(title='select working directory', indir = os.getcwd())
        
    elif work_dir == 'home':
        from os.path import expanduser
        usr_dir = expanduser("~") #get the users windows folder
        
        work_dir = os.path.join(usr_dir, 'SOFDA')
        
        if not os.path.exists(work_dir): os.makedirs(work_dir)
        

    #check it
    if not os.path.exists(work_dir): 
        print('passed work_dir does not exist: \'%s\''%work_dir)
        raise IOError #passed working directory does nto exist
        
    os.chdir(work_dir) #set this to the working directory
    print('working directory set to \"%s\''%os.getcwd())
    

    #===============================================================================
    # Setup root log file ----------------------------------------------------------
    #===============================================================================
    logcfg_file = os.path.join(mod_dir, '_pars',_log_cfg_fn)
    
    if not os.path.exists(logcfg_file):  
        print('No logger Config File found at: \n   %s'%logcfg_file)
        raise IOError
    
    logger = logging.getLogger() #get the root logger
    logging.config.fileConfig(logcfg_file) #load the configuration file
    'this should create a logger int he working directory/_outs/root.log'
    logger.info('root logger initiated and configured from file at %s: %s'%(datetime.datetime.now(), logcfg_file))
    
    #===========================================================================
    # Control file   
    #===========================================================================
    if parspath == 'gui':
        parspath = hp.gui.gui_fileopen(title='Select SOFDA your control file.xls', indir = work_dir,
                                       filetypes = 'xls', logger=logger)
        
    if not os.path.exists(parspath):
        print('passed parfile \'%s\' does not exist'%parspath)
        raise IOError
    
    _, pars_fn = os.path.split(parspath) #get the conrol file name

    #===========================================================================
    # #outputs folder    
    #===========================================================================
    
    import hp.basic
    
    if _write_data:
        if outpath == 'auto':
            'defaulting to a _outs sub directory'
            outparent_path =  os.path.join(work_dir, '_outs')
            if not os.path.exists(outparent_path):
                logger.warning('default outs path does not exist. building')
                os.makedirs(outparent_path)
                
            outpath = hp.basic.setup_workdir(outparent_path, basename = pars_fn[:-4])
            
        elif outpath == 'gui':
            outpath = hp.gui.file_saveas(title='enter output folder name/location', indir = work_dir, logger=logger)
            
        #check and build
        if os.path.exists(outpath):
            logger.warning('selected outpath exists (%s)'%outpath)
        else:
            os.makedirs(outpath)

        #setup the ins copy
        inscopy_path = os.path.join(outpath, '_inscopy')
        if _write_ins: os.makedirs(inscopy_path)
        
    else:
        _write_ins = False
        _write_figs = False
        outpath, inscopy_path = '_none', '_none'
        
    #===========================================================================
    # handle files
    #===========================================================================
    if dynp_hnd_file == 'auto':
        dynp_hnd_file = os.path.join(mod_dir, '_pars', 'dynp_handles_20180928.xls')
        
    if not os.path.exists(dynp_hnd_file):
        raise IOError
       

    
    import scripts
    #===============================================================================
    # copy input files
    #===============================================================================
    #copy pyscripts
    if _write_ins: 
        _ = hp.basic.copy_file(__file__,inscopy_path) #copy this script
        _ = hp.basic.copy_file(scripts.__file__,inscopy_path) #copy the scripts script
        _ = hp.basic.copy_file(parspath, inscopy_path) #copy the control file
    
    #===============================================================================
    # LOAD SCRIPTS
    #===============================================================================
    session = scripts.Session(parspath = parspath, 
                              outpath = outpath, 
                              inscopy_path = inscopy_path,
                              dynp_hnd_file = dynp_hnd_file,
                              
                              _logstep = _lg_stp, 
                              lg_lvl = lg_lvl, 
                              
                              _write_data = _write_data, 
                              _write_figs = _write_figs, 
                              _write_ins = _write_ins, 
                              
                              _dbgmstr = _dbgmstr, 
                              _parlo_f = _parlo_f,

                              _prof_time = _prof_time, 
                              _prof_mem = _prof_mem)
    
    session.load_models()
    
    #===========================================================================
    # RUN SCRIPTS
    #===========================================================================
    session.run_session()
    
    #===========================================================================
    # WRITE RESULTS
    #===========================================================================
    session.write_results()
    #===============================================================================
    # WRAP UP
    #===============================================================================
    session.wrap_up()
    if force_open: hp.basic.force_open_dir(outpath)
    
    stop = time.time()
    logger.info('\n \n    in %.4f mins \'%s.%s\' finished at %s on \n    %s\n    %s\n'
                %((stop-start)/60.0, __name__,__version__, datetime.datetime.now(), pars_fn[:-4], outpath))



#===============================================================================
# IDE/standalone runs runs
#===============================================================================
if __name__ =="__main__": 
    
    if _prof_time: #profile the run
        import hp.basic
        
        #=======================================================================
        # file setup
        #=======================================================================
        work_dir = os.path.dirname(os.path.dirname(__file__)) #default to 1 directories up
        out_fldr = '_outs'

        outparent_path =  os.path.join(work_dir, out_fldr)
        master_out, inscopy_path = hp.basic.setup_workdir(outparent_path)

        
        run_str = 'run(outpath = master_out)'
        


        import hp.prof
        hp.prof.profile_run_skinny(run_str, outpath = master_out, localz = locals())

    else:
        run(work_dir='pauto', _dbgmstr='any') #for standalone runs


        
