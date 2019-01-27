'''
Created on Aug 30, 2018

@author: cef

scripts for handling dynamic objects

'''
#===============================================================================
#IMOPRTS --------------------------------------------------------------------
#===============================================================================
import os, sys, copy, random, re, logging, weakref, time, inspect

"""using modified version with 'prepend' method
from collections import OrderedDict"""
from hp.dict import MyOrderedDict as OrderedDict

from weakref import WeakValueDictionary as wdict

import pandas as pd
import numpy as np
import scipy.stats 

import hp.basic
import hp.oop
import hp.sim
import hp.sel

mod_logger = logging.getLogger(__name__)
mod_logger.debug('initilized')

class Dyno_wrap(object): #wraspper for objects which will ahve dynpamic pars applied to them
    

    
    
    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Dyno_wrap')
        logger.debug('start __init__')
        
        #=======================================================================
        # defaults
        #=======================================================================
        # user provided handles
        'all of these are now stored on the Session'

        # calculated pars
        self.upd_cmd_od       = None #dictionary of functions queued for update. upd_cmd_od[str(upd_cmd)] = [att_name, req_o, req_str]
        self.fzn_an_d        = None #dictinoary of frozen attribute names. d[att_name] = [req_o, req_str]
        self.dynk_lib        = None #library of dynk dictionaries
        
        self.dynk_perm_f     = True #flag that your dynks are permanent       
        
        self.post_upd_func_s     = None #common container of post upd functions per object

        # data containers
        self.upd_cnt = 0 #counter for number of update() runs
        
        
        
        #=======================================================================
        # _init_ cascade
        #=======================================================================
        
        super(Dyno_wrap, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
        
        #=======================================================================
        # unique attributes
        #=======================================================================
        
        'put thsi here so each object gets a unique empty dictionary'
        self.upd_cmd_od      = OrderedDict()  #dictionary of functions queued for update
        self.fzn_an_d       = dict() #d['att_nme':'controller'] of frozen attribute names. see apply_to_set(). should clear each sim
        
        self.reset_d.update({'upd_cnt':0, 'fzn_an_d':dict()})
        
        if not isinstance(self.gid, basestring):
            raise IOError
        
        #=======================================================================
        # common attributes
        #=======================================================================
        if self.sib_cnt == 0:
            self.dynk_ns        = set() #set of (dynamically wrapped) kid names specified in your handles (other classes you update)
            'I think we ar eOK with this being shared'
            logger.debug('setting handles \n')
        
        #=======================================================================
        # unique
        #=======================================================================
        'todo: make sure cascade is in the proper order so we can update this if necessary'
        self.post_upd_func_s = set()
        
        if self.db_f: 
            pass
            """called during _init_dyno
            logger.debug('check_dynh \n')
            self.check_dynh()"""
            
            
        """unfortunately, the originating caller has not completed its __init__
            (setup function shave not completed
        self.init_dyno()"""
        
        logger.debug('__init__ finished \n')
        
        return

    def get_hndl(self, par): #shortcut to pull the passed handle from the session
        
        return self.session.dyno_pars_d[self.__class__.__name__][par]
        
 
    def init_dyno(self): #initizlie my dynamic par attributes
        """
        because this requries the full library to be initilized, ive pulled all these commands out
        
        generally, this needs to be explicitly called (generally at the end)of the callers __init__
        
        called for all siblings
        """
        logger = self.logger.getChild('init_dyno')
        if self.perm_f:
            
            #===================================================================
            # handle post updating
            #===================================================================
            if len(self.post_upd_func_s) > 0:
                
                #see if any of your stats are being output
                if not self.__class__.__name__ in self.session.outpars_d.keys():
                    logger.warning('I have no outputers loaded on myself. clearing self.post_upd_func_s')
                    self.post_upd_func_s = set()
                    
                else:
                    #calc all these before setting the og
                    if not self.mypost_update():
                        raise IOError
                
                    #add yourself to the post updating que
                    self.session.post_updaters_wd[self.gid] = self
                
            
            logger.debug('set_og_vals \n')
            self.set_og_vals()

        
            """NO! each sibling has a unique set of dynk
            if self.sib_cnt == 0:
                logger.debug('getting dyno kids \n')"""
        
        #=======================================================================
        # setup your dependents
        #=======================================================================
        'for non-permanents this sets an empty dict'
        self.set_dynk_lib()  
                
        if self.db_f:
            logger.debug('check_dynh \n')
            self.check_dynh()
            
        logger.debug('finished \n')
        return
            
            
    def check_dynh(self): #check that the obj and its parent have the passed handels
        """
        checking hte handles under set_dyno_handles
        #=======================================================================
        # CALLS
        #=======================================================================
        init_dyno()
        """
        logger = self.logger.getChild('check_dynh')
        df = self.session.dynp_hnd_d[self.__class__.__name__] #get your handle pars
        logger.debug('on dynp_hnd_df %s'%str(df.shape))
        #=======================================================================
        # check yourself
        #=======================================================================
        self_upd_cmds = df.loc[:,'self_upd'].iloc[0]
        
        self.check_upd_cmds(self_upd_cmds)
        
        """ not using real children, should perform full check
        if not self.perm_f:
            logger.debug('perm_f = FALSE. skipping children check')"""
            
        if self.dynk_lib is None:
            raise IOError
        
        for attn in self.get_hndl('dyn_anl'):
            if not hasattr(self, attn):
                logger.error('I dont have attribute \'%s\''%attn)
                raise IOError
            
        return
        
        
    def check_upd_cmds(self, upd_cmds): #check the updating command
        
        #=======================================================================
        # exclusions
        #=======================================================================
        if upd_cmds == 'none': return #dont check these
        if pd.isnull(upd_cmds): raise IOError  #dont check these

        #=======================================================================
        # checker
        #=======================================================================
        if hp.basic.is_str_list(upd_cmds):  att_nl = hp.basic.str_to_list(upd_cmds)
        else:                               att_nl = [upd_cmds]
        
        if not hp.oop.has_att_l(self, att_nl): 
            logger = self.logger.getChild('check_upd_cmds')
            logger.error('missing some atts: %s'%att_nl)
            #hp.oop.has_att_l(self, att_nl)
            raise IOError
        
    def set_dynk_lib(self, container=wdict): #build your subscription list
        """
        The goal here  is to build a library of subscriber dictionaries during __init__
            so we dont have to make additional calls to this
        
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if len(self.dynk_ns) == 0: 
            self.dynk_lib = container() #just set an empty container
            return
        
        if not self.perm_f:
            self.dynk_lib = container() #just set an empty container
            return
            
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_dynk_lib')
        s = self.dynk_ns
        d = dict() #a library of wdicts
        
        #=======================================================================
        # prechecks
        #=======================================================================
        #if not self.session.state == 'init':
        if self.db_f:
            if not self.perm_f: raise IOError
            
        #=======================================================================
        # build the dictionary
        #=======================================================================
        logger.debug('on %i subscriber types: %s \n'%(len(s), s))
        for dynk_cn in s:
            logger.debug('building for dynk_cn \'%s\''%dynk_cn)
            
            book = self.get_dyn_kids_nd(dynk_cn, container=container)
            
            obj1 = book.values()[0]
            
            if obj1.perm_f:
            
                d[dynk_cn] = book
                
            else:
                logger.debug('this dynk \'%s\' is non-permanent. excluding from dynk_lib'%obj1.__class__.__name__)
                continue
            
            #if not isinstance(book, container): raise IOError
            
        logger.debug('setting dynk_lib with %i entries'%len(d))
        self.dynk_lib = copy.copy(d)
        
        return
        
        
    def get_dyn_kids_nd(self,  #get the child objects you apply updates to
                        dynk_cn,  #class name to build set of
                        container=wdict):
        """
        This is a hierarchical simple object selection (does not use Selectors)
        
        #=======================================================================
        # TODO
        #=======================================================================
        consider making this more explicit 
            user should specify if they expect a descendant object returned o
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('get_dyn_kids_nd')
        dynk_nd = container()
        'using update commands so weak references are set'
        
        
        logger.debug('building container of \'%s\''%dynk_cn)
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if dynk_cn.startswith('*'):
                pass
            
            elif dynk_cn.startswith('+'):
                dynk_cn1 = dynk_cn[1:]#drop the prefix
                #check this cn is in the library
                if not dynk_cn1 in self.session.family_d.keys():
                    logger.error('requested cn \'%s\' not fo und in the family_d.. load order?'%(dynk_cn))
                    raise IOError
            else:
                pass
        
        #=======================================================================
        # special vars
        #=======================================================================
        if dynk_cn.startswith('*'):
            dynk_cn1 = dynk_cn[1:]#drop flag
            
            if re.search(dynk_cn1, 'parent', re.IGNORECASE):
                dynk_nd.update({self.parent.name:self.parent.get_self()})

                logger.debug('got \'%s\'. setting to parent \'%s\''%(dynk_cn, self.parent.name))
            
            else:
                raise IOError #add more vars
            
        #=======================================================================
        # pull all objects of that type
        #=======================================================================
        elif dynk_cn.startswith('+'):
            dynk_cn1 = dynk_cn[1:]#drop the prefix
            dynk_nd.update(self.session.family_d[dynk_cn1]) #get subset of this generation
            logger.debug('pulled all %i  objects of type \'%s\' from teh family_d)'%(len(dynk_nd), dynk_cn))

                
        #=======================================================================
        # normal code of a class name
        #=======================================================================
        else:
            #=======================================================================
            # complex parent
            #=======================================================================
            if hasattr(self, 'kids_sd'):
                dynk_nd.update(self.kids_sd[dynk_cn])
                logger.debug('complex parent. pulled %i kids from page \'%s\' in teh kids_sd'%(len(dynk_nd), dynk_cn))
               
            #=======================================================================
            # simple parent
            #=======================================================================
            elif len(self.kids_d) > 0: #excluding parents with out kids (Flood)
                #===============================================================
                # see if theyve just asked for your direct descendants
                #===============================================================
                if dynk_cn == self.kids_d.values()[0].__class__.__name__:
                    dynk_nd.update(self.kids_d)
                    logger.debug('simple parent. pulled all \'%s\' children from kids_d (%i)'%(dynk_cn, len(dynk_nd)))
                else:
                    """not all of our users have this wrap.. easier to just copy/paste commands
                    'using the Sel_usr_wrap command '
                    dynk_nd.update(self.drop_subset(self.kids_d, pclass_n = dynk_cn)) #look for grandchildren as well"""


                    # run condenser to get pick correct level set
                    kcond_o = hp.oop.Kid_condenser(self.kids_d, 
                                                   dynk_cn, 
                                                   db_f = self.db_f, 
                                                   key_att = 'gid', #object attribte on which to key the result container
                                                   container = container,
                                                   logger = logger)
                         
                    dynk_nd.update(kcond_o.drop_all())
                    
                    if self.db_f:
                        for k, v in dynk_nd.iteritems(): 
                            if not v.parent.parent.__repr__() == self.__repr__(): 
                                raise IOError


        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            if len(dynk_nd) == 0: 
                raise IOError
        
            self.dyn_kid_check(dynk_nd) #check the consistency of all these

                
        #=======================================================================
        # set update flag
        #=======================================================================
        if self.session.state == 'init':
            if self.dynk_perm_f:
                'setting this as a global flag (least common denominator of all kids'
                self.dynk_perm_f = dynk_nd.values()[0].perm_f #just steal from the first kid
                logger.debug('during init took perm_f = \'%s\' from first dynk'%(self.dynk_perm_f))

        return dynk_nd
        
    def set_og_vals(self): #set the reset vals for all the dynp kids
        
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('set_og_vals')
        
        #check if this is a permeanet object
        if not self.perm_f:
            logger.debug('perm_f=FALSE. no need to set ogs. skipping')
            return
        
        #=======================================================================
        # get atts on which to store
        #=======================================================================
        #pull dynamic pars
        attn_s = set(self.get_hndl('dyn_anl'))
        
        #pull stat pars from ou tputrs
        cn = self.__class__.__name__
        if cn in self.session.outpars_d.keys():
            attn_s.update(self.session.outpars_d[cn])

        'this will probably overwrite a lot of the attns because we often output dyanmic atts'
        
        if self.db_f:
            #===================================================================
            # check for valid outputr request
            #===================================================================
            s = self.session.outpars_d[cn]
            for attn in s:
                if not hasattr(self, attn):
                    logger.error('got invalid output attribute request \'%s\''%attn)
                    raise IOError
                
            s = set(self.get_hndl('dyn_anl'))
            
            for attn in s:
                if not hasattr(self, attn):
                    logger.error('got invalid dynp handle attribute request \'%s\''%attn)
                    raise IOError
            
            

        #=======================================================================
        # pull values and store
        #=======================================================================
        logger.debug('from the dynp file, collecting og att vals on %i attributes: %s \n'%(len(attn_s), attn_s))
        cnt = 0
        for attn in attn_s:
            
            #get the og
            try:attv = getattr(self, attn)
            except:
                logger.error('attribute \'%s\' not found. check the handle file?. bad output attn request?'%(attn))
                raise IOError
                    
            #store the og
            if attv is None:
                logger.warning('\'%s\' not loaded yet. skipping'%(attn))
                'some Dfuncs dont use all the attributes'
                'outputrs with selectors'
                #raise IOError
            
            else:
                self.reset_d[attn] = copy.copy(attv)
                
                if hasattr(attv, 'shape'): 
                    logger.debug('in reset_d added \'%s\' with shape %s'%(attn, str(attv.shape)))
                else: 
                    logger.debug('in reset_d added \'%s\' = \'%s\''%(attn, attv))
                
                cnt +=1
            

        logger.debug('finished with %i total entries colected and stored into the reset_d (%i)'%(cnt, len(self.reset_d)))
        

    def handle_upd(self, #apply the approriate updates to this object based on what att was modified
                        att_name, new_val, req_o, call_func=None):
        """
        adds all the commands listed in the handle pars for this attribute (and its children)
        
        2018 08 21
        reworked this so it should accept updates from dynps or functions
        #=======================================================================
        # CALLS
        #=======================================================================
        Dynamic_par.apply_to_set() #dynamic parameter changes
        
        Dyn_wrapper.some_func() #object driven changes
           
        
        #=======================================================================
        # INPUTS
        #=======================================================================
        run_upd_f = TRUE:
            more efficients. allows each dynp to make its changes
                then updates are applied only to the objects once they are run
        
        run_upd_f = False:
            applies the update during each dynp
            necessary for those objects which do not have a run routine
        
        req_o: object making the update request
        req_str: some string describing the call better
        """
        #=======================================================================
        # shrotcuts
        #=======================================================================
        """NO! we us the updaters and the setters during init
        if self.session.state == 'init': return"""
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('handle_upd')
        
        old_val = getattr(self, att_name)
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if req_o.__class__.__name__ == 'Dynamic_par':
                if att_name in self.get_hndl('lock_anl'):
                    logger.error('change attemnpted on locked attribute \'%s\''%att_name)
                    raise IOError
            
            if not att_name in self.get_hndl('dyn_anl'):
                logger.error('change attempted on attribute not in the dynp handles \'%s\''%att_name)
                raise IOError
            
            if not hasattr(self, att_name): raise IOError
            
            if not isinstance(req_o, weakref.ProxyType):
                raise IOError
        
        #=======================================================================
        # clear the caller
        #=======================================================================
        #try and remove the calling function from teh updates
        """
        want to do this in all cases as the caller has just executed
        for object.somefunc custom calls, this makes sure we havent requed the command
        
        for some complex updates, we may still be added back by some later, larger update function string"""
        if call_func is None: call_func = inspect.stack()[1][3] #get the caller function
        if not call_func.startswith('_'):
            self.del_upd_cmd(cmd_str = call_func) #try and remove the caller from the queue
            
        #=======================================================================
        # shrotcuts
        #=======================================================================
        if not old_val is None:
            if np.array(new_val == old_val).all():
                logger.debug('for \'%s\' new_val == old_val: skipping'%att_name)
                return
        
               
            if hp.basic.isnum(new_val):
                try:
                    if round(new_val, 4) == round(old_val, 4):
                        logger.debug('values are close enough. skipping')
                        return
                except:
                    if hp.basic.isnum(old_val): 
                        raise IOError

            
        #=======================================================================
        # msg setup
        #=======================================================================
        if self.db_f: #log it
            log_str = 'with att_name \'%s\', req_o \'%s\', req_str \'%s\' '%(att_name,  req_o.name, call_func)
            
            'need to handle multidimensional types'
            nv_str = hp.pd.val_to_str(new_val)
            ov_str = hp.pd.val_to_str(old_val)
                
            #logger.debug(log_str + 'old_val \'%s\' and new_val \'%s\''%(ov_str, nv_str))
            
        
        #=======================================================================
        # Freeze check
        #=======================================================================
        if att_name in self.fzn_an_d.keys():
        
            logger.debug('change on \'%s\' requested by \'%s\' is frozen by \'%s.%s\'. skipping'
                         %(att_name, req_o.name, self.fzn_an_d[att_name][0].name, self.fzn_an_d[att_name][1]))
            
            if self.fzn_an_d[att_name].name == req_o.name:
                logger.error('The requested froze this attribute and am trying to change it again')
                raise IOError
            
            return
        
        """ we only want to allow dynps to freeze attributes"""
        #=======================================================================
        # set new value
        #=======================================================================
        if not pd.isnull(np.array(new_val)).all():

            
            
            setattr(self, att_name, new_val)
            logger.debug('set attribute \'%s\' with \'%s\''%(att_name, type(new_val)))
            
            
            if self.db_f:
                if not isinstance(new_val, type(old_val)):
                    if not isinstance(new_val, basestring): #ignore unicode/str fli9ps
                        logger.warning('for \'%s\' got type mismatch from old \'%s\' to new \'%s\''%
                                       (att_name, type(old_val), type(new_val)))
            
            
        else: 
            logger.warning('got null new_value. not setting')
        
        #=======================================================================
        # SECONDARY UPDATE HANDLING
        #=======================================================================

        #=======================================================================
        # get handles
        #=======================================================================
        df = self.session.dynp_hnd_d[self.__class__.__name__] #get your handle pars
        
        #make slice
        try:
            boolidx = df.loc[:,'att_name'] == att_name #find this attribute
            ser = df[boolidx].iloc[0]
            ser.name = att_name #set the name
        
        #error handling 
        except:
            if not att_name in df.loc[:,'att_name'].values:
                logger.error('passed att_name \'%s\' was not found in the handles'%att_name)
            else:
                logger.error('unable to get pars for \'%s\''%att_name)
                
            raise IOError #check that this entry is in the dynp_handles

        #logger.debug('for \'%s\' got handles: \n %s'%(att_name, df[boolidx]))
        #=======================================================================
        # pass teh commands
        #=======================================================================
        if not self.session.state == 'init':
            #self.handle_upd_funcs(att_name, ser, req_o, call_func)
            if not ser['self_upd'] == 'none':
                logger.debug('from \'%s\' handling self_upd with \'%s\''%(att_name, ser['self_upd']))
                self.que_upd_full(ser['self_upd'], att_name, req_o, call_func = call_func)
                
            if not ser['dynk_hndl'] == 'none':
                self.handle_kid_upd(att_name, req_o, call_func=call_func)
                        
        #update the parents df
        if ser['upd_df']: 
            #logger.debug('updating parents df')
            self.parent_df_upd(att_name, new_val)

        
        #logger.debug('finished on \'%s\'\n'%att_name)

        return
        
        
      
    def handle_kid_upd(self,  #handle updates on a single type of kid
                       att_name, req_o, call_func = None, #additiona pars to pass onto upd_que
                       method ='add',
                       **que_upd_kwargs):  #kwargs to pass onto que_upd
        """
        #=======================================================================
        # kids set handling ()
        #=======================================================================
        if the dynk_cn is not found in teh kids_sd, 
            updates are passed to all objects of that cn (family_od)
        #=======================================================================
        # CALLS
        #=======================================================================
        self.handle_upd
        
        #=======================================================================
        # key vargs
        #=======================================================================
        raw_cmd_str: this could be a list of commands or a single command
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('handle_kid_upd')  
        
        #=======================================================================
        # get handles
        #=======================================================================
        hnd = self.get_hndl('dynk_hndl_d')[att_name]
        
        logger.debug('on dynk_cn: %s \n'%hnd.keys())
        
        for dynk_cn, raw_cmd_str in hnd.iteritems():
            
            logger.debug('from handles for \'%s\' with cmd_str \'%s\' for dynk_cn \'%s\''%(att_name, raw_cmd_str, dynk_cn))

            #=======================================================================
            # get theses dynkids
            #=======================================================================
            #dynk_nd = self.get_dyn_dynk_nd(dynk_cn)
            if dynk_cn == '*parent':  #parent shortcut
                dynk_nd = {self.parent.name: self.parent}
                
            else: #normal pull
                try:
                    dynk_nd = self.dynk_lib[dynk_cn]
                    'set_dynk_lib excludes non-permanents from this'
                except:
                    #===============================================================
                    # no entry. (non-permanent dynkids?)
                    #===============================================================
                    if not dynk_cn in self.dynk_lib.keys():
                        logger.debug("passed dynk_cn \'%s\' not in my dynk_lib. getting new pick"%dynk_cn)
                        dynk_nd = self.get_dyn_kids_nd(dynk_cn)
                        
                    else:
                        raise IOError
                    
            #=======================================================================
            # prechecks
            #=======================================================================
            if self.db_f: 
                if dynk_cn.startswith('+'): dynk_cn1 = dynk_cn[1:] #drop the prefix
                else: dynk_cn1 = dynk_cn
                    
                if dynk_nd is None: raise IOError
                if len(dynk_nd) == 0: raise IOError
                
                obj1 = dynk_nd.values()[0]
                
                if not obj1.perm_f:
                    if dynk_cn1 in self.dynk_lib.keys(): 
                        raise IOError
                    
                if not dynk_cn.startswith('*'): #exclude specials

                    if not obj1.__class__.__name__ == dynk_cn1: 
                        raise IOError
                    
                if self.__class__.__name__ == 'House':
                    if dynk_cn == 'Dfunc':
                        for k, v in dynk_nd.iteritems():
                            if not v.parent.__repr__ == self.__repr__:
                                raise IOError
                
                """ we allow non children dynk
                self.name
                for k, v in dynk_nd.iteritems():
                    print '%s.%s'%(v.parent.name, k)
                
                    if not v.parent = self: raise IOError"""
    
            #=======================================================================
            # handle each kids update
            #=======================================================================
            if len(dynk_nd) > 1: logger.debug('on dynk_nd with %i entries'%len(dynk_nd))
            cnt = 0
            for name, obj in dynk_nd.iteritems():
                cnt += 1
                logger.debug('\'%s\' cmd_str \'%s\' on \'%s\''%(method, raw_cmd_str, name ))
                
                if method == 'add':
                    obj.que_upd_full(raw_cmd_str,att_name, req_o, call_func = call_func, **que_upd_kwargs)
                elif method == 'delete':
                    obj.del_upd_cmd(cmd_str = raw_cmd_str)
                else: 
                    logger.error('got unexpected method kwarg \'%s\''%method)
                    raise IOError
                
            if cnt > 1:
                logger.debug('by \'%s\' handled \'%s\' on %i dependents '%(method, raw_cmd_str, cnt))
            
        #logger.debug('finished \n')
        
        return

    def que_upd_full(self, #que an update command on myself (from teh handler)
                    upd_cmd_str, att_name, req_o, 
                    call_func = None,
                    allow_self_que = False): 
        """
        #=======================================================================
        # USE
        #=======================================================================       
        
        #=======================================================================
        # INPUTS
        #=======================================================================
        upd_cmd: update command sent for queing by the controller
        controller: object requesting the update command
        
        upd_ovr: update override flag. forces  update here (instead of during computational run0
        
        self.run_upd_f #flag controlling whether updates are applied during a run or during each dynp.
        
        #=======================================================================
        # OUTPUTS
        #=======================================================================
        upd_cmd_od: dictionary of update commands and meta data
            keys: update command
            values: [controller requesting update, att_name controller's att name triggering this]
            
        made this a dictionary with metadata for better tracking of where the updates come from
        
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if upd_cmd_str is None: 
            return
        #already queued
        if str(upd_cmd_str) in self.upd_cmd_od.keys(): 
            'we have a more sophisticated version below'
            #logger.debug('command \'%s\' already queued. skipping'%str(upd_cmd_str) )
            return
        
        upd_cmd_str = str(upd_cmd_str)
        upd_ovr = False
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('que_upd_full') 
        
        if call_func is None: call_func = inspect.stack()[1][3]
        #upd_cmd_str = str(upd_cmd_str)
        upd_ovr = False
        
        #check if were actually just re-queueing the requester
        if not allow_self_que:            
            if re.search(call_func, str(upd_cmd_str), re.IGNORECASE):
                logger.debug('self request by \'%s.%s\'. doing nothing'%(req_o.name, call_func))
                self.del_upd_cmd(cmd_str = upd_cmd_str) #try and remove it
                return
                    
        #wrong session state
        """ some chidlren may que changes to the parent during __init__
        but the parents init functions shoudl trigger in such an order as to capture this
        
        """
        if self.session.state == 'init': 
            logger.debug('session.state== init. doing nothing')
            return #never que updates during init
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not isinstance(req_o, weakref.ProxyType):
                raise IOError
        #=======================================================================
        # handle the updates
        #=======================================================================
        #self.upd_f = True #flag me for updating
        
        logger.debug('with upd_cmd_str \'%s\', controller \'%s\' and att_name \'%s\''%
                     (upd_cmd_str, req_o.name, att_name))
        

        
        #=======================================================================
        # reformat upd_cmd_str
        #=======================================================================
        
            
        #=======================================================================
        # #list formatting
        #=======================================================================
        if hp.basic.is_str_list(upd_cmd_str, logger=logger): #multi entry list
            upd_cmd_l = hp.basic.str_to_list(upd_cmd_str, new_type = 'str')
        
        else: 
            
            if upd_cmd_str.startswith('*'):
                """for certain commands, wed rather execute than wait for an update cycle
                raise IOError #dont want to do this anymore"""
                upd_cmd_str = upd_cmd_str[1:] #drop first character
                
                #===================================================================
                # update shortcut
                #===================================================================
                if upd_cmd_str.startswith('update('):
                    logger.debug('received \'*%s\'. forcing update now \n'%upd_cmd_str)
                    
                    #add yourself to the update que
                    self.session.update_upd_que(self)
            
                    self.execute_upd_cmd(upd_cmd_str, att_name=att_name, req_o=req_o)
                    return
                
                upd_ovr = True
                logger.debug('upd_cmd_str.beginswith(*). set upd_ovr = True')
                'todo: consider deleteing '
            
            
            upd_cmd_l = [upd_cmd_str] #single entry lists
        
        """done by teh skinny
        upd_cmd_s = set()
        for cmd_raw in upd_cmd_l:
            if not cmd_raw.endswith(')'): upd_cmd_s.update([cmd_raw+'()'])
            else: upd_cmd_s.update([cmd_raw])"""
        
        #=======================================================================
        # que the update
        #=======================================================================
        if upd_ovr: position ='head'
        else: position = 'tail'
        
        for upd_cmd in upd_cmd_l:
            'this is probably redundant with the shortcut from above'
            if upd_cmd.startswith('update('): continue #skipping these         
            self.que_upd_skinny(upd_cmd, att_name, req_o, call_func, position = position)
                
        #=======================================================================
        # #perform update now   
        #=======================================================================
        if upd_ovr:
            logger.debug('executing update() \n')
            self.update() #force the update now
        #logger.debug("finished \n")
            
        return
    
    def que_upd_skinny(self, #que an update on myself (direct)
                       upd_cmd_raw, att_name, req_o, call_func,
                       position='tail'): #where in teh que to add the command
        
        """we dont need all the formatting of the above function (that the handles do)
       
       self.que_upd_skinny( upd_cmd_raw, att_name, req_o, call_func)
        
        """
        logger = self.logger.getChild('que_upd_skinny')
        #=======================================================================
        # shortcuts
        #=======================================================================
        if str(upd_cmd_raw) in self.upd_cmd_od.keys(): 
            logger.debug('\'%s\' already qued. skipping'%upd_cmd_raw)
            if self.db_f: que_check(self)
            return
        if upd_cmd_raw is None: return
        if self.session.state == 'init': return
        
        #=======================================================================
        # defaults
        #=======================================================================
        
        
        #add yourself to the update que
        self.session.update_upd_que(self)
        
        #=======================================================================
        # formatting
        #=======================================================================
        if not upd_cmd_raw.endswith(')'): 
            cmd = upd_cmd_raw+'()'
        else: cmd = upd_cmd_raw
        
        #=======================================================================
        # add to the que
        #=======================================================================
        k, v = str(cmd), [att_name, req_o, call_func]
        if position == 'tail':
            self.upd_cmd_od[k] = v 
            
        elif position == 'head':
            self.upd_cmd_od.prepend(k, v)
        
        logger.debug('added \'%s\' to \'%s\' of upd_cmd_od: %s \n'%(cmd, position, self.upd_cmd_od.keys()))
        
        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            if not isinstance(req_o, weakref.ProxyType):
                raise IOError
            
            que_check(self)
        
        return
        
    
    
    def del_upd_cmd(self, cmd_str = None): #remove teh cmd_str from the cmd_str queue
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        handle_upd()
            when an custom object.somefunc() calls the handle_upd
            
        needs to be called as
            self.del_upd_cmd()
        
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if self.session.state == '_init_': return
        
        #=======================================================================
        # defaults
        #=======================================================================
        if cmd_str is None: cmd_str = inspect.stack()[1][3] #just pull from the last caller
        
        #reformat to command style
        if not cmd_str.endswith(')'): 
            cmd_str = cmd_str+'()'
                
        #try to remove yourself from the update queue
        try: 
            del self.upd_cmd_od[cmd_str]
            
            #logger = self.logger.getChild('del_upd_cmd')
            
            #logger.debug("removed \'%s\' from the upd_cmd_od"%cmd_str)
            
            #=======================================================================
            # que check
            #=======================================================================
            'if we couldnt even remove the command from teh que.. thenw e probably dont need to remove the obj'
            if len(self.upd_cmd_od) == 0:
                self.session.update_upd_que(self, method='delete') #remvoe yourself from the que
        except: 
            pass
            #logger = self.logger.getChild('del_upd_cmd')
            #logger.debug('failed to remove \'%s\' from teh upd_cmd_od'%cmd_str)

        return
                
    def parent_df_upd(self, att_name, new_val): #make updates to the parents df
        logger = self.logger.getChild('parent_df_upd')
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            df = self.parent.childmeta_df
            if not self.name in df.loc[:,'name'].tolist(): 
                raise IOError
            
            if not self.dfloc in df.index.tolist():
                raise IOError
            
            if not att_name in df.columns.tolist(): 
                logger.warning('passed \'%s\' not in \'%s\'s childmeta_df columns.\n %s'%
                             (att_name, self.parent.name, df.columns.tolist()))
                """ allowing now
                raise IOError"""
            """
            
            hp.pd.v(df)
            """
            
            if pd.isnull(new_val):
                raise IOError
        
        #=======================================================================
        # execute df write
        #=======================================================================
        try:
            logger.debug('updating parent \'%s\' with \'%s\' at %i'%(self.parent.name, new_val, self.dfloc))
            self.parent.childmeta_df.loc[self.dfloc, att_name] = new_val
        except:
            #===================================================================
            # error handling
            #===================================================================

            try:
                df = self.parent.childmeta_df
                if not att_name in df.columns:
                    logger.error('passed att_name \'%s\' not in the columns')
            except: 
                logger.error('something wrong with parent')
                
            if not hasattr(self, 'dfloc'):
                logger.error('I dont have a dfloc attribute')
                
            raise IOError
        
        
        
    def update(self, propagate=False):
        """ 
        #=======================================================================
        # CALLS
        #=======================================================================
        dynp.Kid.que_upd()
            run_upd_f==FALSE: this can happen during the pres session state
            
        Udev.wrap_up()
            for all objects stored during run loopd in the upd_all_d
            
        fdmg.House.run_hse()
            upd_f == TRUE
            
        fdmg.Dfunc.run_dfunc()
            upd_f == TRUE
        #=======================================================================
        # INPUTS
        #=======================================================================
        upd_cmd_od: dictionary of update commands and metadata. see Kid.que_upd
        propagate: flag to propagate your update onto your children
        
        #=======================================================================
        # TESTING
        #=======================================================================
        self.upd_cmd_od.keys()
        self.upd_cmd_od.values()
        """
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('update(%s)'%self.get_id())
        self.upd_cnt += 1
        d = copy.copy(self.upd_cmd_od) #set teh copy (as this may change throughout the loop)
        cnt = 0
        #=======================================================================
        # precheck
        #=======================================================================
        if self.db_f:
            if self.upd_cmd_od is None: raise IOError
            if len(self.upd_cmd_od) == 0: 
                logger.error('I have no commands in upd_cmd_od')
                raise IOError

            #check format of the dictionary
            if not len(self.upd_cmd_od.values()[0]) == 3:
                raise IOError
            
            que_check(self)
        
        #=======================================================================
        # loop and execute the LIVE updating commands
        #=======================================================================
        #id_str = self.get_id()
        logger.debug('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
        logger.debug('upd_cnt = %i state \'%s\' with %i cmds: %s'
                     %(self.upd_cnt, self.session.state, len(self.upd_cmd_od), self.upd_cmd_od.keys()))

                
        for cmd_raw, v in d.iteritems():
            att_name, req_o, req_str = v
            

            """ clearing upd_cmd_od alone (with some command) does not break this loop
                python seems to continue with the initial call to the dictionary"""
                
            if not cmd_raw in self.upd_cmd_od.keys():
                logger.debug('cmd \'%s\' removed from dictionary. skipping'%cmd_raw)
                #continue
            else:
                cnt +=1
                logger.debug('cnt %i executing upd_cmd() with cmd_raw \'%s\''%(cnt, cmd_raw))
            
                self.execute_upd_cmd(cmd_raw, att_name=att_name, req_o=req_o, req_str=req_str)

        logger.debug('finished %i upd_cmds (%i remain: %s)'%(cnt,len(self.upd_cmd_od),  self.upd_cmd_od.keys()))
        #=======================================================================
        # recursive udpating
        #=======================================================================
        if not len(self.upd_cmd_od) == 0: 
            pass
            #logger.debug('some items remain in the upd_cmd_od: %s'%self.upd_cmd_od.keys())
        else:
            try:
                del self.session.update_upd_que_d[self.gid] #try and remove yourself from teh update que
                logger.debug('upd_cmd_od empty. removed myself from the update_upd_que_d')
            except:
                pass
            
        
        #=======================================================================
        # post checking
        #=======================================================================
        if self.db_f: 
            #the updates dict should have been cleared
            if not len(self.upd_cmd_od) == 0: 
                if not self.gid in self.session.update_upd_que_d.keys(): raise IOError
          
        """I dont want to clear this.. what if children requeud on me?  
        self.halt_updates(req_o = self)"""
        
        if cnt > 0: return True
        else: return False

    
    def halt_update(self): #force an update halt on me
        logger = self.logger.getChild('halt_update')
        self.upd_cmd_od = OrderedDict() #empty the dictionary
        try:
            self.session.update_upd_que(self, method='delete')
        except: pass
        logger.debug('cleared my upd_cmd_od and removed myself from the que \n')
        
        if self.db_f:
            if is_dated(self, '*any'): raise IOError
        
        
    def execute_upd_cmd(self, #execute the passed command on yourself
                cmd_raw, #command to execute
                **ref_kwargs):  #reference kwargs (for display only)
        
        """
        broke this out so we can run individual update commands
        """
        logger = self.logger.getChild('execute_upd_cmd')
        
        #=======================================================================
        # reformat
        #=======================================================================
        exe_str = 'self.' + cmd_raw

        #=======================================================================
        # pre checks
        #=======================================================================
        if self.db_f:
            if not exe_str.endswith(')'): 
                raise IOError
            """this trips when we are passing kwargs
            if not hasattr(self, cmd_raw[:-2]):

                logger.error('object type \'%s\' does not have passed upd_cmd att \'%s\''
                             %(self.__class__.__name__, cmd_raw))
                raise IOError
                

            
            if not callable(eval(exe_str[:-2])):
                logger.error('object type \'%s\'s passed exe_str \'%s\' is not callable'
                             %(self.__class__.__name__, exe_str))
                                                         
                raise IOError"""

        #=======================================================================
        # execute update
        #=======================================================================

        

        logger.debug('executing  cmd_raw \'%s\' with kwargs: %s \n'
                     %(cmd_raw, ref_kwargs))
        
        """changed to flag passing
        exec(exe_str)"""
        
        result = eval(exe_str)
        
        if result:
            #try and remove the update command
            'this is often redundant on functions which use the internal handle setter'
            self.del_upd_cmd(cmd_str = cmd_raw)
        
        else:
            logger.debug('failed to execute \'%s\'. leaving in que'%cmd_raw)
            if self.db_f: que_check(self)
                
            
        if self.db_f:
            if result is None:
                raise IOError
            
        return
    
    def mypost_update(self):
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        init_dyno()    #first call before setting the OG values
        session.post_update() #called at the end of all the update loops
        
        """
        if self.post_upd_func_s is None: return False
        logger = self.logger.getChild('mypost_update')
        for func in self.post_upd_func_s:
            logger.debug('executing \'%s\''%(func))
            func() #execute the fu nction
            
        return True

        
    def dyn_kid_check(self, kids_nd):
        
        logger = self.logger.getChild('dyn_kid_check')
        
        kid1 = kids_nd.values()[0] #first kid
        #logger.debug('on %i kids with cn \'%s\''%(len(kids_nd), kid1.__class__.__name__))
    
                #unique kid check
        l = []
        for name, obj in kids_nd.iteritems():
            cn = obj.__class__.__name__
            
            if not cn in l: l.append(cn)
            
            """ not using this any more
            #check for unique run_upd_f
            if not hasattr(obj, 'run_upd_f'):
                logger.error('passed kid type \'%s\' is not a sim obj'%cn)
                raise IOError
            
            if not obj.run_upd_f == kid1.run_upd_f:
                logger.error('got non-unique run_upd_f on \'%s\''%obj.name)
                raise IOError"""
            
        if len(l) > 1:
            logger.error('got multiple kid types in passed kids_nd: %s'%l)
            raise IOError
        
        #logger.debug('cleared all')
        return
        
    def is_frozen(self, att_name, logger=None): #check if this attribute is frozen. with printouts
        """
        #=======================================================================
        # CALLS
        #=======================================================================
        custom calls should check at the beginning:
            if self.is_frozen('anchor_el', logger = logger): return
        
        dynp calls a similar function during apply_to_set()
        """
        if logger is None: logger = self.logger
        logger = logger.getChild('is_frozen')
        
        if att_name in self.fzn_an_d.keys():
                        
            req_o, req_str = self.fzn_an_d[att_name]
            
            logger.debug('attribute \'%s\' was frozen by \'%s.%s\''%(att_name, req_o.name, req_str))
            
            if self.db_f:
                if self.session.state == 'init': raise IOError
            
            return True
        else: 
            #logger.debug('\'%s\' not frozen'%att_name)
            return False
        
    
        
    def deps_is_dated(self,
                      dep_l, #list of [(dependency container, commands to search for)]
                      method = 'reque', #what to do if your dependencies are outdated
                      caller = None): #caller function
        
        """
        #=======================================================================
        # INPUTS
        #=======================================================================
        dep_l: 
            this canot be a dictionary because we are keyeing (first entry) but a group of objects sometimes
        """
        
        if self.session.state == 'init': return False
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('deps_is_dated')
        if caller is None: 
            caller = inspect.stack()[1][3]
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if not isinstance(dep_l, list): raise IOError
            
            #check contents type
            l1, _ = dep_l[0]
            if not isinstance(l1, list): raise IOError
        
        logger.debug('\'%s\' is looking on %i dependency pairs'%(caller, len(dep_l)))
        
        #=======================================================================
        # container cmd_l pairing
        #=======================================================================
        for dep_container, cmd_l in dep_l:
            #===================================================================
            # check each dependent provided
            #===================================================================
            for depend in dep_container:
                #===================================================================
                # check each command provided
                #===================================================================\
                for cmd in cmd_l:                
                    if is_dated(depend, cmd):
                        
                        #=======================================================
                        # prechecks
                        #=======================================================
                        if self.db_f:
                            que_check(depend)
                            
                        logger.debug('FOUND \'%s\' queued for \'%s\''%(cmd, depend.gid))
                        #===============================================================
                        # reque the caller
                        #===============================================================
                        if method == 'reque':                            
                            logger.debug('re-queing caller \'%s\' in my upd_cmd_od'%caller)
                            self.que_upd_skinny(caller, 'na', weakref.proxy(self), caller)

                            if self.db_f:
                                'because we have so many for loops.. ther is not a neat way to unify these'
                                pass
                                
                                
                            return True
                        
                        elif method == 'pass':
                            logger.debug('dependency \'%s\' is outdated... but we are just passing'%depend.gid)
                            return True
                            
                        elif method == 'halt':
                            
                            logger.debug("passed halt=TRUE. clearing the upd_cmd_od") 
                            self.upd_cmd_od = OrderedDict() #empty the dictionary 
                            return True
                            
                        elif method == 'force':
                            logger.debug('forcing update on depdendnt')
                            depend.update()
                            continue
                            
                        elif method == 'cascade':
                            'I dont think were using this'
                            self.session.update_all()
                            return False
                            
                        else: raise IOError
                
            else:
                pass 
                #logger.debug('no \'%s\' queud for \'%s\''%(cmd, depend.gid))
            
        logger.debug('all passed dependency pairs (%i) up to date'%len(dep_l))
        return False
                    
                    
        
        
    def depend_outdated(self,  #handle outdated dependencies
                        depend = None, #dependency to check
                        search_key_l = None, #list of keys to search for (of update commands)
                        halt=False, #whether to clean out my own update commands
                        reque = True, #whether to re-que the caller
                        force_upd = False,  #flag to force an update on the depdende
                        caller = None):  
        """
        
        """
        #=======================================================================
        # shortcuts
        #=======================================================================
        if self.session.state == 'init': return False
        
        #=======================================================================
        # defaults
        #=======================================================================
        if depend is None: depend = self.parent
        
        logger = self.logger.getChild('depend_outdated')
        outdated = False
        
        if self.db_f:
            if not search_key_l is None:
                if not isinstance(search_key_l, list): raise IOError
        #=======================================================================
        #prove it out dated
        #=======================================================================
        if len(depend.upd_cmd_od) >0:
            if not search_key_l is None: #some commands with a subset provided
                 
                for k in search_key_l:
                    if k in depend.upd_cmd_od.keys():
                        outdated = True
                        break #stop the loop
            else:
                outdated = True #some commands with no subset provided
        
        
        #=======================================================================
        # handle the outdated dependent
        #=======================================================================
        if outdated:
            logger.debug('depdendnet \"%s\' is outdated with %i upd_cmds: %s'
                         %(depend.name,  len(depend.upd_cmd_od), depend.upd_cmd_od.keys()))
            
            #===================================================================
            # reque the caller
            #===================================================================
            if reque: #add this command back intot he que
                'TODO: see execute_upd_cmd(). consider returning flags rather tahn delete/add cycles '
                if caller is None: caller = inspect.stack()[1][3]
                logger.debug('re-queing caller \'%s\' in my upd_cmd_od'%caller)
                
                self.que_upd_skinny(caller, 'na', weakref.proxy(self), caller)
                
                """NO! need to handle yourself in the que as well
                'just placing a direct entry'           
                self.upd_cmd_od[caller+'()'] = ['na', weakref.proxy(self), caller]"""

                
            
            #===================================================================
            # #halt overrid
            #===================================================================
            if halt:
                logger.debug("passed halt=TRUE. clearing the upd_cmd_od") 
                self.upd_cmd_od = OrderedDict() #empty the dictionary 
                
            #===================================================================
            # forced udpate
            #===================================================================
            if force_upd:
                logger.debug("passed force_Upd =TRUE. forcing update on depdendnt \'%s\' \n"%depend)
                depend.update()
                
            #===================================================================
            # checks
            #===================================================================
            if self.db_f:
                if self.session.state == 'init': raise IOError
                if reque:
                    if not self.gid in self.session.update_upd_que_d.keys(): 
                        logger.error('\n I was instructed to reque if my depend \'%s\' is out of date, but Im not in the update_upd_que_d'
                                     %depend.gid)
                        raise IOError
                if not force_upd:
                    if not depend.gid in self.session.update_upd_que_d.keys(): raise IOError
                #if not 'build_dfunc()' in self.upd_cmd_od.keys(): raise IOError
            
            """better to use the recursive update que
            just let the que keep looping until the depend is fully updated
            
            logger.debug('forcing update on depend \'%s\''%depend.name)
            depend.update()
            logger.debug('finished updating depend \n')
            depend.upd_cmd_od.keys()
            
            """
        
        return outdated

class Dyno_controller(object): #wrapper for controlling dynamic objects
    
    #===========================================================================
    # calcluated pars
    #===========================================================================
    update_upd_que_d = None    #container for objects needing updating
    upd_iter_cnt = 0
    
    post_updaters_wd = wdict()      #container of gids that received updates
    
    def __init__(self, *vars, **kwargs):
        logger = mod_logger.getChild('Dyno_controller') #have to use this as our own logger hasnt loaded yet
        logger.debug('start __init__ as \'%s\''%self.__class__.__name__)
        super(Dyno_controller, self).__init__(*vars, **kwargs) 
        
        self.update_upd_que_d = OrderedDict() #container for objects needing updating
        
        #=======================================================================
        # resetting
        #=======================================================================
        """ dont need these for the session
        self.reset_d.update({'update_upd_que_d':OrderedDict(), 'upd_iter_cnt':0})"""
        
        logger.debug('finished _init_ \n')
        return
    
    def update_all(self, loc='?'): #update all objects in the queue
        """
        old_state = self.state
        self.state = '%s.update'%old_state"""

        start = time.time()
        #=======================================================================
        # shortcuts
        #=======================================================================
        if len(self.update_upd_que_d) == 0: return
        #=======================================================================
        # defaults
        #=======================================================================
        logger = self.logger.getChild('update_all')
        
        logger.info("\n uauauauauauauauauauauauauauauauauauauauauauauauauauauauauauauauuauauauauauauauauauauauauauauauauauauauau")
        logger.info('at \'%s\' with %i objects in que'%(loc, len(self.update_upd_que_d )))
        
        self.update_iter()
        
        """using static wd
        if len(self.post_updaters_wd) > 0:
            'we often dont update any objects with post update commands'"""
            
        """moved this to get_res_tstep()
            then we are only calling it after model changes, but before running the outputrs
        logger.debug("executing post_update()")
        self.post_update()"""
        
        #=======================================================================
        # wrap up
        #=======================================================================
        stop = time.time()
        logger.info('finished in %.4f secs with %i scans'%(stop - start, self.upd_iter_cnt))
        logger.debug('\n')
        
        self.upd_iter_cnt = 0 #reset this
        """
        self.state = old_state #return the state"""
        
        #=======================================================================
        # post checks
        #=======================================================================
        if self.db_f:
            if len(self.update_upd_que_d) > 0: raise IOError
        

        return
        
    def update_iter(self): # a single update iteration
            
        logger = self.logger.getChild('update_iter')
        self.upd_iter_cnt +=1
        this_cnt = int(self.upd_iter_cnt)
        #logger.info('upd_iter_cnt: %i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i%i')
        logger.debug('for upd_iter_cnt %i executing on %i objects in que \n'
                     %(self.upd_iter_cnt, len(self.update_upd_que_d)))
        
        
        
        d_copy = copy.copy(OrderedDict(sorted(self.update_upd_que_d.items(), key=lambda t: t[0])))
        """setting a sorted copy here 
            so no changes made druing the update commands affect the original que
            this update top down (low branch_level -> high) to the lowest"""
        
        #if self.db_f:
        if self.upd_iter_cnt > 10: 
            logger.error('stuck in a loop with %i objs queued \n %s'%(len(d_copy), d_copy.keys()))
            
            raise IOError
        #=======================================================================
        # loop and update
        #=======================================================================
        cnt = 0
        for k, obj in d_copy.iteritems():
            cnt+=1
            if cnt%self.session._logstep == 0: logger.info('    (%i/%i)'%(cnt, len(d_copy)))
            
            if not obj.gid in self.update_upd_que_d.keys():
                'some siblings can pull each other out of the que'
                logger.debug('\'%s\' has been removed from teh que. skipping'%obj.gid)
                continue
            
            logger.debug('updating \'%s\''%(k))
            
            _ = obj.update()
            """using a static update_wd
            if obj.update():
                if not obj.post_upd_func_s is None: #only que those objects with post funcs
                    self.post_updaters_wd[obj.gid] = obj #append this gid to the list of objects updated"""
            
            
            """ objects update() should do this
            del self.update_upd_que_d[k] #remove this from teh que"""
            
        logger.debug('finished iteration on %i objects \n'%cnt)
        
        if len(self.update_upd_que_d) > 0:
            logger.info('after scan %i, %i remain in que (vs original %i). repeating'
                         %(self.upd_iter_cnt, len(self.update_upd_que_d), len(d_copy)))
            self.update_iter()
            
            logger.debug('closing loop %i'%(this_cnt +1))
            
        else:
            logger.debug('update_upd_que_d emptied after %i iters'%self.upd_iter_cnt)
            
        return
    
    def post_update(self): #run through all updated objects and execute any post/stats commands
        """
        only statistic commands should be executed here
            those that only outputers rely on... do not influence the simulation model
        
        these are excueted on all those objects with receved update() = TRUE during the update iteration
        
        
        
        #=======================================================================
        # PURPOSE
        #=======================================================================
        This allows for only a subset of objects to run some post stats calc functions
            where objects can add/remove themselves to this que based on their own properties
        
        Allows for stats to be calcualted on objects NOT in run loops
        
        #=======================================================================
        # Calls
        #=======================================================================
        Tstep.run_dt()
            Tstep.get_res_tstep() #during wrap up
        
        """
        logger = self.logger.getChild('post_update')
        
        """objects are added to this during __init_dyno_ if they have any post_upd_func_s"""
        d = self.post_updaters_wd
        
        #=======================================================================
        # prechecks
        #=======================================================================
        if self.db_f:
            if len(d) == 0:
                raise IOError
            
        #=======================================================================
        # loop and update all the objects
        #=======================================================================
        logger.debug('executing on %i objects in the post_updaters_wd \n'%len(d))
        for gid, obj in d.iteritems(): 
            if not obj.mypost_update(): raise IOError

                
        #=======================================================================
        # wrap up
        #=======================================================================
        logger.debug('finished \n')
        
        """letting this carry over
        self.post_updaters_wd = wdict() #clear this"""
        
        return
        """
        d.keys()
        """
        
        
    def update_upd_que(self, obj, method='add'): #add the object to the que
        """ should be able to update dictionary directly
        """
        logger = self.logger.getChild('update_upd_que')
        #=======================================================================
        # addition to the library
        #=======================================================================
        if method == 'add':
            self.update_upd_que_d[obj.gid] = weakref.proxy(obj)
            logger.debug('added \'%s\' to the \'update_upd_que_d\' (%i)'%(obj.gid, len(self.update_upd_que_d)))
            
        #=======================================================================
        # deletions from the library
        #=======================================================================
        elif method == 'delete':
            if len(self.update_upd_que_d) == 0: return
            try:
                del self.update_upd_que_d[obj.gid]
                logger.debug('deleted \'%s\' from the \'update_upd_que_d\' (%i)'%(obj.gid, len(self.update_upd_que_d)))
            except:
                #if self.session.state == 'update':
                
                logger.debug('unable to remove \'%s\' from teh update_upd_que'%obj.gid)
                raise IOError
                
        else: raise IOError
        
        return
                     
                     
def is_dated(obj, cmd):
        """
        match = hp.basic.list_search(obj.upd_cmd_d.keys(), cmd)"""
        #=======================================================================
        # any updates
        #=======================================================================
        if cmd == '*any':
            if len(obj.upd_cmd_od) > 0: return True
            
        #=======================================================================
        # specific updates
        #=======================================================================
        for e in obj.upd_cmd_od.keys():
            if re.search(cmd, e, re.IGNORECASE):                
                return True
            
        return False
        
def que_check(obj, logger = mod_logger): #raise errors if this object is not properly queued
    if not len(obj.upd_cmd_od) >0:
        logger = logger.getChild('que_check') 
        logger.error('\'%s\' doesnt have any commands queud on itself'%obj.gid)
        raise IOError
    
    if not obj.gid in obj.session.update_upd_que_d.keys(): 
        logger = logger.getChild('que_check') 
        logger.error('\'%s\' is not in the update que'%obj.gid)
        raise IOError
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        