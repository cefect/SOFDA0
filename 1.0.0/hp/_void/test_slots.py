'''
Created on Sep 7, 2018

@author: cef
'''
#import hp.oop

from itertools import chain
from inspect import isclass
import weakref

class Child(object):#foundational object 
    #__slots__ = ()    
    
    c   = 'some default c'
    
    def __init__(self, *vars, **kwargs):
        #self.__slots__ = 'b'
        print 'initilizing Child'
        
        super(Child, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
          
        try:
            print('Child got these slots: %s'%self.__slots__)
        except:
            print('Child doesnt have slots')
          
        return
    
class Child2(object):#foundational object 
    __slots__ = ()
    
    d   = 'some_default d' #this attribute will be read only a nd shared
    pass
    #__slots__ = 'c'
    
    def __init__(self, *vars, **kwargs):
        
        #self.tester = 'tester'
        
        print 'initilizing Child2'
        
        super(Child2, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
         
        try:
            print('Child2 got these slots: %s'%self.__slots__)
        except:
            print('Child2 doesnt have slots')
         
        return
    

        

class Dummy1(Child, Child2):
    pass
    #__slots__ = ('a', 'b', '__weakref__')
    
    __slots__ = ('a', 'b',)
    
    def __init__(self, *vars, **kwargs):
        print 'initilizing Dummy1'
         
        super(Dummy1, self).__init__(*vars, **kwargs) #initilzie teh baseclass 
        

            
        if not hasattr(self, '__dict__'):
            print ('no dict... locked obj')
        else:
            print ('have a dict... open obj')
         
        return



o1 = Dummy1()

print o1.d

try:
    o1.d = 'changeme'
except:
    print 'att \'d\' s locked as \'%s\''%o1.d


o1.b = 'test'
o1.a = 'test'

try:
    o1.anything = 'test'
except:
    print ('failed to set a new att.. obj must be locked')

try:
    print ('__dict__: %s'%o1.__dict__.keys())
except:
    print 'no dict'

import hp.oop

slots_d = hp.oop.get_slots(o1)


print 'finished with all slots: %s'%list(set(slots_d.values()))

try:
    po1 = weakref.proxy(o1)
    print 'made a proxy'
except:
    print ('failed to make a proxy')


print 'finished'



