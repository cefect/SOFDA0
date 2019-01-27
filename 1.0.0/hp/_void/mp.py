
import pandas as pd
import numpy as np
import multiprocessing as mp



def parallelize(pool, df, func, threads, **kwargs):
    """ from: http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    
    #===========================================================================
    # WARNING
    #===========================================================================
    could not get this to work as desired within my standard main/scripts architecture
    
    """


    split_dfs = np.array_split(df, threads) #split the df into threads dfs
    
    
    df = pd.concat(pool.map(func, split_dfs))
    pool.close()
    pool.join()
    return df


output = mp.Queue()

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]