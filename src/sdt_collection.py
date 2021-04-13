import pandas as pd
from scipy.stats import norm
import math
import numpy as np
Z = norm.ppf

def sdt_setup(n_trials, conds):
    """
        Function to set-up trials for a Signal Detection Theory experiment.
        The number of trials will be distributed equally across the number of conditions.
        For instance, if we have 1 condition (e.g. cold stimulation with touch), 
        and you set n_trials = 10, there'll be 10 trials of this condition.
        However, if we have 2 conditions (e.g. cold stimulation with and without touch),
        and you set n_trials = 10, there'll be 5 trials of each condition.
        Stimulus absent and present are coded with 0s (absent) and 1s (present), respectively.
        Conditions are coded from 0-n.
        Condition and stimulus absent/present come in tuples.
    """

    stimulations = []

    if not n_trials % (2*conds) == 0:
        printme(f'Number of trials is not divisable by {2*conds}')
        if not n_trials % 2 == 0:
            printme(f'Number of trials is an odd number')
        printme('WARNING: Uneven number of conditions')
        code_conds = np.arange(conds)
        n_cond_trials = n_trials/conds

        n_conds = np.repeat(code_conds, n_cond_trials, axis = 0)
        unique, counts = np.unique(n_conds, return_counts=True)
        print(counts)

        for u, c in zip(unique, counts):
            abs_pres = np.repeat([0, 1], c, axis = 0)

            for ap in abs_pres:
                stimulations.append((u, ap))

        np.random.shuffle(stimulations)
        stimulations = stimulations[:n_trials]

    else:
        code_conds = np.arange(conds)
        n_cond_trials = n_trials/conds

        n_conds = np.repeat(code_conds, n_cond_trials, axis = 0)
        unique, counts = np.unique(n_conds, return_counts=True)
        # print(counts)
        
        for u, c in zip(unique, counts):
            abs_pres = np.repeat([0, 1], c/2, axis = 0)
            
            for ap in abs_pres:
                stimulations.append((u, ap))

        np.random.shuffle(stimulations)


    return stimulations