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


def tableTosdtDoble(table, num_sdt):
    table_single_sdt = table.loc[table['Touch'] == num_sdt]

    table_cold = table_single_sdt.loc[table_single_sdt['Cold'] == 1]
    table_nocold = table_single_sdt.loc[table_single_sdt['Cold'] == 0]

    present_yes = table_cold.loc[table_cold['Responses'] == 1]
    present_no = table_cold.loc[table_cold['Responses'] == 0]
     
    absent_yes = table_nocold.loc[table_nocold['Responses'] == 1]
    absent_no = table_nocold.loc[table_nocold['Responses'] == 0]

    return present_yes, present_no, absent_yes, absent_no


def SDTextremes(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    # print(fa_rate)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa

    # print(hit_rate)
    # print(fa_rate)
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate) # Hint: normalise the centre of each curvey and subtract them (find the distance between the normalised centre
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = (Z(hit_rate) + Z(fa_rate)) / 2 # Hint: like d prime but you add the centres instead, find the negative value and half it
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate
    
    return(out)


def SDTloglinear(hits, misses, fas, crs):
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Calculate hit_rate and avoid d' infinity
    hits += 0.5
    hit_rate = hits / (hits + misses + 1)

    # Calculate false alarm rate and avoid d' infinity
    fas += 0.5
    fa_rate = fas / (fas + crs + 1)

    # print(hit_rate)
    # print(fa_rate)

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate) # Hint: normalise the centre of each curvey and subtract them (find the distance between the normalised centre
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = (Z(hit_rate) + Z(fa_rate)) / 2 # Hint: like d prime but you add the centres instead, find the negative value and half it
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate
    
    return(out)

def SDTAprime(hits, misses, fas, crs):
    """ 
        Original equation: Pollack & Norman, 1979
        Adapted: Stanislaw and Todorov
    """
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)

    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)

    # print(hit_rate)
    # print(fa_rate)

    # Return d', beta, c and Ad'
    out = {}
    # out['Aprime'] = 1 - (1/4) * ( (fa_rate/hit_rate) + ((1-hit_rate)/(1-fa_rate))) # pollack & norman
    out['Aprime'] = 0.5 + (np.sign(hit_rate - fa_rate) * (((hit_rate - fa_rate)**2 + abs(hit_rate - fa_rate))/(4*max(hit_rate, fa_rate) - 4*hit_rate*fa_rate)))  # adapted 
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate
    
    return(out)