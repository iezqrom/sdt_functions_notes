import pandas as pd
from scipy.stats import norm
import math
import numpy as np
Z = norm.ppf

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
    """
        Compute d', response criterion, hit and false alarm rate, beta and Ad'
        by correcting extreme values(ceiling/floor).
        This code was adapted from an existing repository, which I can't find at the moment. (April, 2021)
    """
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

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = (Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate
    
    return(out)


def SDTloglinear(hits, misses, fas, crs):
    """ 
        Compute d', response criterion, hit and false alarm rate, beta and Ad'
        by adding 0.5 to both the number ofhits and the number offalse alarms and adding 1 to both the number of signal trials and the number ofnoise trials.
        Adapted from Stanislaw and Todorov, 1999.
    """
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
        Non-parametric SDT metric.
        Original equation: Pollack & Norman, 1979
        Adapted: Stanislaw and Todorov, 1999
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