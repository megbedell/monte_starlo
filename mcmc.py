import numpy as np
from numpy import log, exp, pi, sqrt, sin, cos, tan, arctan
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pickle
import pdb
import corner
import emcee
import q2
import copy
import multiprocessing as mp


def lnprior(theta,modatm='odfnew',verbose=False):
    teff,logg,feh,micro = theta
    if (modatm == 'odfnew'):
        if (teff > 7500 or teff < 3500 or logg < 0.0 or logg > 5.0 or micro < 0.0 or feh < -2.5 or feh > 0.5):  # interpolation fails
	    if verbose:
                print "parameters out of range!"
	    return -np.inf
    return 0.0

def lnlike(theta,errors,star,ref,modatm='odfnew',verbose=False):
    sig_Sep, sig_Srew, sig_delFe, sig_Fe = errors

    # make a test case "star2" with theta parameters and retrieve its iron stats

    star2 = copy.copy(star)  # could try deepcopy but I don't think it's needed
    [star2.teff, star2.logg, star2.feh, star2.vt] = theta
    star2.get_model_atmosphere(modatm)

    if verbose:    
	    print "Teff  = {0:5.0f}, logg = {1:5.2f}, [Fe/H] = {2:5.3f}, vt = {3:5.2f}".format(star2.teff, star2.logg, star2.feh, star2.vt)

    q2.specpars.iron_stats(star2, Ref=ref)
  
    delFe = star2.iron_stats['afe1'] - star2.iron_stats['afe2']
    Fe_inout = star2.feh - star2.iron_stats['afe1']
    #alpha_inout = alpha - np.mean(diff_abund[ind_alpha])

    #calculate likelihood for theta

    lnlike = (-1.0/2.0) * (star2.iron_stats['slope_ep']**2/sig_Sep**2 + log(2.*pi*sig_Sep**2) + star2.iron_stats['slope_rew']**2/sig_Srew**2 + log(2.*pi*sig_Srew**2))
    lnlike += (-1.0/2.0) * (delFe**2/sig_delFe**2 + log(2.*pi*sig_delFe**2))
    lnlike += (-1.0/2.0) * (Fe_inout**2/sig_Fe**2 + log(2.*pi*sig_Fe**2))   # + alpha_inout**2/sig_alpha**2 + log(2.*pi*sig_alpha**2))
     
    if verbose:
    	print "[Fe/H]_model - A(FeI)  = {0:5.3f} +/- {1:5.3f}".\
       		format(Fe_inout, sig_Fe)
    	print "A(FeI) - A(FeII) = {0:5.3f} +/- {1:5.3f}".\
      		format(delFe, sig_delFe)
    	print "A(FeI) vs. EP slope  = {0:.4f} +/- {1:.4f}".\
      		format(star2.iron_stats['slope_ep'], sig_Sep)
    	print "A(FeI) vs. REW slope = {0:.4f} +/- {1:.4f}".\
      		format(star2.iron_stats['slope_rew'], sig_Srew)
    
    	print "lnlike = {0:3.1f}".format(lnlike)
    	print "-----------------------------------------------------------------------"
    return lnlike
    
def lnprob(theta, errors, star, ref, modatm='odfnew', verbose=False):
    lp = lnprior(theta,modatm=modatm, verbose=verbose)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, errors, star, ref, modatm=modatm, verbose=verbose)
      
    
   
