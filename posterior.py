import numpy as np
from numpy import log, exp, pi, sqrt, sin, cos, tan, arctan
from scipy.optimize import leastsq, curve_fit
import matplotlib.pyplot as plt
import pickle
import pdb
import corner
import emcee
import q2
import copy
import multiprocessing as mp


class Posterior(q2.Star):
    """This class is similar to q2.Star, except all parameters and
    abundances are replaced by np.ndarray objects with one MCMC step
    result per row.
    """
    def __init__(self, StarObj):
        q2.Star.__init__(self)
        self.name = StarObj.name
        self.linelist = StarObj.linelist
        
    def save_step(self, param):
        """Save parameters to the posterior object."""
        for i,attr in enumerate(['teff','logg','feh','vt']):
            new = param[i]
            saved = getattr(self, attr, None)
            if saved is None:
                setattr(self, attr, new)
            else:
                saved = np.vstack((saved, new))
                setattr(self, attr, saved)
        return True
        
    def calc_ab(self, modatm='odfnew', species_ids=None, ref=None):
        if species_ids == None:
            species_codes = sorted(set(self.linelist['species']))
            species_ids = q2.abundances.getsp_ids(species_codes)
        # loop through all of the parameters:
        try:
            all_par = zip(self.teff, self.logg, self.feh, self.vt)
        except AttributeError:
            print "cannot calculate abundances without stellar parameters"
            return False
        for param in all_par:
            # do the MOOG calculation:
            star2 = copy.copy(self)
            star2.teff = param[0][0] # figure out a better way to do this later
            star2.logg = param[1][0]
            star2.feh = param[2][0]
            star2.vt = param[3][0]
            star2.get_model_atmosphere(modatm)
            q2.abundances.one(star2, Ref=ref, species_ids=species_ids, silent=True)
            for species_id in species_ids:
                new = getattr(star2, species_id)
                saved = getattr(self, species_id, None)
                if saved is None:
                    setattr(self, species_id, new)
                else:
                    saved['ab'] = np.vstack((saved['ab'],new['ab']))
                    saved['difab'] = np.vstack((saved['difab'],new['difab']))
        return True
    
    def calc_isochrone(self, isochrone_db='yy01.sql3', feh_offset = 0):
        # set up isochrone solver:
        sp = q2.isopars.SolvePars()
        sp.db = isochrone_db
        sp.key_parameter_known = 'logg'
        sp.feh_offset = feh_offset  # optional offset to the isochrone tracks
        pp = q2.isopars.PlotPars()
        pp.make_figures = False   # don't generate any figures
        # set errors:
        star2 = copy.copy(self)
        star2.err_teff = np.std(self.teff)
        star2.err_logg = np.std(self.logg)
        star2.err_feh = np.std(self.feh)
        star2.err_vt = np.std(self.vt)
        # loop through all of the parameters:
        try:
            all_par = zip(self.teff, self.logg, self.feh, self.vt)
        except AttributeError:
            print "cannot fit isochrone without stellar parameters"
            return False
        for param in all_par:
            # do the isochrone fitting:
            star2.teff = param[0][0] # figure out a better way to do this later
            star2.logg = param[1][0]
            star2.feh = param[2][0]
            star2.vt = param[3][0]
            q2.isopars.solve_one(star2, sp, PlotPars=pp, silent=True)
            for output in ('isomass', 'isor', 'isoage', 'isomv', 'isologl'):
                new = getattr(star2, output)['most_probable']
                saved = getattr(self, output, None)
                if saved is None:
                    setattr(self, output, new)
                else:
                    saved = np.vstack((saved,new))
                    setattr(self, output, saved)
        return True



def make_posterior(star, sampler, n_burn=None, n_thin=None):
    """Takes a q2 Star object and an emcee sampler object
    Returns a Posterior object with fundamental parameters.
    """
    if n_burn == None:
        n_burn = np.shape(sampler.chain)[1]/10  # default burn-in: 10% of chain length
    if n_thin == None:
        n_thin = round(np.max(sampler.acor)) # default thinning: autocorrelation length
    samples = sampler.chain[:, n_burn:, :].reshape((-1, 4))
    post = Posterior(star)
    for param in samples[::n_thin,:]:
        post.save_step(param)
    return post
    
def print_abundance(post, species_ids=None, print_ab=True, print_difab=False):
    """Takes a Posterior object and prints statistics on elemental abundance(s)
    with errors."""
    if species_ids == None:
        species_codes = sorted(set(post.linelist['species']))
        species_ids = q2.abundances.getsp_ids(species_codes)
    for sp in species_ids:
        try:
            ab = getattr(post, sp)['ab']
        except AttributeError:
            print "abundances have not been calculated for species {0}".format(sp)
            continue
        if print_ab:
            sp_ab = np.mean(ab, axis=1)  # species abundance at each step is the average of the lines
            x = np.percentile(sp_ab, [16, 50, 84])
            err_correct = np.sqrt(len(ab)*1.0)  # standard error on the mean
            print "{0}/H: {1:.3f} + {2:.3f} - {3:.3f}".format(sp, \
                x[1], (x[2]-x[1])/err_correct, (x[1]-x[0])/err_correct)
        if print_difab:
            try:
                diff = getattr(post, sp)['difab']
                line_check = np.all(diff != np.array(None), axis=0) # check which lines have ref missing
                diff = diff[:,line_check] # remove any bad lines
                sp_diff = np.mean(diff, axis=1)
                x = np.percentile(sp_diff, [16, 50, 84])
                err_correct = np.sqrt(np.shape(diff)[1])  # standard error on the mean
                print "[{0}/H]: {1:.3f} + {2:.3f} - {3:.3f}".format(sp, \
                    x[1], (x[2]-x[1])/err_correct, (x[1]-x[0])/err_correct)
            except:
                print "No differential abundance available for {0}".format(sp)
    return True
    
def linear(x, m, b):
     model = m*x + b
     return model
    
def tc_trend(abund, err=None, species_ids=[0], Tc=[0]):
    """Takes a set of abundances and outputs [X/H] vs. T_condensation 
    best-fit param (slope, intercept) as an array."""
    if len(species_ids) != len(abund) and len(Tc) != len(abund):
        print "Must input either species_ids or Tc for all elements to be fit"
        return
    if len(Tc) <= 1:
        # fetch condensation temperatures for the elements to be fit
        Tc = []
        for species_id in species_ids:
            Tc = np.append(Tc, q2.abundances.gettc(species_id))
    if err == None:
        popt, pcov = curve_fit(linear, Tc, abund) # fit without errors
    else:
        popt, pcov = curve_fit(linear, Tc, abund, sigma=err, absolute_sigma=True) # fit
    return popt # (slope, intercept)
    
    
def tc_bootstrap(post, trials=10000, species_ids=None, Ref_age=4.6, OI_override=[0.0,0.0]):
    """Takes a Posterior object with abundances AND isochrones, does a Grand Bootstrap*,
    and adds the posterior distribution of Tc trend parameters to Posterior object.
    
    * where Grand Bootstrap is defined as randomizing over stellar parameters + resulting 
    age/abundance and galactic chemical evolution correction 
    factors simultaneously."""
    if species_ids == None:
        species_codes = sorted(set(post.linelist['species']))
        species_ids = q2.abundances.getsp_ids(species_codes)
    if getattr(post, species_ids[0])['ref'] != 'Sun' and Ref_age == 4.6:
        print "WARNING: assuming that the reference star is solar age!"
        print "If this is not true, set the Ref_age keyword accordingly."
    tc_fit = {'age':np.empty(trials), 'slope':np.empty(trials), 'intercept':np.empty(trials)} # set up dict to store results
    rand_ind = np.random.choice(len(post.isoage),size=trials) # random steps of posterior
    for i,j in zip(range(trials),rand_ind):
        age = post.isoage[j][0]
        if age == None:
            continue
        abund = []
        Tc = []
        err = []
        # generate GCE-corrected abundances for this step in posterior:
        for species_id in species_ids:
            if species_id == 'KI':
                continue
            # get elemental abundance
            difab_all = getattr(post,species_id)['difab'][j]
            difab_all = difab_all[difab_all != np.array(None)]
            abund = np.append(abund, np.mean(difab_all)) # average over lines
            err = np.append(err, np.std(difab_all)/sqrt(len(difab_all))) # standard error on mean
            Tc = np.append(Tc, q2.abundances.gettc(species_id))
            if (species_id == 'OI' and OI_override != [0.0,0.0]): # manually input NLTE-corrected oxygen
                o_mean, o_sig = OI_override
                abund[-1] = np.random.normal(o_mean, o_sig) # a lil randomization
                err[-1] = o_sig
            # apply GCE correction
            (b, err_b) = q2.gce.getb_linear(species_id)
            if b != 0.0: #check that there is a correction available
                rand_b = np.random.normal(b,err_b) # assumes Gaussian error on GCE correction
                abund[-1] -= (age - Ref_age)*rand_b # adjust abundance
        for t in set(Tc):
            # eliminate duplicate measurements of the same element
            ind = np.where(Tc == t)[0]
            if len(ind) == 2:  # won't take care of 3+ states of the same thing
                (abund[ind[0]], err[ind[0]]) = np.average(abund[ind], weights=err[ind], returned=True)
                abund = np.delete(abund, ind[1])
                err = np.delete(err, ind[1])
                Tc = np.delete(Tc, ind[1])
                species_ids = np.delete(species_ids, ind[1])
        # fit the trend
        popt = tc_trend(abund, err=err, Tc=Tc)
        # append result to tc_fit
        tc_fit['age'][i] = age
        tc_fit['slope'][i] = popt[0]
        tc_fit['intercept'][i] = popt[1]
    # save the results to the posterior object
    post.tc_fit = tc_fit
    x = np.percentile(post.tc_fit['slope'], [16, 50, 84])
    print "Tc slope = {0:.2e} + {1:.2e} - {2:.2e}".format(x[1], (x[2]-x[1]), (x[1]-x[0]))
    return True
