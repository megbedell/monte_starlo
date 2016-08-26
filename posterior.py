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
    
def abundance_err(post, species_ids=None, difab=False):
    """Takes a Posterior object and prints statistics on abundances."""
    if species_ids == None:
        species_codes = sorted(set(post.linelist['species']))
        species_ids = q2.abundances.getsp_ids(species_codes)
    for sp in species_ids:
        try:
            ab = getattr(post, sp)['ab']
        except AttributeError:
            print "abundances have not been calculated for species {0}".format(sp)
            continue
        sp_ab = np.mean(ab, axis=1)  # species abundance at each step is the average of the lines
        x = np.percentile(sp_ab, [16, 50, 84])
        err_correct = 1.0  # edit this later to reflect number of lines used
        print "{0}/H: {1:.3f} + {2:.3f} - {3:.3f}".format(sp, x[1], (x[2]-x[1])/err_correct, (x[1]-x[0])/err_correct)
        if difab:
            diff = getattr(post, sp)['difab']
            sp_diff = np.mean(diff, axis=1)
            x = np.percentile(sp_diff, [16, 50, 84])
            print "[{0}/H]: {1:.3f} + {2:.3f} - {3:.3f}".format(sp, x[1], (x[2]-x[1])/err_correct, (x[1]-x[0])/err_correct)
    return True
