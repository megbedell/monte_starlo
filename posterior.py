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
        
    def save_step(self, param, modatm='odfnew', species_ids=None, ref=None):
        if species_ids == None:
            species_codes = sorted(set(self.linelist['species']))
            species_ids = q2.abundances.getsp_ids(species_codes)
        # do the MOOG calculation:
        star2 = copy.copy(self)
        [star2.teff, star2.logg, star2.feh, star2.vt] = param
        star2.get_model_atmosphere(modatm)
        q2.abundances.one(star2, Ref=ref, species_ids=species_ids, silent=True)
        # save parameters:
        for attr in ['teff','logg','feh','vt']:
            new = getattr(star2, attr)
            saved = getattr(self, attr, None)
            if saved is None:
                setattr(self, attr, new)
            else:
                saved = np.vstack((saved, new))
                setattr(self, attr, saved)
        # save abundances:
        for species_id in species_ids:
            new = getattr(star2, species_id)
            saved = getattr(self, species_id, None)
            if saved is None:
                setattr(self, species_id, new)
            else:
                saved['ab'] = np.vstack((saved['ab'],new['ab']))
                saved['difab'] = np.vstack((saved['difab'],new['difab']))
        return True


def make_posterior(star, sampler, modatm='odfnew', ref=None, n_burn=None, n_thin=None):
    """Takes a q2 Star object and an emcee sampler object
    Returns a Posterior object.
    """
    if n_burn == None:
        n_burn = np.shape(sampler.chain)[1]/10  # default burn-in: 10% of chain length
    if n_thin == None:
        n_thin = round(np.max(sampler.acor)) # default thinning: autocorrelation length
    samples = sampler.chain[:, n_burn:, :].reshape((-1, 4))
    p = Posterior(star)
    for param in samples[::n_thin,:]:
        p.save_step(param, ref=ref, modatm=modatm)
    return p
    