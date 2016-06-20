import sys
sys.path.insert(1,'/home/mbedell/python')
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
import mcmc
import posterior
import time

if __name__ == "__main__":

    starname = 'K11'
    refname = 'Sun'
    modatm = 'odfnew'  # choose the model grid
    data = q2.Data('K11_solution.csv', 'K11_lines.csv')
    
    # set up star objects:
    star = q2.Star(starname)
    ref = q2.Star(refname)
    star.get_data_from(data)
    ref.get_data_from(data)

    # solve for best-fit parameters:
    sp = q2.specpars.SolvePars()
    sp.step_teff = 4
    sp.step_logg = 0.04
    sp.step_vt = 0.04
    sp.niter = 100
    sp.grid = modatm
    sp.errors = True
    q2.specpars.solve_one(star, sp, Ref=ref)

    print "Best-fit parameters:"
    print "Teff  = {0:5.0f}, logg = {1:5.2f}, [Fe/H] = {2:5.3f}, vt = {3:5.2f}".format(star.teff, star.logg, star.feh, star.vt)

    # estimate errors from best-fit parameters:
    star.get_model_atmosphere(modatm)
    ref.get_model_atmosphere(modatm)
    q2.specpars.iron_stats(star, Ref=ref)

    print "A(Fe I)  = {0:5.3f} +/- {1:5.3f}".\
       format(star.iron_stats['afe1'], star.iron_stats['err_afe1'])
    print "A(Fe II) = {0:5.3f} +/- {1:5.3f}".\
      format(star.iron_stats['afe2'], star.iron_stats['err_afe2'])
    print "A(FeI) vs. EP slope  = {0:.4f} +/- {1:.4f}".\
      format(star.iron_stats['slope_ep'], star.iron_stats['err_slope_ep'])
    print "A(FeI) vs. REW slope = {0:.4f} +/- {1:.4f}".\
      format(star.iron_stats['slope_rew'], star.iron_stats['err_slope_rew'])
      
    sig_Sep = np.copy(star.iron_stats['err_slope_ep'])
    sig_Srew = np.copy(star.iron_stats['err_slope_rew'])
    sig_delFe = np.sqrt(star.iron_stats['err_afe1']**2/len(star.fe1['ab']) + star.iron_stats['err_afe2']**2/len(star.fe2['ab']))
    sig_Fe = star.iron_stats['err_afe1']/np.sqrt(len(star.fe1['ab']))
    errors = np.array([sig_Sep, sig_Srew, sig_delFe, sig_Fe])
    
    # run mcmc:
    print "Starting MCMC..."
    start_time = time.time()
    start_theta = np.array([star.teff, star.logg, star.feh, star.vt])
    jump_theta = np.array([8.0,0.04,0.03,0.04])
    n_theta = len(start_theta)
    n_walkers = 10
    n_steps = 10000
    verbose = False
    pos = [start_theta + np.random.randn(n_theta)*jump_theta for i in range(n_walkers)]
    sampler = emcee.EnsembleSampler(n_walkers, n_theta, mcmc.lnprob, args=[errors, star, ref], kwargs={"modatm":modatm, "verbose":verbose}, threads=2)
    sampler.run_mcmc(pos, n_steps)
    plt.clf()
    print 'MCMC took {0:.2f} minutes'.format((time.time()-start_time)/60.0)
    print 'auto-correlation lengths:', sampler.acor
    
    pickle.dump(sampler.chain,open('emceechain_{0:s}-{1:s}.p'.format(starname, refname), 'wb'))
    pickle.dump(sampler.lnprobability,open('emceelike_{0:s}-{1:s}.p'.format(starname, refname), 'wb'))

    post = posterior.make_posterior(star,sampler)
    
    # save abundances:
    print "Calculating abundances..."
    start_time = time.time()
    p.calc_ab(modatm=modatm,ref=ref)
    print 'Abundances took {0:.2f} minutes'.format((time.time()-start_time)/60.0)
    
    # save isochrones:
    print "Calculating isochrones..."
    start_time = time.time()
    p.calc_isochrone(feh_offset=-0.04) # offset improves solar age & mass values
    print 'Isochrones took {0:.2f} minutes'.format((time.time()-start_time)/60.0)
    

    pickle.dump(post,open('posterior_{0:s}-{1:s}.p'.format(starname, refname), 'wb'))
    
    n_burn = np.shape(sampler.chain)[1]/10  # default burn-in: 10% of chain length
    samples = sampler.chain[:, n_burn:, :].reshape((-1, 4))
    figure = corner.corner(samples, labels=["T$_{eff}$","log(g)","[Fe/H]","$v_t$"], show_titles=True)
    figure.savefig('pairsplot_{0:s}-{1:s}.png'.format(starname, refname))
    figure.clear()
    
    pdb.set_trace()
  
