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

if __name__ == "__main__":

    starname = 'XO-2N'
    refname = 'Sun'
    modatm = 'odfnew'  # choose the model grid
    verbose = False

    data = q2.Data('XO2_stars.csv', 'XO2_lines.csv')
    star = q2.Star(starname)
    ref = q2.Star(refname)

    star.get_data_from(data)
    ref.get_data_from(data)

    # solve for best-fit parameters:

    sp = q2.specpars.SolvePars()
    sp.step_teff = 8
    sp.step_logg = 0.08
    sp.step_vt = 0.08
    sp.niter = 35
    sp.grid = 'odfnew'

    q2.specpars.solve_one(star, sp, Ref=ref)

    print "Best-fit parameters:"
    print "Teff  = {0:5.0f}, logg = {1:5.2f}, [Fe/H] = {2:5.3f}, vt = {3:5.2f}".format(star.teff, star.logg, star.feh, star.vt)

    # estimate errors from best-fit parameters:
    star.get_model_atmosphere(modatm)
    ref.get_model_atmosphere(modatm)
    q2.specpars.iron_stats(star, Ref=ref)
    #alpha_species = ['MgI','SiI','CaI','TiI','TiII']
    #q2.abundances.one(star, species_ids=alpha_species, Ref=ref) 
    #alpha_difab = np.concatenate((star.MgI['difab'],star.SiI['difab'],star.CaI['difab'],star.TiI['difab'],star.TiII['difab']))  # there's got to be a more elegant way to do this...

    #pdb.set_trace()

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
    #sig_alpha = 0
    #errors = np.array([sig_Sep, sig_Srew, sig_delFe, sig_Fe, sig_alpha])
    errors = np.array([sig_Sep, sig_Srew, sig_delFe, sig_Fe])


    
    # run mcmc:
    start_theta = np.array([star.teff, star.logg, star.feh, star.vt])
    jump_theta = np.array([8.0,0.08,0.05,0.08])
    n_theta = len(start_theta)
    nwalkers = 10
    pos = [start_theta + np.random.randn(n_theta)*jump_theta for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, n_theta, mcmc.lnprob, args=[errors, star, ref], kwargs={"modatm":modatm, "verbose":verbose}, threads=2)
    sampler.run_mcmc(pos, 10)

    plt.clf()
    #plt.plot(sampler.chain[0,:,0])
    #plt.plot(sampler.chain[4,:,0])
    #plt.plot(sampler.chain[9,:,0])
    #plt.show()

    pdb.set_trace()
    
    pickle.dump(sampler.chain,open('chainresults_XO2N-Sun.p', 'wb'))
    
    n_burn = 500
    samples = sampler.chain[:, n_burn:, :].reshape((-1, n_theta))
    #figure = triangle.corner(samples, labels=["T$_{eff}$","log(g)","[Fe/H]","$v_t$","[$\alpha$/H]"], show_titles=True)
    figure = corner.corner(samples, labels=["T$_{eff}$","log(g)","[Fe/H]","$v_t$"], show_titles=True)
    figure.savefig("pairsplot_XO2N-Sun.png")
    figure.clear()
  