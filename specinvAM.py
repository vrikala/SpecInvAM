from amread import *
from amdata import *
from adasdata import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pymc
import arviz as az
import pytensor.tensor as pt
az.style.use("arviz-variat")
from scipy.optimize import least_squares, minimize, direct
import sys, os



class SpecInvAM():

    def __init__(self, atomic_emission_list: list[np.ndarray[np.float64]],atomic_transitions_list: list[tuple[int, int]], mol_emission: np.ndarray = None, mol_transition = "fulcher", **kwargs):
        assert len(atomic_emission_list)==len(atomic_transitions_list), "emission list must have the same length as transition list"
        
        self.N = atomic_emission_list[0].shape[0]
        self.atomic_emission_dict = {trans: atomic_emission_list[i] for i, trans in enumerate(atomic_transitions_list)}
        self.atomic_emission_arr = np.vstack(atomic_emission_list)
        if mol_emission is None:
            print("No molecular emission provided, disabling molecular contributions")
            self.mol_emission_dict = {}
            self.mol_contrib = False
            self.mol_transition = None
        else:
            self.mol_emission_dict = {mol_transition: mol_emission}
            self.mol_emission_arr = mol_emission
            self.mol_transition = mol_transition
            self.mol_contrib = True

        
        # handle kwargs
        self.h_neg = kwargs.get("h_neg", False)
        self.h3_pos = kwargs.get("h3_pos", False)
        self.mol_contrib = kwargs.get("force_mol_contrib", self.mol_contrib) # A flag to force on molecular part

        self.AM_data = AM_data(atomic_transitions_list, mol_transition, h_neg = self.h_neg, h3_pos = self.h3_pos)
        self.AM_data_pt = AM_data_pt(atomic_transitions_list, mol_transition, h_neg = self.h_neg, h3_pos = self.h3_pos)

        self.adasdata = ADF(atomic_transitions_list, discard_low_N = True) # use ADAS for N>=7

    def run_MCMC(self, **kwargs):
        """
        Perfroms Markov Chain Monte Carlo (MCMC) on the measured spectrum

        params:
            data_uncertainty [float]: variance for each data point for a priori resampling of the data.
                                      Setting this to e.g. 0.05 gives variance 0.05*measured emission
            samples_n [int]: number of a priori samples to draw. If this is <= 0, use the data directly
            model [pyMC Model]: pyMC model so that you have a handle to it outside this function
            num_cores [int]: number of cores to use in MCMC
            num_chains [int]: number of chains to run in MCMC

            te_distribution [pyMC distribution]: default uniform. Must have size = N, where N is the length of the data
            ne_distribution [pyMC distribution]: see te_distribution
            nh_distribution [pyMC distribution]: see te_distribution
            nh2_distribution [pyMC distribution]: see te_distribution

        """
        data_uncertainty = kwargs.get("data_uncertainty", 0.05)
        model = kwargs.get("model", pymc.Model())
        samples_n = kwargs.get("samples", 20)
        num_mc_draws = kwargs.get("mc_draws", 4000)
        num_cores = kwargs.get("cores", 4)
        num_chains = kwargs.get("chains", 4)

        if (self.mol_contrib and self.mol_transition is not None):
            am_mcmc = self.AM_MCMC # Fulcher constrained
            data = np.vstack((self.atomic_emission_arr, self.mol_emission_arr))
        elif self.mol_contrib:
            am_mcmc = self.AM_MCMC_nm # Only atomic constrained
            data = self.atomic_emission_arr
        else:
            am_mcmc = self.AM_MCMC_atomic # Only atomic constrained only exc. + rec.
            data = self.atomic_emission_arr
        
        if samples_n>0:
            data = data.flatten()
            data_cov = np.diag((data_uncertainty*data)**2)
            samples = np.random.multivariate_normal(data, data_cov, samples_n)
        else:
            # TODO
            data
            data_cov = np.cov(data)
            samples = data


        with model:
            te = kwargs.get("te_distribution", pymc.Uniform("te", np.log10(0.5), np.log10(60), size = self.N))
            ne= kwargs.get("ne_distribution",pymc.Uniform("ne", 17, 22, size = self.N))
            nh = kwargs.get("nh_distribution",pymc.Uniform("nh", 16, 22, size = self.N))
            nh2 = kwargs.get("nh2_distribution",pymc.Uniform("nh2", 15, 22, size = self.N))

            obs = pymc.MvNormal("em", mu = am_mcmc(te, ne, nh, nh2), cov = data_cov, observed=samples)

            idata = pymc.sample(cores = num_cores, chains = num_chains, draws = num_mc_draws)

        return  idata

    def AM_MCMC(self, te, ne, nh, nh2):

        te_ = pt.power(10, te)
        ne_ = pt.power(10, ne)
        nh_ = pt.power(10, nh)
        nh2_ = pt.power(10, nh2)

        res_a = self.AM_data_pt.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_a_adas = self.adasdata.calc_photon_rates_pt(te_, ne_, nh_)
        res_mol = self.AM_data_pt.calc_mol_band_photon_rate(te_, ne_, nh_, nh2_)

        return pt.concatenate([
            pt.flatten(res_a),
            pt.flatten(res_a_adas),
            pt.flatten(res_mol)
        ])
    
    def AM_MCMC_nm(self, te, ne, nh, nh2):

        te_ = pt.power(10, te)
        ne_ = pt.power(10, ne)
        nh_ = pt.power(10, nh)
        nh2_ = pt.power(10, nh2)

        res_a = self.AM_data_pt.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_a_adas = self.adasdata.calc_photon_rates_pt(te_, ne_, nh_)
        return pt.concatenate([
            pt.flatten(res_a),
            pt.flatten(res_a_adas)
        ])
    
    def AM_MCMC_atomic(self, te, ne, nh, nh2):

        te_ = pt.power(10, te)
        ne_ = pt.power(10, ne)
        nh_ = pt.power(10, nh)
        nh2_ = pt.power(10, nh2)

        
        res_a = self.AM_data_pt.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_a_adas = self.adasdata.calc_photon_rates_pt(te_, ne_, nh_)
        return pt.concatenate([
            pt.flatten(res_a),
            pt.flatten(res_a_adas)
        ])
        

    def residual(self, p):
        p_ = 10**p
        te_ = p_[:self.N]
        ne_ = p_[self.N:self.N*2]
        nh_ = p_[self.N*2:self.N*3]
        nh2_ = None if not self.mol_contrib else p_[self.N*3:self.N*4]

        res = [em- calc_photon_rate(trans, te_, ne_, nh_, mol_n_density = nh2_, h3 = self.h3_pos, h_neg = self.h_neg) for trans, em in self.atomic_emission_dict.items()]
        if self.mol_contrib:
            em, _ =calc_H2_band_emission(te_, ne_, nh2_, band = self.mol_transition)
            res += [self.mol_emission_dict[self.mol_transition] -em]
        return np.array(res).flatten()
    
    def AM_residual(self, p):
        p_ = 10**p
        te_ = p_[:self.N]
        ne_ = p_[self.N:self.N*2]
        nh_ = p_[self.N*2:self.N*3]
        nh2_ = p_[self.N*3:self.N*4]

        res_a = self.atomic_emission_arr - self.AM_data.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_mol = self.mol_emission_dict[self.mol_transition] - self.AM_data.calc_mol_band_photon_rate(te_, ne_, nh_, nh2_)

        return np.vstack((res_a, res_mol)).flatten()

    def AM_em_brute(self, p_):

        te_ = 10**p_[0]
        ne_ = 10**p_[1]
        nh_ = 10**p_[2]
        nh2_ = 10**p_[3]

        res_a = self.AM_data.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_mol = self.AM_data.calc_mol_band_photon_rate(te_, ne_, nh_, nh2_)

        return np.vstack((res_a, res_mol))

    def AM_minimum(self, p):
        p_ = 10**p
        te_ = p_[:self.N]
        ne_ = p_[self.N:self.N*2]
        nh_ = p_[self.N*2:self.N*3]
        nh2_ = p_[self.N*3:self.N*4]

        res_a = self.atomic_emission_arr - self.AM_data.calc_photon_rates_mol(te_, ne_, nh_, nh2_)
        res_mol = self.mol_emission_dict[self.mol_transition] - self.AM_data.calc_mol_band_photon_rate(te_, ne_, nh_, nh2_)

        res_flat = np.vstack((res_a, res_mol)).flatten()
        return np.sqrt(np.dot(res_flat, res_flat))
    
    def lsq(self, **kwargs):
        '''
        A simple least-squares solver for Te, ne, nh, and nh2. Does not provide error estimate or statistics
        
        '''
        gtol = kwargs.get("gtol", 1e-8); ftol = kwargs.get("ftol", 1e-8); xtol = kwargs.get("xtol", 1e-8); verbose = kwargs.get("verbose", 0)

        N_vec = np.ones((self.N,))

        Te_bound_l = 0.5 * N_vec
        ne_bound_l = 1e16 * N_vec
        nh_bound_l = 1e14 * N_vec
        nh2_bound_l = 1e13 * N_vec
        Te_bound_u = 60.0 * N_vec
        ne_bound_u = 1e22 * N_vec
        nh_bound_u = 1e22 * N_vec
        nh2_bound_u = 1e22 * N_vec

        p0 = np.log10(np.concatenate((Te_bound_u, ne_bound_l, nh_bound_l, nh2_bound_l)))
        bounds = [
            np.log10(np.concatenate((Te_bound_l, ne_bound_l, nh_bound_l, nh2_bound_l))),
            np.log10(np.concatenate((Te_bound_u, ne_bound_u, nh_bound_u, nh2_bound_u)))
        ]

        res = least_squares(self.AM_residual, p0, bounds=bounds, gtol = gtol, ftol = ftol, xtol = xtol, verbose = verbose)

        return res
    
    def brute_lsq(self, **kwargs):
        '''
        Brute candidate minima, then feed into a
        A simple least-squares solver for Te, ne, nh, and nh2. Does not provide error estimate or statistics
        
        '''
        tes = np.log10(np.linspace(1.0, 60, 51, dtype= np.float32))
        nes = np.linspace(17, 21, 51, dtype= np.float32)
        hs = np.linspace(17, 21, 51, dtype= np.float32)
        h2s = np.linspace(17, 21, 51, dtype= np.float32)
        T, N, H, H2 = np.meshgrid(tes, nes, hs, h2s)
        
        Tf = T.flatten()
        Nf = N.flatten()
        Hf = H.flatten()
        H2f = H2.flatten()
        p = [Tf, Nf , Hf, H2f]

        datapoints = self.atomic_emission_arr.T#[self.atomic_emission_arr[i, :] for i in self.atomic_emission_arr.shape]
        te0 = np.zeros((self.N,))
        ne0 = np.zeros((self.N,))
        nh0 = np.zeros((self.N,))
        nh20 = np.zeros((self.N,))
        r = self.AM_em_brute(p)
        for i in range(self.N):
            emis = np.concatenate((datapoints[i, :], np.array([self.mol_emission_dict[self.mol_transition][i]])))
            r2 = np.sum((emis[:, None] - r)**2, axis=0)

            min_idx = np.argmin(r2)
            te0[i] = Tf[min_idx]
            ne0[i] = Nf[min_idx]
            nh0[i] = Hf[min_idx]
            nh20[i] = H2f[min_idx]

        p0 = np.concatenate((te0, ne0, nh0, nh20))

        gtol = kwargs.get("gtol", 1e-8); ftol = kwargs.get("ftol", 1e-8); xtol = kwargs.get("xtol", 1e-8); verbose = kwargs.get("verbose", 0)

        N_vec = np.ones((self.N,))

        Te_bound_l = 0.5 * N_vec
        ne_bound_l = 1e16 * N_vec
        nh_bound_l = 1e14 * N_vec
        nh2_bound_l = 1e13 * N_vec
        Te_bound_u = 60.0 * N_vec
        ne_bound_u = 1e22 * N_vec
        nh_bound_u = 1e22 * N_vec
        nh2_bound_u = 1e22 * N_vec

        
        bounds = [
            np.log10(np.concatenate((Te_bound_l, ne_bound_l, nh_bound_l, nh2_bound_l))),
            np.log10(np.concatenate((Te_bound_u, ne_bound_u, nh_bound_u, nh2_bound_u)))
        ]

        res = least_squares(self.AM_residual, p0, bounds=bounds, gtol = gtol, ftol = ftol, xtol = xtol, verbose = verbose)

        return res
    
    def direct(self, **kwargs):
        '''
        Utilizes the DIRECT solver for Te, ne, nh, and nh2. Does not provide error estimate or statistics
        
        '''
        N_vec = np.ones((self.N,))

        Te_bound_l = 0.5 * N_vec
        ne_bound_l = 1e16 * N_vec
        nh_bound_l = 1e14 * N_vec
        nh2_bound_l = 1e13 * N_vec
        Te_bound_u = 60.0 * N_vec
        ne_bound_u = 1e22 * N_vec
        nh_bound_u = 1e22 * N_vec
        nh2_bound_u = 1e22 * N_vec

        bounds = (
            np.log10(np.concatenate((Te_bound_l, ne_bound_l, nh_bound_l, nh2_bound_l))),
            np.log10(np.concatenate((Te_bound_u, ne_bound_u, nh_bound_u, nh2_bound_u)))
        )
        bounds = list(zip(*bounds))

        res = direct(self.AM_minimum, bounds=bounds)

        return res

if __name__ == "__main__":


    '''
    n = 10
    Te = np.random.uniform(5, 30, n)

    ne = np.random.uniform(1e18, 5e20, n)

    nh = np.random.uniform(1e17, 1e20, n)

    nh2 = np.random.uniform(1e16, 1e20, n)


    transitions = [(3,2), (5, 2), (6,2)]
    em = [calc_photon_rate(trans, Te, ne, nh, mol_n_density = nh2) for trans in transitions]
    fu, _ = calc_H2_band_emission(Te, ne, nh2)
    spec = SpecInvAM(em, transitions, fu)

    #res = spec.lsq()
    res = spec.brute_lsq()

    r = 10**res.x.reshape((4, n))
    r_true = np.vstack((Te, ne, nh, nh2))
    print(np.abs(r-r_true)/r_true)
    '''
    
    n = 10
    Te = np.random.uniform(5, 30, n)

    ne = np.random.uniform(1e18, 5e20, n)

    nh = np.random.uniform(1e17, 1e20, n)

    nh2 = np.random.uniform(1e16, 1e20, n)

    r_true = np.vstack((Te, ne, nh, nh2))
    am_transitions = [(3,2), (5, 2)]#, (6,2)]
    em = [calc_photon_rate(trans, Te, ne, nh, mol_n_density = nh2) for trans in am_transitions]
    adf = ADF([(7,2)])
    em72 = adf.calc_photon_rates(Te, ne, nh)
    em += [em72]
    fu, _ = calc_H2_band_emission(Te, ne, nh2)
    spec = SpecInvAM(em, am_transitions + [(7,2)], mol_transition=None, force_mol_contrib = True)

    #res = spec.lsq()
    model = pymc.Model()
    res = spec.run_MCMC(model = model)
    az.plot_trace_dist(res)
    az.plot_rank_dist(res)
    az.summary(res, kind="diagnostics")
    plt.show()