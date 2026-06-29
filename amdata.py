from amread import *
import numpy as np
import pytensor.tensor as pt

class AM_data():
    inv4pi = 1/(4*np.pi)
    def __init__(self, atomic_transitions_list: list[tuple[int, int]], mol_transition = "fulcher", **kwargs):
        

        trans_to_keep = []
        for i, trans in enumerate(atomic_transitions_list):
            if trans[0] > 6:
                print(f"discarding line {trans} (Upper state not found in AMJUEL)")
                continue
            else:
                trans_to_keep.append(trans)
        self.atomic_transitions_list = trans_to_keep
        self.mol_transition = mol_transition

        self.A_coeffs = np.array([A_coeff(trans) for trans in self.atomic_transitions_list])
        self.cs = [[photon_rate_coeffs(trans[0])] for trans in self.atomic_transitions_list]
        self.M = len(self.atomic_transitions_list)
        
        # Molecules
        self.h_neg = kwargs.get("h_neg", False)
        self.h3pos = kwargs.get("h3pos", False)
        self.den_cs = []
        self.den_cs.append(read_amjuel_2d("H.12", "2.0c"))
        if self.h3pos: self.den_cs += read_amjuel_2d("H.11", "4.0a")
        if self.h_neg: self.den_cs += read_amjuel_2d("H.11", "7.0a")


        # Assign photon_rate function based on included reactions
        # Do it this way to avoid repeated evaluation of if clauses in the evaluation of photon rates
        if self.h3pos and self.h_neg:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v3
        elif self.h_neg:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v2
        elif self.h3pos:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v1
        else:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v0
        
        self.cs = [self.photon_rate_coeffs(trans[0], h_neg= self.h_neg,h3_pos= self.h3pos) for trans in self.atomic_transitions_list]

        if self.mol_transition is not None:
            h_name, collisionName, acoeff = H2_reactions(band=self.mol_transition)
            self.mol_A_coeff = acoeff
            #self.atomic_transitions_list += self.mol_transition
            #self.M = len(self.atomic_transitions_list)
            self.cs.append( [read_amjuel_2d(h_name, collisionName)])
            
        
    def photon_rate_coeffs(self,n, **kwargs):
        h_neg = kwargs.get("h_neg", False)
        h3pos = kwargs.get("h3pos", False)

        reac = reactions(n)
        marcs = []

        marcs += [read_amjuel_2d(reac["atomic_exc"][0], reac["atomic_exc"][1])]
        marcs += [read_amjuel_2d(reac["atomic_rec"][0], reac["atomic_rec"][1])]
        marcs += [read_amjuel_2d(reac["H2"][0],reac["H2"][1])]
        marcs += [read_amjuel_2d(reac["H2+"][0],reac["H2+"][1])]
        if h3pos: marcs += [read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])]
        if h_neg: marcs += [read_amjuel_2d(reac["H-"][0],reac["H-"][1])]

        return marcs

    def calc_photon_rates_mol_v0(self, te, ne, nh, nh2, **kwargs):
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6

        nh2pos = calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        #nh3pos = calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        #nhneg  = calc_cross_sections(self.den_cs[2], te, ne_)*nh2 
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][2], te, ne_)*nh2*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][3], te, ne_)*nh2pos*self.inv4pi
            #res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[4], te, ne_)*nh3pos*self.inv4pi
            #res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[5], te, ne_)*nhneg*self.inv4pi
        

        return res
    
    def calc_photon_rates_mol_v1(self, te, ne, nh, nh2, **kwargs):
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6

        nh2pos = calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        nh3pos = calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        #nhneg  = calc_cross_sections(self.den_cs[2], te, ne_)*nh2 
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][2], te, ne_)*nh2*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][3], te, ne_)*nh2pos*self.inv4pi
            res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[4], te, ne_)*nh3pos*self.inv4pi
            #res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[5], te, ne_)*nhneg*self.inv4pi
        

        return res
    
    def calc_photon_rates_mol_v2(self, te, ne, nh, nh2, **kwargs):
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6

        nh2pos = calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        #nh3pos = calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        nhneg  = calc_cross_sections(self.den_cs[2], te, ne_)*nh2 
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][2], te, ne_)*nh2*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][3], te, ne_)*nh2pos*self.inv4pi
            #res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[4], te, ne_)*nh3pos*self.inv4pi
            res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[5], te, ne_)*nhneg*self.inv4pi
        

        return res
    
    def calc_photon_rates_mol_v3(self, te, ne, nh, nh2, **kwargs):
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6

        nh2pos = calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        nh3pos = calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        nhneg  = calc_cross_sections(self.den_cs[2], te, ne_)*nh2 
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][2], te, ne_)*nh2*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][3], te, ne_)*nh2pos*self.inv4pi
            res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[4], te, ne_)*nh3pos*self.inv4pi
            res[i, N] += self.A_coeffs[i]*calc_cross_sections(self.cs[5], te, ne_)*nhneg*self.inv4pi
        

        return res

    def calc_mol_band_photon_rate(self, te, ne, nh, nh2):

        return self.mol_A_coeff*calc_cross_sections(self.cs[-1][0], te, ne*1e-6)*nh2*self.inv4pi
    
    def calc_photon_rates_no_mol(self, te, ne, nh, nh2, **kwargs):
        '''
        Include only excitation and recombination
        
        '''
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
        return res
    

class AM_data_pt():
    inv4pi = 1/(4*np.pi)
    @staticmethod
    def calc_cross_sections(MARc, T=None, n=None, E=None):
        MARc = pt.as_tensor_variable(MARc)
        logT = pt.log(T)

        if MARc.ndim == 2:

            nE = n / 1e8 if n is not None else E
            lognE = pt.log(nE)

            powers_T = pt.arange(9)
            powers_n = pt.arange(9)

            T_terms = logT[None, :] ** powers_T[:, None]
            n_terms = lognE[None, :] ** powers_n[:, None]

            cross_sections = (
                MARc[:, :, None]
                * T_terms[:, None, :]
                * n_terms[None, :, :]
            )
            return pt.exp(pt.sum(cross_sections, axis=(0, 1)))
        else:

            powers_T = pt.arange(9)
            cross_sections = (
                MARc[:, None]
                * logT[None, :] ** powers_T[:, None]
            )

            return pt.exp(pt.sum(cross_sections, axis=0))
        
    def __init__(self, atomic_transitions_list: list[tuple[int, int]], mol_transition = "fulcher", **kwargs):
        trans_to_keep = []
        for i, trans in enumerate(atomic_transitions_list):
            if trans[0] > 6:
                print(f"discarding line {trans} (Upper state not found in AMJUEL)")
                continue
            else:
                trans_to_keep.append(trans)
        self.atomic_transitions_list = trans_to_keep
        self.mol_transition = mol_transition

        self.A_coeffs = np.array([A_coeff(trans) for trans in self.atomic_transitions_list])
        self.cs = [[photon_rate_coeffs(trans[0])] for trans in self.atomic_transitions_list]
        self.M = len(self.atomic_transitions_list)
        
        # Molecules
        self.h_neg = kwargs.get("h_neg", False)
        self.h3pos = kwargs.get("h3pos", False)
        self.den_cs = []
        self.den_cs.append(read_amjuel_2d("H.12", "2.0c"))
        if self.h3pos: self.den_cs += read_amjuel_2d("H.11", "4.0a")
        if self.h_neg: self.den_cs += read_amjuel_2d("H.11", "7.0a")


        # Assign photon_rate function based on included reactions
        # Do it this way to avoid repeated evaluation of if clauses in the evaluation of photon rates
        if self.h3pos and self.h_neg:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v3
        elif self.h_neg:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v2
        elif self.h3pos:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v1
        else:
            self.calc_photon_rates_mol = self.calc_photon_rates_mol_v0
        
        self.cs = [self.photon_rate_coeffs(trans[0], h_neg= self.h_neg,h3_pos= self.h3pos) for trans in self.atomic_transitions_list]

        if self.mol_transition is not None:
            h_name, collisionName, acoeff = H2_reactions(band=self.mol_transition)
            self.mol_A_coeff = acoeff
            #self.atomic_transitions_list += self.mol_transition
            #self.M = len(self.atomic_transitions_list)
            self.cs.append( [read_amjuel_2d(h_name, collisionName)])
            
        
    def photon_rate_coeffs(self,n, **kwargs):
        h_neg = kwargs.get("h_neg", False)
        h3pos = kwargs.get("h3pos", False)

        reac = reactions(n)
        marcs = []

        marcs += [read_amjuel_2d(reac["atomic_exc"][0], reac["atomic_exc"][1])]
        marcs += [read_amjuel_2d(reac["atomic_rec"][0], reac["atomic_rec"][1])]
        marcs += [read_amjuel_2d(reac["H2"][0],reac["H2"][1])]
        marcs += [read_amjuel_2d(reac["H2+"][0],reac["H2+"][1])]
        if h3pos: marcs += [read_amjuel_2d(reac["H3+"][0],reac["H3+"][1])]
        if h_neg: marcs += [read_amjuel_2d(reac["H-"][0],reac["H-"][1])]

        return marcs

    def calc_photon_rates_mol_v0(self, te, ne, nh, nh2):
        ne_ = ne * 1e-6
        nh2pos = self.calc_cross_sections(self.den_cs[0], te, ne_) * nh2

        rows = []

        for i in range(self.M):
            row = (
                self.A_coeffs[i]
                * (   self.calc_cross_sections(self.cs[i][0], te, ne_) * nh
                    + self.calc_cross_sections(self.cs[i][1], te, ne_) * ne
                    + self.calc_cross_sections(self.cs[i][2], te, ne_) * nh2
                    + self.calc_cross_sections(self.cs[i][3], te, ne_) * nh2pos
                )* self.inv4pi
            )

            rows.append(row)

        return pt.stack(rows)
    
    def calc_photon_rates_mol_v1(self, te, ne, nh, nh2):
        ne_ = ne * 1e-6
        nh2pos = self.calc_cross_sections(self.den_cs[0], te, ne_) * nh2
        nh3pos = self.calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        rows = []

        for i in range(self.M):
            row = (
                self.A_coeffs[i]
                * (   self.calc_cross_sections(self.cs[i][0], te, ne_) * nh
                    + self.calc_cross_sections(self.cs[i][1], te, ne_) * ne
                    + self.calc_cross_sections(self.cs[i][2], te, ne_) * nh2
                    + self.calc_cross_sections(self.cs[i][3], te, ne_) * nh2pos
                    + self.calc_cross_sections(self.cs[i][4], te, ne_) * nh3pos
                )* self.inv4pi
            )

            rows.append(row)

        return pt.stack(rows)
    
    def calc_photon_rates_mol_v2(self, te, ne, nh, nh2):
        ne_ = ne * 1e-6
        nh2pos = self.calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        nhneg = self.calc_cross_sections(self.den_cs[2], te, ne_)*nh2
        rows = []

        for i in range(self.M):
            row = (
                self.A_coeffs[i]
                * (   self.calc_cross_sections(self.cs[i][0], te, ne_) * nh
                    + self.calc_cross_sections(self.cs[i][1], te, ne_) * ne
                    + self.calc_cross_sections(self.cs[i][2], te, ne_) * nh2
                    + self.calc_cross_sections(self.cs[i][3], te, ne_) * nh2pos
                    + self.calc_cross_sections(self.cs[i][5], te, ne_) * nhneg
                )* self.inv4pi
            )

            rows.append(row)

        return pt.stack(rows)
    
    def calc_photon_rates_mol_v2(self, te, ne, nh, nh2):
        ne_ = ne * 1e-6
        nh2pos = self.calc_cross_sections(self.den_cs[0], te, ne_)*nh2
        nhneg = self.calc_cross_sections(self.den_cs[2], te, ne_)*nh2
        nh3pos = self.calc_cross_sections(self.den_cs[1], te, ne_)*nh2*nh2pos/ne
        rows = []

        for i in range(self.M):
            row = (
                self.A_coeffs[i]
                * (   self.calc_cross_sections(self.cs[i][0], te, ne_) * nh
                    + self.calc_cross_sections(self.cs[i][1], te, ne_) * ne
                    + self.calc_cross_sections(self.cs[i][2], te, ne_) * nh2
                    + self.calc_cross_sections(self.cs[i][3], te, ne_) * nh2pos
                    + self.calc_cross_sections(self.cs[i][4], te, ne_) * nh3pos
                    + self.calc_cross_sections(self.cs[i][5], te, ne_) * nhneg
                )* self.inv4pi
            )

            rows.append(row)

        return pt.stack(rows)
    def calc_mol_band_photon_rate(self, te, ne, nh, nh2):

        return self.mol_A_coeff*calc_cross_sections(self.cs[-1][0], te, ne*1e-6)*nh2*self.inv4pi
    
    def calc_photon_rates_no_mol(self, te, ne, nh, nh2, **kwargs):
        '''
        Include only excitation and recombination
        
        '''
        N = te.shape[0] #assume flattened array
        res = np.zeros((self.M, N))
        ne_ = ne*1e-6
        
        for i in range(self.M):
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][0], te, ne_)*nh*self.inv4pi
            res[i, :] += self.A_coeffs[i]*calc_cross_sections(self.cs[i][1], te, ne_)*ne*self.inv4pi
        return res
    