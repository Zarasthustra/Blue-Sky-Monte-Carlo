# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:09:10 2019

@author: Zarathustra
"""

"""
Blue Sky Monte Carlo

A.K.A. BSMC
"""

import numpy as np
import numba as nb
import time

#from KolafaNezbeda import ULJ, PressureLJ, FreeEnergyLJ_res
###############################################################################
#
#                    Simulation code for MC-NVT monatomic LJ
#                            Written by Braden Kelly
#                                 November 24, 2019
#
###############################################################################


###############################################################################
#
#            Simulation Variables (as few as possible)
#
###############################################################################

number_of_atoms = 400
number_of_atom_types = 1 #not actually used, this code is for monatomic LJ
atomType = "Ar"
epsilon = 1.0
sigma   = 1.0
boxSize = 8.93
cutOff = boxSize / 2
temperature = 1.0
nEquilSteps = 100000
outputInterval=2000

###############################################################################
#
#       Some Class definitions
#
###############################################################################

class System():
    
    def __init__(self,number_of_atoms, epsilon, sigma, boxSize, temp, cutOff,atomType):
        self.natoms = number_of_atoms
        self.atomType = atomType
        self.eps    = epsilon
        self.sig    = sigma
        self.boxSize = boxSize
        self.volume  = boxSize ** 3
        self.rho     = self.natoms / self.volume
        self.pressure = 0.0
        self.setPressure = None
        self.energy = 0.0
        self.virial = 0.0
        self.temp = temp
        self.rCut = cutOff
        self.rCut_sq = cutOff ** 2
        self.beta = 1.0 / temp
        self.positions = np.zeros((self.natoms,3),dtype=np.float_)
        self.velocities = None
        self.forces = None

    def GenerateRandomBox(self):
        self.positions = np.random.rand(self.natoms,3) * self.boxSize
     
    def GetPressure(self,virial):
        return self.rho / self.beta  + virial / ( 3.0 * self.volume )
    
    def PressureTailCorrection(self):
        return  16.0 / 3.0 * np.pi * self.rho **2 * self.sig **3 * self.eps * ( (2.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )
    def EnergyTailCorrection(self):
        return  8.0 / 3.0 * np.pi * self.rho * self.natoms * self.sig **3 * self.eps * ( (1.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )
    def ChemPotTailCorrection(self):
        """ beta * mu_corr = 2 * u_corr """
        return 16.0 / 3.0 * np.pi * self.rho * self.sig **3 * self.eps * ( (1.0/3.0)*(self.sig / self.rCut)**9 - (self.sig / self.rCut)**3 )
    def TotalEnergy(self):
        self.energy,self.virial = totalEnergy(self.positions, self.boxSize, \
                                              self.rCut_sq, self.natoms, sig=self.sig, eps=self.eps)


@nb.njit
def totalEnergy(r,box,r_cut_box_sq,n,sig=1,eps=1):
    potential = 0.0
    virial = 0.0
    for i in range(n-1): # Outer loop over atoms
        for j in range(i+1,n): # Inner loop over atoms
            rij = r[i,:] - r[j,:]       # Separation vector
            rij = rij - np.rint ( rij / box  ) * box # Periodic boundary conditions in box=1 units
            rij_sq = np.sum ( rij**2 )  # Squared separation

            if rij_sq < r_cut_box_sq: # Check within cutoff

                sr2    = sig / rij_sq    # (sigma/rij)**2
                sr6  = sr2 ** 3
                sr12 = sr6 ** 2
                pot  = eps * (sr12 - sr6)        # LJ pair potential (cut but not shifted)
                vir  = eps * (2.0 * sr12 - sr6) 
                
                potential += pot
                virial += vir
                
    return potential*4.0, virial*24.0
                
""" THis is outside class since Numba has issues with compiling methods """
@nb.njit #(nb.int64,nb.float64[:],nb.float64,nb.float64,nb.float64[:,:],nb.float64[:,:], nb.float64,nb.float64, nb.float64)       
def updateEnergies(ri, rj, box, r_cut_box_sq, eps, sig):
    rsq = 0.0
    potential = 0.0
    virial = 0.0

    """Calculate chosen particle's potential energy with rest of system """

    #for index, rj in enumerate(r):
    for index in range(len(rj) ):

        rij = ri - rj[index] #rj            # Separation vector
        rij = rij - np.rint(rij / box ) * box # Mirror Image Seperation

        rsq = np.sum(rij**2)  # Squared separation
        if rsq < r_cut_box_sq: # Check within cutoff

            sr2    = sig / rsq    # (sigma/rij)**2
            sr6  = sr2 ** 3
            sr12 = sr6 ** 2
            pot  = eps * (sr12 - sr6)        # LJ pair potential (cut but not shifted)
            vir  = eps * (2.0 * sr12 - sr6)                    # LJ pair virial
            
            potential += pot
            virial += vir

                            
    return 4*potential, 24*virial
       
class MCSample():
    def __init__(self,rho=0.0,pressure=0.0,virial=0.0,energy=0.0,at=0.5, \
                 av=0.5, ar=0.5, asw=0.5,output=20,dr_max=0.1 \
                 ):
        self.rho = rho
        self.pressure = pressure
        self.virial = virial
        self.energy = energy
        self.acceptTranslation = at
        self.acceptVolume = av
        self.acceptReaction = ar
        self.acceptSwap = asw
        self.nSamples = 0
        self.outputInterval = output
        self.updateInterval = output
        self.dr_max = dr_max

              
class MC_NVT(MCSample):

    def __init__(self, steps, system, runType='Blank',rho=0.0, pressure=0.0, \
                 virial=0.0, energy=0.0, at=0.5, av=0.5, ar=0.5, asw=0.5, \
                 output=20000,dr_max=0.1):
        MCSample.__init__(self,rho=rho,pressure=pressure,virial=virial, \
                          energy=energy,at=at,av=av,ar=ar,asw=asw, \
                          output=output)
        self.nSteps = steps
        self.system = system
        self.runType = runType
        

        print('Beginning {0:s} for {1:d} steps at {2:f} temperature'.format(
                self.runType, self.nSteps, self.system.temp))
        self.InitializeSimulation()
        r = self.system.positions               # simpler notation

        #######################################################################
        #
        #                          NVT Simulation
        #
        #######################################################################        
        acceptedMoves = 0
        steps = 0
        while steps < self.nSteps:
            steps += 1
            part =self.PickParticle()
            rj = np.delete(r,part,0) # Array of all the other atoms
            ri = self.MoveParticle(self.dr_max, r[part,:])
            old_potential, old_virial = self.system.energy, self.system.virial
            new_potential, new_virial = self.UpdateEnergies(ri,rj)
            TEST = self.Metropolis( self.system.beta* (new_potential - old_potential)  )
            
            if TEST:
                r[part,:] = ri
                self.system.energy += new_potential - old_potential
                self.system.virial += new_virial - old_virial
                acceptedMoves += 1
                
            if steps % self.outputInterval == 0:
                self.Sample()
            if steps % self.updateInterval == 0:
                self.UpdateMaxMove()
                #PrintPDB(self.system, steps ,"during_")
            
        self.system.positions = r
        #######################################################################   
    def UpdateMaxMove(self):
        pass
        
    def InitializeSimulation(self):
        """
        Calculate pairwise energy and virial for entire system
        """
        self.system.TotalEnergy()

                   
    def Sample(self):
        
        self.nSamples += 1
        self.pressure += self.system.GetPressure(self.system.virial)
        self.pressure += self.system.PressureTailCorrection()
        self.virial   += self.system.virial
        self.energy   += self.system.energy
        
        print('Pressure: {:6.4f}, Energy: {:6.4f}, Samples: {:6d} Pressure: {:6.4f} PCorr {:6.4f}'.format(
                self.pressure/self.nSamples, self.energy/self.nSamples, \
                self.nSamples,self.system.GetPressure(self.system.virial), \
                self.system.PressureTailCorrection()))
    
    def Metropolis(self, delta ):
        """Conduct Metropolis test, with safeguards."""
    
        exponent_guard = 75.0

        if delta > exponent_guard: # Too high, reject without evaluating
            return False
        elif delta < 0.0: # Downhill, accept without evaluating
            return True
        else:
            zeta = np.random.rand() # Uniform random number in range (0,1)
            return np.exp(-delta) > zeta # Metropolis test


############################################################################### 

    def UpdateEnergies(self,ri,rj):
        """Calls outside function because it is @njit"""

        eps = self.system.eps
        sig = self.system.sig
        return updateEnergies(ri, rj, self.system.boxSize, self.system.rCut_sq, eps, sig)

###############################################################################
   
    def MoveParticle(self,dr_max,r):
        return r + (np.random.rand(3) - [1,1,1]) * dr_max
                             
    def pick_r(self, w ): 
        # pick a particle based on the "rosenbluth" weights
        zeta = np.random.rand() * self.totalMobility 

        k    = 0
        cumw = w[0]
        
        while True: 
            if ( zeta <= cumw ): break # 
            k += 1
            if ( k >= len( w ) ): 
                print("Welp, we messed up. Probably forgot to start indexing at 0 for pick_r()")
                print(len(self.mobilities), np.sum(self.mobilities), cumw,self.totalMobility,zeta )
                exit

            cumw += w[k]
        return k

    def PickParticle(self):
        return np.random.randint(0,high=self.system.natoms-1) 



def LennardJones(rij, sigma, epsilon):
    sr = sigma / rij
    return 4.0 * epsilon * ( ( sr )**12 - ( sr )**6  ) 
    

def PrintPDB(system,step, name=""):
    f = open(str(name) + "system_step_" + str(step) + ".pdb",'w') 

    for i in range(system.natoms):
        j = []
        j.append( "ATOM".ljust(6) )#atom#6s
        j.append( '{}'.format(str(i+1)).rjust(5) )#aomnum#5d
        j.append( system.atomType.center(4) )#atomname$#4s
        j.append( "MOL".ljust(3) )#resname#1s
        j.append( "A".rjust(1) )#Astring
        j.append( str(i+1).rjust(4) )#resnum
        j.append( str('%8.3f' % (float(system.positions[i][0]))).rjust(8) ) #x
        j.append( str('%8.3f' % (float(system.positions[i][1]))).rjust(8) )#y
        j.append( str('%8.3f' % (float(system.positions[i][2]))).rjust(8) ) #z\
        j.append( str('%6.2f'%(float(1))).rjust(6) )#occ
        j.append( str('%6.2f'%(float(0))).ljust(6) )#temp
        j.append( system.atomType.rjust(12) )#elname  
        #print(i,str(i).rjust(5),j[1], j[2])
        f.write('{}{} {} {} {}{}    {}{}{}{}{}{}\n'.format( j[0],j[1],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11]))

    f.close()  

        
        
###############################################################################
#
#          Begin Simulation - Equilibrate then Production Run
#
###############################################################################       
        
        
# create a system
phase1 = System(number_of_atoms,epsilon, sigma, boxSize, temperature, cutOff, atomType)
phase1.GenerateRandomBox()

PrintPDB(phase1, 0,"pre_")

t_equil0=time.time()
equilibrate = MC_NVT(nEquilSteps,  phase1, "Equilibration")
t_equil1=time.time()
print("Time to equilibrate: ", t_equil1-t_equil0)

PrintPDB(equilibrate.system, equilibrate.nSteps,"equil_")

t_prod0=time.time()
production1  = MC_NVT(1000000,  equilibrate.system, "Production1")
#production2  = KMC_NVT(1000,  overLap, production1.system, "Production2")
t_prod1=time.time()
print("Time for production: ", t_prod1-t_prod0)

PrintPDB(production1.system, production1.nSteps,"post_")

print(phase1.rho)
