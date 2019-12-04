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

 
atomType = "Ar"
epsilon = (0.8, 0.9, 1.0)
sigma   = (0.8, 0.9, 1.0)
composition = np.array([100,200,100])
number_of_atoms = sum(composition)
number_of_atom_types = len(composition)
boxSize = 7.93
cutOff = boxSize / 2
temperature = 1.0
beta = 1/ temperature
nEquilSteps = 100000
outputInterval=2000

###############################################################################
#
#       Some Class definitions
#
###############################################################################

class Table():
    def __init__(self, epsTable,sigTable):
        self.eps = np.asarray(epsTable)
        self.sig = np.asarray(sigTable)
        
        
#        if rule.lower() == ("vdw"):
#            self.table = np.asarray([(x + y)/2 for x in vector for y in vector])
#        elif rule.lower() == ("LB" or "Lorenz-Berthelot"):
#            self.table = np.asarray([np.sqrt(x+y) for x in vector for y in vector])

class System():
    
    def __init__(self,number_of_atoms,epsilon,sigma,boxSize,temp,cutOff,atomType):
        self.natoms = number_of_atoms
        self.atomType = atomType
        self.eps    = np.asarray(epsilon)
        self.sig    = np.asarray(sigma)
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
        
        self.GenerateVdWTable(self.eps, self.sig,rule1='vdw',rule2='LB')

    def GenerateRandomBox(self):
        self.positions = np.random.rand(self.natoms,3) * self.boxSize
        self.atomTypes = np.zeros((self.natoms),dtype=np.int_)
        start = 0
        for iter,item in enumerate(composition):
            self.atomTypes[start:start + item] = iter
            start += item
     
    def GetPressure(self,virial):
        return self.rho / self.beta  + virial / ( 3.0 * self.volume )
    
    def PressureTailCorrection(self):
        return  16.0 / 3.0 * np.pi * self.rho **2 * self.sig **3 * self.eps \
                * ( (2.0/3.0)*(self.sig / self.rCut)**9 \
                   - (self.sig / self.rCut)**3 )
    def EnergyTailCorrection(self):
        return  8.0 / 3.0 * np.pi * self.rho * self.natoms * self.sig **3 \
                * self.eps * ( (1.0/3.0)*(self.sig / self.rCut)**9 \
                 - (self.sig / self.rCut)**3 )
    def ChemPotTailCorrection(self):
        """ beta * mu_corr = 2 * u_corr """
        return 16.0 / 3.0 * np.pi * self.rho * self.sig **3 * self.eps \
               * ( (1.0/3.0)*(self.sig / self.rCut)**9 \
               - (self.sig / self.rCut)**3 )
    def TotalEnergy(self):
        self.energy,self.virial = totalEnergy(self.positions, self.boxSize, \
                                              self.rCut_sq, self.natoms, \
                                              sig=self.vdwTable.sig, \
                                              eps=self.vdwTable.eps,\
                                              atomtype = self.atomTypes)
    def GenerateVdWTable(self,vector1,vector2,rule1='vdw',rule2='LB'):
        self.vdwTable = Table(np.asarray([(x + y)/2 for x in vector1 for y \
                                          in vector1]).reshape(3,3), \
                              np.asarray([np.sqrt(x+y) for x in vector2 for y \
                                          in vector2]).reshape(3,3) \
                              )  
    def WidomInsertion(self,molecule):
        print("testing")

@nb.njit
def totalEnergy(r,box,r_cut_box_sq,n,sig,eps,atomtype):
    potential = 0.0
    virial = 0.0
    for i in range(n-1): # Outer loop over atoms
        ai = atomtype[i]
        for j in range(i+1,n): # Inner loop over atoms
            aj = atomtype[j]
            rij = np.absolute(r[i,:] - r[j,:])      # Separation vector
            rij = np.minimum(rij,box-rij)
            rij_sq = np.sum( rij**2 )  # Squared separation

            if rij_sq < r_cut_box_sq: # Check within cutoff

                sr2    = sig[ai,aj] / rij_sq    # (sigma/rij)**2
                sr6  = sr2 ** 3
                sr12 = sr6 ** 2
                pot  = eps[ai,aj] * (sr12 - sr6)
                vir  = eps[ai,aj] * (2.0 * sr12 - sr6) 
                
                potential += pot
                virial += vir
                
    return potential*4.0, virial*24.0
                
""" THis is outside class since Numba has issues with compiling methods """
@nb.njit #(nb.int64,nb.float64[:],nb.float64,nb.float64,nb.float64[:,:], \
# nb.float64[:,:], nb.float64,nb.float64, nb.float64)       
def updateEnergies(ri, rj, box, r_cut_box_sq, eps, sig, atomtypes,ai):
    rsq = 0.0
    potential = 0.0
    virial = 0.0
    ai = ai
    """Calculate chosen particle's potential energy with rest of system """

    #for index, rj in enumerate(r):
    for j in range(len(rj) ):
        aj = atomtypes[j]
        rij = np.absolute(ri - rj[j]) #rj            # Separation vector
        rij = np.minimum(rij,box-rij)

        rsq = np.sum(rij**2)  # Squared separation
        if rsq < r_cut_box_sq: # Check within cutoff

            sr2    = sig[ai,aj] / rsq    # (sigma/rij)**2
            sr6  = sr2 ** 3
            sr12 = sr6 ** 2
            pot  = eps[ai,aj] * (sr12 - sr6)  
            vir  = eps[ai,aj] * (2.0 * sr12 - sr6) # LJ pair virial
            
            potential += pot
            virial += vir

                            
    return 4*potential, 24*virial

@nb.njit    
def WidomInsertion(rj,box,r_cut_box_sq,eps, sig, atomtypes,ai, N):
    # molType :: Integer (in)
             # random coords in box
    chemPot = 0.0
    for i in range(N):
        ri = np.random.rand(3) * box 
        testPot, testVirial = updateEnergies( ri, rj, box, r_cut_box_sq, eps,
                                             sig, atomtypes,ai)
        chemPot += np.exp( -beta * testPot )
    
    #print("chemPot: ", chemPot, beta)
    return -np.log( chemPot/N )
       
class MCSample():
    def __init__(self,rho=0.0,pressure=0.0,virial=0.0,energy=0.0,at=0.5, \
                 av=0.5, ar=0.5, asw=0.5,output=20,dr_max=0.1 \
                 ):
        self.rho = rho
        self.pressure = pressure
        self.virial = virial
        self.energy = energy
        self.acceptTranslation = at
        self.moveAccept = 0
        self.moveAttempt = 0
        self.acceptVolume = av
        self.volumeAccept = 0
        self.volumeAttempt = 0
        self.acceptReaction = ar
        self.reactionAccept = 0
        self.reactionAttempt = 0
        self.acceptSwap = asw
        self.swapAccept = 0
        self.swapAttempt = 0
        self.nSamples = 0
        self.outputInterval = output
        self.updateInterval = output
        self.dr_max = dr_max

              
class MC_NVT(MCSample):

    def __init__(self, steps, system, runType='Blank',rho=0.0, pressure=0.0, \
                 virial=0.0, energy=0.0, at=0.5, av=0.5, ar=0.5, asw=0.5, \
                 output=2000,dr_max=0.1):
        MCSample.__init__(self,rho=rho,pressure=pressure,virial=virial, \
                          energy=energy,at=at,av=av,ar=ar,asw=asw, \
                          output=output)
        self.nSteps = steps
        self.system = system
        self.runType = runType
        chemPot = []
        chemSample = 0
        NchemTests = 1000
        testMol = 1
        WIDOM = False
        

        print('Beginning {0:s} for {1:d} steps at {2:f} temperature'.format(
                self.runType, self.nSteps, self.system.temp))
        self.InitializeSimulation()
        r = self.system.positions               # simpler notation

        #######################################################################
        #
        #                          NVT Simulation
        #
        #######################################################################        

        steps = 0
        while steps < self.nSteps:
            
            steps += 1
            part = self.PickParticle()
            
            ri = r[part,:]
            rj = np.delete(r,part,0) # Array of all the other atoms
            atypes = np.delete(self.system.atomTypes,part,0)
            
            old_potential, old_virial = self.UpdateEnergies(ri,rj,atypes, \
                                                   self.system.atomTypes[part])
            ri = self.MoveParticle(self.dr_max, r[part,:],self.system.boxSize)
            
            if ri[0] > self.system.boxSize or ri[0] < 0:
                print("FAKE NEWS: ", ri, self.system.boxSize)
            new_potential, new_virial = self.UpdateEnergies(ri,rj,atypes, \
                                                  self.system.atomTypes[part])
            TEST = self.Metropolis( self.system.beta* (new_potential \
                                     - old_potential)  )
            self.moveAttempt += 1
            
            if TEST:
                r[part,:] = ri
                self.system.energy += (new_potential - old_potential)
                self.system.virial += (new_virial - old_virial)
                self.moveAccept += 1
                
            if steps % self.outputInterval == 0:
                self.Sample()
                
                eps = self.system.vdwTable.eps
                sig = self.system.vdwTable.sig
                
                if WIDOM:
                    chemPot.append( WidomInsertion(r, self.system.boxSize,\
                                               self.system.rCut_sq, eps, sig, \
                                               self.system.atomTypes,testMol, \
                                               NchemTests) 
                                  )

                    print("Chemical Potential: ", chemPot[chemSample])
                    chemSample += 1
                
            if steps % self.updateInterval == 0:
                self.UpdateMaxMove()
                PrintPDB(self.system, steps ,"during_")
            
        self.system.positions = r
        #######################################################################   
    def UpdateMaxMove(self):

        ratio = self.moveAccept / self.moveAttempt

        dr_old = self.dr_max

        self.dr_max = self.dr_max * ratio / self.acceptTranslation 
        dr_ratio = self.dr_max / dr_old
        if dr_ratio > 1.5: self.dr_max = dr_old * 1.5
        if dr_ratio < 0.5: self.dr_max = dr_old * 0.5
        if self.dr_max > self.system.boxSize/2: self.dr_max = \
                                                self.system.boxSize/2
        


    def InitializeSimulation(self):
        """
        Calculate pairwise energy and virial for entire system
        """
        self.system.TotalEnergy()

                   
    def Sample(self):
        
        self.nSamples += 1
        self.pressure += self.system.GetPressure(self.system.virial)
        #self.pressure += self.system.PressureTailCorrection()
        self.virial   += self.system.virial
        self.energy   += self.system.energy
        
        print('Pressure: {:6.4f}, Energy: {:6.4f}, Samples: {:6d} Pressure: {:6.4f} '.format(  # PCorr {:6.4f}
                self.pressure/self.nSamples, self.energy/self.nSamples, \
                self.nSamples,self.system.GetPressure(self.system.virial))) #,, \
        #self.system.PressureTailCorrection()

    
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

    def UpdateEnergies(self,ri,rj,atomtypes,ai):
        """Calls outside function because it is @njit"""

        eps = self.system.vdwTable.eps
        sig = self.system.vdwTable.sig
        return updateEnergies(ri, rj, self.system.boxSize, self.system.rCut_sq, \
                              eps, sig,atomtypes,ai)

###############################################################################
   
    def MoveParticle(self,dr_max,r,box):     
        return PBC(r + (np.random.rand(3) - [0.5,0.5,0.5]) * dr_max, box   )
                             
    def pick_r(self, w ): 
        # pick a particle based on the "rosenbluth" weights
        zeta = np.random.rand() * self.totalMobility 

        k    = 0
        cumw = w[0]
        
        while True: 
            if ( zeta <= cumw ): break # 
            k += 1
            if ( k >= len( w ) ): 
                print("Probably forgot to start indexing at 0 for pick_r()")
                print(len(self.mobilities), np.sum(self.mobilities), \
                      cumw,self.totalMobility,zeta )
                exit

            cumw += w[k]
        return k

    def PickParticle(self):
        return np.random.randint(0,high=self.system.natoms-1) 
@nb.njit
def PBC(vector,box):
    for i in range(3):
        if vector[i] > box: vector[i] -= box
        if vector[i] < 0: vector[i] += box
    return vector

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
        f.write('{}{} {} {} {}{}    {}{}{}{}{}{}\n'.format( j[0],j[1],j[2],\
                j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11]))

    f.close()  

        
###############################################################################
#
#          Begin Simulation - Equilibrate then Production Run
#
###############################################################################       
        
        
# create a system
phase1 = System(number_of_atoms,epsilon,sigma,boxSize,temperature,cutOff,atomType)
phase1.GenerateRandomBox()

PrintPDB(phase1, 0,"pre_")

t_equil0=time.time()
preequilibrate = MC_NVT(10000,  phase1, "Equilibration")
t_equil1=time.time()
print("Time to equilibrate: ", t_equil1-t_equil0)

t_equil0=time.time()
equilibrate = MC_NVT(nEquilSteps,  preequilibrate.system, "Equilibration")
t_equil1=time.time()
print("Time to equilibrate: ", t_equil1-t_equil0)

PrintPDB(equilibrate.system, equilibrate.nSteps,"equil_")

t_prod0=time.time()
production1  = MC_NVT(10000000,  equilibrate.system, "Production1")
#production2  = KMC_NVT(1000,  overLap, production1.system, "Production2")
t_prod1=time.time()
print("Time for production: ", t_prod1-t_prod0)

PrintPDB(production1.system, production1.nSteps,"post_")

print(phase1.rho)
