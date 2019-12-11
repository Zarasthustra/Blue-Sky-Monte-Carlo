# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:09:10 2019
@author: Zarathustra
"""

"""
Blue Sky Monte Carlo
A.K.A. BSMC
polyatomic code for LJ
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

# need body_fixed coordinates for water
# need proper units of measurement
# need quaternions
# need COM and shift COM
# need book-keeping
#
#  Hve array of grid coords, once one is assigned, remove it from list

###############################################################################
#
#            Simulation Variables (as few as possible)
#
###############################################################################

 
atomType = "Ar"
epsilon = (0.8, 1.0, 1.2)
sigma   = (1.2, 1.0, 0.8)
epsilon = [0.3, 1.0]
sigma = [0.1, 1.0]
composition = np.array( [512] )  
nMol = sum(composition)
number_of_atom_types = len(composition)
number_of_atoms = nMol*3
boxSize = 9.93 #15.52245 #9.99 #7.93
cutOff = boxSize / 2
temperature = 1.0
beta = 1/ temperature
nEquilSteps = 10000
outputInterval=100
initConfig = "positive"

"""
Water
need:
    atom types
    molecule number
    charges
    atomic weights
    ewald/wolf
    body-fixed coordinates
    
    atom array
    molecular array
    atomInMolecule
    quaternion array
"""
h2oBF = np.array([[0,0,0],
                  [0.79926,0.56518,0],
                  [1.59852,0,0]],dtype=np.float_)
atomNames = ["Hw","OW","HW"]


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
    
    def __init__(self,number_of_atoms,nMol,epsilon,sigma,boxSize,temp,cutOff,atomType):
        self.natoms = number_of_atoms
        self.nMol  = nMol
        self.atomType = atomType
        self.eps    = np.asarray(epsilon)
        self.sig    = np.asarray(sigma)
        self.boxSize = boxSize
        self.volume  = boxSize ** 3
        self.rho     = self.nMol / self.volume
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
        
        self.rho = self.natoms / self.volume

    def GenerateBox(self,style="Random"):
        if style.lower() == "random":
            self.COM = np.random.rand(self.natoms,3) * self.boxSize
        elif style.lower() == "cube" or style.lower() == "cubic":
            self.COM, self.rho, self.boxSize = InitCubicGrid(self.nMol,boxSize=self.boxSize)
        else:
            print("Type of simulation box unknown, failing GenerateBox")
            
        self.atomTypes = [] #np.zeros((self.natoms),dtype=np.int_)
        self.db = self.shiftCOM(h2oBF,[1.008,15.9998,1.008])
        self.atomXYZ = []
        self.atomName = []
        self.molTypes = []
        self.molNum = []
        self.atomNum = []
        self.startAtomEndAtom = np.zeros((self.nMol,2),dtype=np.int_)
        atom = 0
        for mol,coord in enumerate(self.COM):
            self.startAtomEndAtom[mol,0] = atom
            for i in range(3):
                self.atomNum.append(atom)
                atom += 1
                self.atomXYZ.append(coord + self.db[i,:])
                self.molTypes.append(0)
                self.molNum.append(mol)
                if i == 1:
                    self.atomTypes.append(1)
                    self.atomName.append("OW")
                else:
                    self.atomTypes.append(0)
                    self.atomName.append("HW")
            self.startAtomEndAtom[mol,1] = atom - 1
        self.atomXYZ = np.asarray(self.atomXYZ)
        self.atomTypes= np.asarray(self.atomTypes)
        self.molTypes = np.asarray(self.molTypes)
        self.atomNum = np.asarray(self.atomNum)
        self.molNum = np.asarray(self.molNum)
        self.nMol = len(self.COM)
        self.nAtoms = len(self.atomXYZ)
        print("atoms: ", len(self.atomXYZ),len(self.COM),self.atomTypes)
        print(self.startAtomEndAtom)
        
    """shifts a molecules center of mass to [0,0,0]"""    
    def shiftCOM(self,COM,mass):
        shift = self.COMM(COM,mass)
        for atom in COM:
            atom += (-1) * shift
        return COM
    
    """Calculates center of mass of a molecule"""
    def COMM(self,xyz, masses): # calculate center of mass of a molecule
        mx = 0.0
        my = 0.0
        mz = 0.0
        mm = sum( masses )
        for i in range( len(xyz) ):
            mx += xyz[i,0] * masses[i]
            my += xyz[i,1] * masses[i]
            mz += xyz[i,2] * masses[i]
        COM = np.array([mx/mm, my/mm, mz/mm])
        return COM  
        
        
        start = 0
        for iter,item in enumerate(composition):
            self.atomTypes[start:start + item] = iter
            start += item
    """Calculated pressure from the virial equation""" 
    def GetPressure(self,virial):
        return self.rho / self.beta  + virial / ( 3.0 * self.volume )
   
    """Calculates pressure tail correction for LJ"""
    def PressureTailCorrection(self):
        return  16.0 / 3.0 * np.pi * self.rho **2 * self.vdwTable.sig[0,0] **3 * self.vdwTable.eps[0,0] \
                * ( (2.0/3.0)*(self.vdwTable.sig[0,0] / self.rCut)**9 \
                   - (self.vdwTable.sig[0,0] / self.rCut)**3 )
     
    """Calculates potential energy tail correction for LJ"""           
    def EnergyTailCorrection(self):
        return  8.0 / 3.0 * np.pi * self.rho * self.natoms * self.vdwTable.sig[0,0] **3 \
                * self.vdwTable.eps[0,0] * ( (1.0/3.0)*(self.vdwTable.sig[0,0] / self.rCut)**9 \
                 - (self.vdwTable.sig[0,0] / self.rCut)**3 )
    
    """Calculates chemical potential tail correction for LJ"""            
    def ChemPotTailCorrection(self):
        """ beta * mu_corr = 2 * u_corr """
        return 16.0 / 3.0 * np.pi * self.rho * self.vdwTable.sig[0,0] **3 * self.vdwTable.eps[0] \
               * ( (1.0/3.0)*(self.vdwTable.sig[0] / self.rCut)**9 \
               - (self.vdwTable.sig[0] / self.rCut)**3 )
    """Calculates total potential energy of the system"""
    def TotalEnergy(self):
        self.energy,self.virial = totalEnergy(self.COM,self.atomXYZ,\
                                              self.boxSize, self.rCut_sq,  \
                                              sig=self.vdwTable.sig, \
                                              eps=self.vdwTable.eps,\
                                              atomtype = self.atomTypes, \
                                              sAeA=self.startAtomEndAtom)
        print("energy total: ", self.energy, self.virial)

        
    """Calculates epsilon and sigma mixing rule parameters"""
    def GenerateVdWTable(self,vector1,vector2,rule1='vdw',rule2='LB'):
        lenx = len(vector1)
        leny = len(vector2)
        self.vdwTable = Table(np.asarray([(x + y)/2 for x in vector1 for y \
                                          in vector1]).reshape(lenx,leny), \
                              np.asarray([np.sqrt(x*y) for x in vector2 for y \
                                          in vector2]).reshape(lenx,leny) \
                              )  
        print(self.vdwTable.sig)
        
    """Doesn't currently do anything for the Widom Insertion"""
    def WidomInsertion(self,molecule):
        print("testing")

"""Calculates total LJ potential energy using numba acceleration"""
@nb.njit
def totalEnergy(rMol,rAtoms,box,r_cut_box_sq,sig,eps,atomtype,sAeA):
    # sAeA = startAtomEndAtom
    # this calculates the molecular virial - should move outside, save time
    potential = 0.0
    virial = 0.0
    n = len(rMol)
    for i in range(n-1): # Outer loop over atoms
        for j in range(i+1,n): # Inner loop over atoms
            rij = rMol[i,:] - rMol[j,:]    # molecule-molecule 
            rij = rij - box * np.rint(rij/box)  # mirror image
            rij_sq = np.sum( rij**2 )  # Squared separation

            if rij_sq < r_cut_box_sq: # Check within cutoff
                iatoms = np.arange(sAeA[i,0],sAeA[i,1]+1)
                jatoms = np.arange(sAeA[j,0],sAeA[j,1]+1)
                ir = rAtoms[iatoms,:]     # coords of atoms in mol i
                jr = rAtoms[jatoms,:]     # coords of atoms in mol j
                iatypes = atomtype[iatoms]
                jatypes = atomtype[jatoms]
                
                for k in range(len(iatoms)):
                    ak = iatypes[k]
                    for l in range(len(jatoms)):
                        al = jatypes[l]
                        
                        rab = ir[k,:] - jr[l,:] #atom-atom sep vector
                        rab = rab - box * np.rint(rab/box) # np.minimum(rab,box-rab)
                        rab_sq = np.sum( rab**2 )
                        
                        sr2  = sig[ak,al]**2 / rab_sq    # (sigma/rij)**2
                        sr6  = sr2 ** 3
                        sr12 = sr6 ** 2
                        pot  = eps[ak,al] * (sr12 - sr6)
                        vir  = eps[ak,al] * (2.0 * sr12 - sr6) 
                        
                        fab   = rab * vir / rab_sq   # LJ atom-atom pair force
                        potential += pot
                        virial += np.sum(rij*fab)
                        
                
    return potential*4.0, virial*24.0
                
""" Updates LJ potential energy after a MC move. uses Numba acceleration"""
@nb.njit #(nb.int64,nb.float64[:],nb.float64,nb.float64,nb.float64[:,:], \
# nb.float64[:,:], nb.float64,nb.float64, nb.float64)       
def updateEnergies(rMoli, rMolj, rAtomi, rAtomj,ai,aj, box, r_cut_box_sq, eps, sig, \
                   sAeAi,sAeAj):

    potential = 0.0
    virial = 0.0
    n = len(rMolj)
    iatoms = np.arange(sAeAi[0],sAeAi[1]+1)
    ir = rAtomi    # coords of atoms in mol i
    iatypes = ai
    """Calculate chosen particle's potential energy with rest of system """

    #for index, rj in enumerate(r):
    for i in range(n-1): # Inner loop over atoms
        rij = rMoli[:] - rMolj[i,:]    # molecule-molecule 
        rij = rij - box * np.rint(rij/box)  # mirror image
        rij_sq = np.sum( rij**2 )  # Squared separation

        if rij_sq < r_cut_box_sq: # Check within cutoff

            jatoms = np.arange(sAeAj[i,0],sAeAj[i,1]+1)
            jr = rAtomj[jatoms,:]     # coords of atoms in mol j
            jatypes = aj[jatoms]
            
            for k in range(len(iatoms)):
                ak = iatypes[k]
                for l in range(len(jatoms)):
                    al = jatypes[l]
                    
                    rab = ir[k,:] - jr[l,:] #atom-atom sep vector
                    rab = rab - box * np.rint(rab/box) # np.minimum(rab,box-rab)
                    rab_sq = np.sum( rab**2 )
                    
                    sr2  = sig[ak,al]**2 / rab_sq    # (sigma/rij)**2
                    sr6  = sr2 ** 3
                    sr12 = sr6 ** 2
                    pot  = eps[ak,al] * (sr12 - sr6)
                    vir  = eps[ak,al] * (2.0 * sr12 - sr6) 
                    
                    fab   = rab * vir / rab_sq   # LJ atom-atom pair force
                    potential += pot
                    virial += np.sum(rij*fab)

    return 4*potential, 24*virial

"""Peforms a widom insertion N times, uses Numba acceleration"""
@nb.njit    
def WidomInsertion(rj,box,r_cut_box_sq,eps, sig, atomtypes,ai, N):
    # molType :: Integer (in)
             
    chemPot = 0.0
    for i in range(N):
        ri = np.random.rand(3) * box   # random coords in box
        testPot, testVirial = updateEnergies( ri, rj, box, r_cut_box_sq, eps,
                                             sig, atomtypes,ai)
        chemPot += np.exp( -beta * testPot )
        #print(i, testPot, testVirial, eps, sig, ai, ri, box, rj)
    #print("chemPot: ", chemPot, beta)
    return -np.log( chemPot/N )
 
"""Object for sampling thermodynamic properties in a MC simulation"""      
class MCSample():
    def __init__(self,rho=0.0,pressure=0.0,virial=0.0,energy=0.0,at=0.5, \
                 av=0.5, ar=0.5, asw=0.5,output=20,dr_max=0.8 \
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

"""Object class for an NVT MC simulations. Contains main loop"""              
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
        NchemTests = 10000
        testMol = 0
        WIDOM = False
        
        boxSize, dr_max, beta, atomTypes = self.system.boxSize, self.dr_max, \
            self.system.beta, self.system.atomTypes
        eps, sig, rCut_sq = self.system.vdwTable.eps, self.system.vdwTable.sig,\
            self.system.rCut_sq

        print('Beginning {0:s} for {1:d} steps at {2:f} temperature'.format(
                self.runType, self.nSteps, self.system.temp))
        self.InitializeSimulation()
        rMol = self.system.COM              # simpler notation
        sAeA = self.system.startAtomEndAtom
        rAtom = self.system.atomXYZ

        #######################################################################
        #
        #                          NVT Simulation
        #
        #######################################################################        

        steps = 0
        while steps < self.nSteps:
            
            steps += 1
            part = self.PickParticle()
            #print("=====",part)
            rMoli = rMol[part,:]
            #print("old com: ",rMoli)
            rMolj = np.delete(rMol,part,0) # Array of all the other atoms
            
            atomRange=np.arange(sAeA[part,0],sAeA[part,1]+1)
            rAtomi= rAtom[atomRange,:]
            rAtomj = np.delete(rAtom,atomRange,0)
            tempAtomTypesj = np.delete(atomTypes,atomRange,0)
            tempAtomTypesi = atomTypes[atomRange]
            temp_sAeAi = sAeA[part,:]
            temp_sAeAj = np.delete(sAeA,part,0)
            #print("stage",sAeA[part])
            old_potential, old_virial = self.UpdateEnergies(rMoli,rMolj,rAtomi,\
                                                            rAtomj,tempAtomTypesi, \
                                                            tempAtomTypesj,\
                                                            temp_sAeAi,temp_sAeAj)
            rMoli_new = self.MoveParticle(dr_max, rMol[part,:],boxSize)
            #print("newcom: ",rMoli_new)
          
            change = rMoli_new - rMoli
            #print("change com: ",change, dr_max)
            #print("=================")
            #print(rAtomi)
            rAtomi += change
            #print(rAtomi)

            new_potential, new_virial = self.UpdateEnergies(rMoli_new,rMolj,rAtomi,\
                                                            rAtomj,tempAtomTypesi, \
                                                            tempAtomTypesj,\
                                                            temp_sAeAi,temp_sAeAj)
            TEST = self.Metropolis( beta* (new_potential - old_potential)  )
            self.moveAttempt += 1
            
            if TEST:
                rMol[part,:] = rMoli_new
                rAtom[atomRange,:] = rAtomi
                self.system.energy += (new_potential - old_potential)
                self.system.virial += (new_virial - old_virial)
                self.moveAccept += 1
                #print("passed: ",new_potential - old_potential )
            #else:
                #print("failed: ",new_potential - old_potential )
                
            if steps % self.outputInterval == 0:
                self.Sample()
                
                if WIDOM:
                    chemPot.append( WidomInsertion(rMol,rAtom, boxSize,rCut_sq, eps, \
                                                   sig,atomTypes,testMol, \
                                                   NchemTests) 
                                  )

                    print("Chemical Potential: ", chemPot[chemSample],np.mean(chemPot) )
                    chemSample += 1
                
            if steps % self.updateInterval == 0:
                self.UpdateMaxMove()
                #PrintPDB(self.system, steps ,"during_")
            
        self.system.COM = rMol
        self.atomXYZ = rAtom
        
        self.Output_Conclusions()
        #######################################################################   
    """Update the maximum allowed translation move parameter"""
    def UpdateMaxMove(self):

        ratio = self.moveAccept / self.moveAttempt

        dr_old = self.dr_max

        self.dr_max = self.dr_max * ratio / self.acceptTranslation 
        dr_ratio = self.dr_max / dr_old
        if dr_ratio > 1.5: self.dr_max = dr_old * 1.5
        if dr_ratio < 0.5: self.dr_max = dr_old * 0.5
        if self.dr_max > self.system.boxSize/2: self.dr_max = \
                                                self.system.boxSize/2
        
    def Output_Conclusions(self):
        print("========================================================")
        print("In Conclusion:")
        print('Pressure: {:6.4f} \n Avg Energy: {:6.4f} \n Energy: {:6.4f} \
              \n Samples: {:6d} \n Pressure: {:6.4f} \n Pcorr: {:6.4f} \n Ecorr {:6.4f} \
              \n Density: {:6.4f}'.format(  # PCorr {:6.4f}
                self.pressure/self.nSamples, self.energy/self.nSamples,self.system.energy, \
                self.nSamples,self.system.GetPressure(self.system.virial), \
                self.system.PressureTailCorrection(),self.system.EnergyTailCorrection(),
                self.rho/self.nSamples) ) 
        print("========================================================")
    """Initialize system by calculating total potential energy"""
    def InitializeSimulation(self):
        """
        Calculate pairwise energy and virial for entire system
        """
        self.system.TotalEnergy()

    """Sample properties in this current configuration"""               
    def Sample(self):
        
        self.nSamples += 1
        self.pressure += self.system.GetPressure(self.system.virial)
        self.pressure += self.system.PressureTailCorrection()
        self.virial   += self.system.virial
        self.energy   += self.system.energy
        self.rho      += self.system.natoms / self.system.volume
        
        print('Pressure: {:6.4f}, Avg Energy: {:6.4f}, Energy: {:6.4f}, Samples: {:6d} Pressure: {:6.4f} Pcorr: {:6.4f} Ecorr {:6.4f}'.format(  # PCorr {:6.4f}
                self.pressure/self.nSamples, self.energy/self.nSamples,self.system.energy, \
                self.nSamples,self.system.GetPressure(self.system.virial), \
                self.system.PressureTailCorrection(),\
                self.system.EnergyTailCorrection() ) ) #,, \
        

    """Test if we accept this MC move according to the metropolis criteria"""
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

    def UpdateEnergies(self,rMoli,rMolj,rAtomi,rAtomj,ai,aj,sAeAi,sAeAj):
        """Calls outside function because it is @njit"""

        eps, sig = self.system.vdwTable.eps, self.system.vdwTable.sig

        return updateEnergies(rMoli,rMolj,rAtomi,rAtomj,ai,aj, self.system.boxSize, \
                              self.system.rCut_sq, eps, sig,sAeAi,sAeAj)


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
        return np.random.randint(0,high=self.system.nMol) 
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
        j.append( str('%8.3f' % (float(system.atomXYZ[i][0])*10)).rjust(8) ) #x
        j.append( str('%8.3f' % (float(system.atomXYZ[i][1])*10)).rjust(8) )#y
        j.append( str('%8.3f' % (float(system.atomXYZ[i][2])*10)).rjust(8) ) #z\
        j.append( str('%6.2f'%(float(1))).rjust(6) )#occ
        j.append( str('%6.2f'%(float(0))).ljust(6) )#temp
        j.append( system.atomType.rjust(12) )#elname  
        #print(i,str(i).rjust(5),j[1], j[2])
        f.write('{}{} {} {} {}{}    {}{}{}{}{}{}\n'.format( j[0],j[1],j[2],\
                j[3],j[4],j[5],j[6],j[7],j[8],j[9],j[10],j[11]))

    f.close()  

#============================================================================
def InitCubicGrid(nMol, rho=0.0, boxSize=0.0):
#============================================================================

    #------------------------------------------------------------------------
    # Created by Braden Kelly
    #------------------------------------------------------------------------
    # Creates an initial configuration 
    #------------------------------------------------------------
    # input:  nMol      number of molecules
    #         rho       density (optional)
    #         boxSize   length of simulation box (optional)
    # output: rMol      coordinates of n particles
    #         BoxSize   length of box
    #         rho       density
    #---------------------------------------------------------------------------------------------------
    if boxSize == 0.0 and rho != 0.0:
        boxSize = (float(nMol)/rho)**(1.0/3.0)
    elif rho == 0.0 and boxSize != 0.0:
        rho = float(nMol) / boxSize **3
    else:
        print("Trying to create a cubic grid but neither rho or boxSize is known")
        quit
    # Calculate the lowest perfect cube that will contain all of the particles


    rMol = np.zeros((nMol,3),dtype=np.float_)    
    nCube = 2

    while nCube**3 < nMol:
        nCube += 1
    # initial position of particle 1
        posit=np.zeros((3))

        #begin assigning particle positions
        for i in range(nMol):
            coords = (posit + [0.5,0.5,0.5])*(boxSize/nCube)
            if 'centered' in initConfig:
                coords = coords - boxSize / 2
            rMol[i,:] = coords
            
           # Advancing the index (posit)
            posit[0] += 1
            if  posit[0] == nCube:
                posit[0] = 0
                posit[1] +=  1
                
                if  posit[1] == nCube:
                    posit[1] = 0
                    posit[2] += 1
    
    return rMol, rho, boxSize                   
#    if 'centered' in initConfig:
#        """ shift so that center of box is at (0,0,0)"""
#        rMol[:,:] = rMol[:,:] - L / 2 

        
        
###############################################################################
#
#          Begin Simulation - Equilibrate then Production Run
#
###############################################################################       
        
        
# create a system
phase1 = System(number_of_atoms,nMol,epsilon,sigma,boxSize,temperature,cutOff,atomType)
phase1.GenerateBox("cube")

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
production1  = MC_NVT(100000,  equilibrate.system, "Production1")
#production2  = KMC_NVT(1000,  overLap, production1.system, "Production2")
t_prod1=time.time()
print("Time for production: ", t_prod1-t_prod0)

PrintPDB(production1.system, production1.nSteps,"post_")

print(phase1.rho)
