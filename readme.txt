﻿Frontier expansion sampling (FES): a method to enhance the conformational sampling in molecular dynamics simulations


Prepare the files for simulation at first. CHARMM-GUI and tleap are encouraged to prepare simulation files. 


The script could be run with Python3 with Python packages of following: 
Numpy, Mdtraj, Openmm (GPU version), sklearn and Scipy


The simulation files and the number of cycle should be provided in the script.


After providing above information, the FES simulation could run automaticlally by typing the command as follow:
$ python FES.py 


If more cycles are needed or unexpected halt happends, just change the list that saves the number of seeds of each cycle using the number of seeds at already finished cycles and then let it run again.


Here the simulation of maltodextrin binding protein (MBP) is used as an example in the script. The simulation files are open_abmer.prmtop, open_amber.inpcrd and open_amber.pdb. The cycle number is 30.


For comparison, the Structural Dissimilarity Sampling (SDS) proposed by the article, 
“Harada R, Shigeta Y. Efficient conformational search based on structural dissimilarity sampling: applications for reproducing structural transitions of proteins[J]. Journal of chemical theory and computation, 2017, 13(3): 1411-1423.” , 
also be implemented using Openmm and Python packages. The files and cycle number in FES simulation are also needed in SDS simulation. Type the following command to run SDS automatically: 
$ python SDS.py




multi-Topology Frontier expansion sampling (T-FES): a method to enhance the conformational sampling in molecular dynamics simulations using multiple topologies

This extends the existing FES script to perform FES across multiple topologies of a protein. Each topology run is combined together during the PCA. 
When selecting vertices from the convex hull the corresponding topology is also selected and run. In addition, the script has been modified to allow for the inclusion of existing trajecories as a prior.
T-FES will then expand sampling from this.

Finally T-FES can also be run using CUDA MPS using the corresponding script.

T-FES accepts topologies in the Gromacs format but can be easily modified to accept other formats. Some existing topologies are provided in the RW_10 directory.