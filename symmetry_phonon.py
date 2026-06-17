import numpy as np 
import netCDF4 as nc 
from scipy.spatial import KDTree
import os 
#import torch as to

cm_Ha = (1.0 / (8065.610420 * 27.211407953))
Ha_eV = 27.211407953

eigv = 'eigv_0.dat'         #  Modes 
pw_output = 'scf.out'

#sym_op = np.array([[-1,0,0],[0,-1,0],[0,0,-1]])

sym_op = np.array([
[-0.5000000, -0.5*np.sqrt(3),  0.0000000],
[ 0.5*np.sqrt(3), -0.5000000,  0.0000000],
[ 0.0000000,  0.0000000,  1.0000000]   ])

#sym_op = np.array([
#[0.0000000, -1.0, 0.0000000],
#[ -1.0, 0.0000000,  0.0000000],
#[ 0.0000000,  0.0000000,  1.0000000]   ])
#sym_op = np.array([[1,0,0],[0,1,0],[0,0,1]])

# Atomic symbols of all atoms in unit cell
#atoms = ['Bi','Fe','Fe','Bi','O','O','O','O','O','O']

##############################################################

def read_struct(pw_output):
    #
    # Takes scf.out file and returns 
    # atoms, coordinates and lattice vectors
    #
    lat_vecs = np.zeros((3,3))
    atoms = [] # list for storing atomic symbols
    atom_coord = []
    file = open(pw_output,'r')
    lines = file.readlines()  # Storing the file content lines variable
    #
    for m,line in enumerate(lines):
        # Read number of atoms
        if 'number of atoms' in line:
            n_atoms = int(lines[m].split()[4])
        # Read lattice vectors in alat units
        if 'crystal axes: (cart' in line:
            lat_vecs[0,:] = np.array(lines[m+1].split()[3:6])
            lat_vecs[1,:] = np.array(lines[m+2].split()[3:6])
            lat_vecs[2,:] = np.array(lines[m+3].split()[3:6])
            lat_vecs = lat_vecs.astype(np.float64)
            #print(lat_vecs)
        # Read positions in alat units
        if 'positions (alat units)' in line:
            for n in range(n_atoms):
                j = n + m + 1  # line number over n_atoms
                atoms.append(lines[j].split()[1]) # Store atomic symbols
                atom_coord.append(lines[j].split()[6:9])
            atom_coord = np.array(atom_coord,dtype=np.float64) # string to float
            #print(atom_coord)
    return(atoms,atom_coord,lat_vecs)

def ph_eigs(eigv,atoms):
    #
    # Takes dynmat output modes file and returns 
    # Phonon eigenvectors and energies
    #    
    n_atoms = len(atoms)            # Number of atoms in unit cell 
    n_modes = len(atoms) * 3        # Number of phonon modes 3
    mode_file = open(eigv,'r')      # Reading the input file
    lines = mode_file.readlines()   # Storing it in lines variable
    modes = np.zeros((n_modes,n_atoms,3)) # last dim for projection along axes
    energy = np.zeros((n_modes))
    m=-1
    for l,line in enumerate(lines):
        if 'freq' in line:
            m += 1
            energy[m] = np.abs(np.float64(line.split()[7]))
            for i in range(0,n_atoms):
                line_string =lines[l+i+1].split()
                modes[m,i,0] = np.float64(line_string[1]) #/ np.sqrt(m_atom[i])
                modes[m,i,1] = np.float64(line_string[3]) #/ np.sqrt(m_atom[i])
                modes[m,i,2] = np.float64(line_string[5]) #/ np.sqrt(m_atom[i])
                # eigendispacement dimension : mode, atom, cartesian
    return(modes,energy)

def same_atom_check(atom_coord1,atom_coord2,lat_vecs):
    #
    # Checks if coord2 belongs to set of coords in atom_coord1
    #
    # (all input in alat units)
    #
    # Let's transform to crystal coordinates
    # Let's use inverse of lattice vector matrix
    lat_vec_inv = np.linalg.inv(lat_vecs).T #* (-1.0)
    #print(lat_vec_inv)
    atom_coord1_cry = np.einsum('ij,aj->ai',lat_vec_inv,atom_coord1)
    # Bring everything to first primitive cell
    atom_coord1_cry = atom_coord1_cry - np.floor(atom_coord1_cry)
    # Build a tree for querying, boxsize takes care of periodic boundary
    tree = KDTree(atom_coord1_cry,boxsize=[1,1,1])
    atom_coord2_cry = np.einsum('ij,aj->ai',lat_vec_inv,atom_coord2)
    atom_coord2_cry = atom_coord2_cry - np.floor(atom_coord2_cry)
    #
    distances, indices = tree.query(atom_coord2_cry,k=1)
    # Sanity check : following matrix should be null
    #print(atom_coord2_cry - atom_coord1_cry[[indices],...])
    #
    if np.max(distances) > 0.00001:
        print("error, KDTree distance : ",np.max(distances))
    return indices

def check_sym(atom_coord,lat_vecs,mode,sym_op):
    #
    # Checks if a given phonon mode remains invariant 
    # under a symmetry operation
    #
    # sym_op is 3*3 matrix associated with symmetry operation
    # of specific point group (available on Bilbao server)
    #
    # Let's apply symmetry operator to mode and atomic coords
    mode_new = np.einsum('ij,aj-> ai',sym_op,mode)
    atom_coord_new = np.einsum('ij,aj->ai',sym_op,atom_coord)
    # Let's call function 'same atom check'
    # If sym operation brings an atom to an equvalent position,
    # this should be accounted for by refering to the correct index
    # of atom in original coordinates
    indices = same_atom_check(atom_coord,atom_coord_new,lat_vecs)
    mode = mode[indices,...]
    character = np.einsum('ai,ai->',mode,mode_new)
    return character


atoms, atom_coord, lat_vecs = read_struct(pw_output)
modes = ph_eigs(eigv,atoms)[0]
energy = ph_eigs(eigv,atoms)[1]


n_atoms = len(atoms)            # Number of atoms in unit cell 
n_modes = len(atoms) * 3        # Number of phonon modes 3


for m in range(n_modes):
    mode = modes[m,...]
    character = check_sym(atom_coord,lat_vecs,mode,sym_op)
    #if character < 0.5:
    print(m+1,character,energy[m])

########### get eigenvectors ##############

#modes = ph_eigs(eigv,atoms)[0]
#energy = ph_eigs(eigv,atoms)[1] 
#energy = energy * cm_Ha # Convert from cm-1 to Ha

# reshape the eigenvectors as a square matrix
#modes = np.reshape(modes,(n_modes,n_modes)) # mode, natom*axes (xyz xyz .. natom^th *xyz)


###########################################

#print("Einsum (1) begins")
# Convert to (atom, axis) perturbation basis
#elph_cart = np.einsum('ab,qbvc -> qavc',conv_mat,elph)

#print("Einsum (2) begins")
# Use phonon eig. displ. for specific polar mode to compute elph mat.
#elph_lr = np.einsum('ab,qbvc -> qavc',modes_lr,elph_cart)








