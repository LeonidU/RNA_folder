import tensorflow as tf
from Bio.PDB import PDBParser
from os import listdir
from os.path import isfile, join
import numpy as np

SIZE = 2000
WINDOW = 5
image_size = SIZE

ribose_sugar = [ "P",  "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "OP1", "OP2", "OP3"]

adenine = ["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"]

cytosine = ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]

guanine = ["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"]

uracil = ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]

def get_keys_by_value(my_dict, target_value):
    return [key for key, value in my_dict.items() if value == target_value]

def number_to_rna(atom_nums):
    rna_arrays = string_of_rna()
    return([get_keys_by_value(rna_arrays, atom_num)[0] for atom_num in atom_nums])

import numpy as np
from scipy.optimize import minimize

def error_function(coords_flattened, N, distances):
    coords = coords_flattened.reshape((N, 3))
    error = 0
    for i in range(N):
        for j in range(i+1, N):
            computed_distance = np.linalg.norm(coords[i] - coords[j])
            error += (computed_distance - distances[i, j - i - 1]) ** 2
    return error

def DGD(distances):
    N = distances.shape[0] + 1
    initial_coords = np.random.rand(N * 3)  # initial random 3D configuration
    
    result = minimize(error_function, initial_coords, args=(N, distances))
    return result.x.reshape((N, 3))

distances = np.array([
    [1, 1.5, 2.5, 3, 3.5],
    [1.5, 1, 2, 2.5, 3],
    [2.5, 2, 1, 1.5, 2],
    [3, 2.5, 1.5, 1, 1.5]
])

coords = DGD(distances)
print(coords)


def distance_matrix(pdb_filename, maximal_size):
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_filename)
    
    # Extract atom coordinates
    atoms = [atom.get_coord() for model in structure for chain in model for residue in chain for atom in residue if (atom.get_name() in ribose_sugar)]
    n_atoms = len(atoms)
    # Initialize the distance matrix
    dist_matrix = np.zeros((maximal_size, maximal_size))
    
    # Calculate the pairwise distances
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance = np.linalg.norm(atoms[i] - atoms[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
            
    return(dist_matrix)

def distance_array(pdb_filename, maximal_size):
    # Parse the PDB file
    print(pdb_filename)
    parser = PDBParser()
    structure = parser.get_structure("structure", pdb_filename)
    rna_arrays = string_of_rna()
    # Extract atom coordinates
    atoms = [ atom.get_coord() for model in structure for chain in model for residue in chain for atom in residue if (atom.get_name()[:1] != "H" and residue.get_resname() in ['A', 'C', 'G', 'U']) ]
    types = [ residue.get_resname()+"_"+atom.get_name() for model in structure for chain in model for residue in chain for atom in residue if (atom.get_name()[:1] != "H" and residue.get_resname() in ['A', 'C', 'G', 'U'] ) ]
    n_atoms = len(atoms)
    print(n_atoms)
    # Initialize the distance matrix
    distances = np.zeros((WINDOW*maximal_size))
    atom_types = np.zeros((WINDOW*maximal_size))
    ind = 0
    # Calculate the pairwise distances
    for i in range(n_atoms):
        for j in range(max(0,i-WINDOW-1), i-1):
            distances[ind] = np.linalg.norm(atoms[i] - atoms[j])
            atom_types[ind] = rna_arrays[types[j]]
            ind += 1
            
    return(np.double(np.vstack([distances, atom_types])))

def map_strings_to_numbers(strings):
    counter = 1
    string_to_number = dict()
    for s in strings:
        if s not in string_to_number:
            string_to_number[s] = counter
            counter += 1
    return(string_to_number)

def string_of_rna():
    result = []
    nucleotides = ["A", "C", "G", "U"]
    for nucl in nucleotides:
        result += [nucl+"_"+atom for atom in ribose_sugar]
        if (nucl == "A"):
            result += [nucl+"_"+atom for atom in adenine]
        if (nucl == "C"):
            result += [nucl+"_"+atom for atom in cytosine]
        if (nucl == "G"):
            result += [nucl+"_"+atom for atom in guanine]
        if (nucl == "U"):
            result += [nucl+"_"+atom for atom in uracil]
    return(map_strings_to_numbers(result))

def extract_nucleotide_sequence(pdb_filename):
    # Create a PDB parser
    parser = PDBParser()

    # Read the structure from the PDB file
    structure = parser.get_structure('nucleotide', pdb_filename)

    # Extract the nucleotide sequence
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                if res_name in ['A', 'C', 'G', 'T', 'U']:
                    if res_name == 'U':  # Replace U with T for RNA structures
                        sequence += 'T'
                    else:
                        sequence += res_name

    return sequence

def prepare_dataset(dirpath, batch_size=2, split=["80", "20"]):
    X = []
    for f in listdir(dirpath):
        if isfile(join(dirpath, f)) and (f[-4:] == ".pdb"): 
            filename = join(dirpath, f)
            x = distance_array(filename,SIZE)
            print(x.shape)
            X.append(np.array(x))
    return(tf.data.Dataset.from_tensor_slices(X[:int(len(X)*0.8)]).batch(batch_size=batch_size), tf.data.Dataset.from_tensor_slices(np.array(X[int(len(X)*0.8):])).batch(batch_size=batch_size))

