# In[1]: Importing modules
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import shlex
import re
import itertools
import nglview as nv
import ipywidgets
import random
import argparse
import shutil

import pycharmm
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.settings as settings
import pycharmm.write as write
import pycharmm.nbonds as nbonds
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.select as select
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.param as param
import pycharmm.cons_harm as cons_harm
import pycharmm.cons_fix as cons_fix
import pycharmm.shake as shake
import pycharmm.scalar as scalar
import pycharmm.charmm_file as charmm_file
import argparse
import torch

if torch.cuda.is_available():
    gpu_available = True
else:
    gpu_available = False

# Parameters:
hmr = True
hmr_waters = False
input_folder = '5dfr_asp_glu_hsp_lys'
structure_file = 'solvated'
toppar = 'toppar'
topology_files = [
    'top_all36_prot.rtf',
    'par_all36m_prot.prm',
    'top_all36_na.rtf',
    'par_all36_na.prm',
    'toppar_water_ions.str',
    'top_all36_cgenff.rtf',
    'par_all36_cgenff.prm',
    'my_files/titratable_residues.str',
    'my_files/nucleic_c36.str',
    'my_files/his_patches.str'
    # 'top_all22_prot.rtf',
    # 'par_all22_prot.prm',
    # 'my_files/titratable_residues_c22.str'
]
# selected_patching = [] # Remove this global initialization

# Global list to store parsed selection criteria
parsed_selected_patching = []

import argparse

def parser():
    """
    Parses command line arguments and assigns global variables.
    Clears parsed_selected_patching if called multiple times (e.g. in testing).
    """
    # Create the parser
    global hmr, input_folder, structure_file, hmr_waters, parsed_selected_patching # Remove selected_patching from globals
    parser = argparse.ArgumentParser(description='Process additional parameters for the simulation.')

    # Define the expected command line arguments
    parser.add_argument('-hmr','--hmr', type=bool, default=hmr, help='Whether to use hydrogen mass repartitioning (True/False)',action=argparse.BooleanOptionalAction)
    parser.add_argument('-hmrw','--hmr_waters', type=bool, default=hmr_waters, help='Whether to use hydrogen mass repartitioning for water molecules (True/False)',action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--input_folder', type=str, default=input_folder, help='Name of the input folder')
    parser.add_argument('-f', '--file', type=str, default=structure_file, help='Name of the structure file')
    parser.add_argument('-s', '--selected_patching', type=str, nargs='+', default=[], help='Selected residues for patching')

    # Parse the arguments
    args = parser.parse_args()

    # Assign global variables
    hmr = args.hmr
    hmr_waters = args.hmr_waters
    input_folder = args.input_folder
    structure_file = args.file
    # Parse the selected_patching arguments
    parsed_selected_patching.clear() # Clear if called multiple times (e.g. in testing)
    if args.selected_patching: # If a selection is provided
        for item in args.selected_patching:
            item_upper = item.upper()
            if ':' in item:
                parts = item.split(':', 1)
                segid, resid_str = parts[0].upper(), parts[1]
                if resid_str.isdigit():
                    parsed_selected_patching.append(('segid_resid', (segid, resid_str)))
                else:
                    print(f"Warning: Ignoring invalid segid:resid format '{item}'. Resid must be a number.")
            elif item_upper.isalpha() and len(item_upper) <= 4: # Assuming resnames are alphabetic and typical CHARMM length
                parsed_selected_patching.append(('resname', item_upper))
            elif item.isdigit():
                parsed_selected_patching.append(('resid', item))
            else:
                # Fallback: treat as resname if not clearly resid or segid:resid, can be adjusted
                # This allows for resnames that might contain numbers if not purely isdigit()
                # Or print a warning for unclassified items
                print(f"Warning: Treating '{item}' as a resname (fallback case). If it was intended as a resid or segid:resid, check format.")
                # Optionally, you could log these to a file or collect them in a list for further analysis
                parsed_selected_patching.append(('resname', item_upper))
    # The original selected_patching (list of uppercase strings) might still be used by some parts (e.g. HSP check initially)
    # We should assess if we need to keep it or rely solely on parsed_selected_patching
    # For now, let's keep it but its role will be diminished.
    # selected_patching = [item.upper() for item in selected_patching_args] # Remove this assignment



def should_patch_residue(seg_id, res_id, res_name, criteria_list):
    if not criteria_list: # If no selection criteria are provided, patch the residue by default
        return True

    res_id_str = str(res_id) # Convert res_id to string for comparison
    res_name_upper = res_name.upper() # Ensure res_name is uppercase for comparison

    for criterion_type, value in criteria_list:
        if criterion_type == 'resname' and res_name_upper == value:
            return True
        elif criterion_type == 'resid' and res_id_str == value:
            return True
        elif criterion_type == 'segid_resid':
            sel_seg, sel_res = value
            if seg_id.upper() == sel_seg and res_id_str == sel_res:
                return True
    return False

# Functions & Classes:
class universe():
    def __init__(self):
        self.universe = self.update_universe()

    def update_universe(self):
        try:
            select_all = pycharmm.SelectAtoms().all_atoms()
        except:
            raise Exception('No atoms in system')

        crds = coor.get_positions()
        universe = pd.DataFrame({'index': select_all._atom_indexes, 'res_name': select_all._res_names, 'res_id': select_all._res_ids,
                                'seg_id': select_all._seg_ids, 'chem_type': select_all._chem_types, 'atom_type': select_all._atom_types, 'x': crds['x'].values, 'y': crds['y'].values, 'z': crds['z'].values})
        # universe.set_index('index', inplace=True)
        return universe

    def resid_of_resname(self, resname):
        return self.universe[self.universe['res_name'] == resname]['res_id'].unique()

    def resname_of_resid(self, resid):
        return self.universe[self.universe['res_id'] == resid]['res_name'].unique()


def check_charmm():    
    global toppar
    # Checking CHARMM executable
    path_charmm_match = re.search(r"'(.*?)'", str(pycharmm.lib.__dict__.get('charmm')))
    if path_charmm_match:
        path_charmm = path_charmm_match.group(1)
        print('CHARMM executable is:', path_charmm)
    else:
        print('CHARMM executable path not found.')
        sys.exit()

    # Check that topology_folder is defined and exists
    if 'toppar' not in globals():
        print('Topology folder not defined properly. Using default topology folder.')
        toppar = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path_charmm))), 'toppar')
    if os.path.exists(toppar):
        print('Topology folder is:', toppar)
    else:
        print('Topology folder not found. Please check the path.')
        sys.exit()
    lingo.charmm_script('IOFOrmat EXTEnded')
    return None
       
        
def read_topology_files(verbose=True):
    if not verbose:
        lingo.charmm_script('prnlev -1')
    
    prm_files = [file for file in topology_files if file.endswith('.prm')]
    rtf_files = [file for file in topology_files if file.endswith('.rtf')]
    str_files = [file for file in topology_files if file.endswith('.str')]
    lingo.charmm_script('bomblevel -2')
    settings.set_warn_level(-1)
    if rtf_files:
        read.rtf(os.path.join(toppar, rtf_files[0]))
        for file in rtf_files[1:]:
            read.rtf(os.path.join(toppar, file), append=True)
            
    if prm_files:
        read.prm(os.path.join(toppar, prm_files[0]), flex=True)
        for file in prm_files[1:]:
            read.prm(os.path.join(toppar, file), flex=True, append=True)
            
    for file in str_files:
        lingo.charmm_script(f'stream {os.path.join(toppar, file)}')
    
    settings.set_warn_level(5)
    lingo.charmm_script('bomblevel 0')
    if not verbose:
        lingo.charmm_script('prnlev 5')
    return None


class PatchParser:
    # Find patches in topologies

    # open file and find relavanat lines to amino acid

    # find lines starting with 'pres {amino_acid}', save patch_name pres {patch_name} as dictionary key,
    # read following lines starting with 'atom', and save atom names for patch, as dictionary values
    def __init__(self, segment_path='toppar/my_files/titratable_residues.str', topology_path='toppar/top_all36_prot.rtf'):
        self.segment_path = segment_path
        self.topology_path = topology_path
        self.atom_groups = {}
        self.patches = {}
        self.pka = {}
        self.residues = []
        self.default_patches = []
        self.load_patches()
        self.load_topology()
        self.load_default_patches()
        # delete repeats in atom_groups
        for group in self.atom_groups:
            self.atom_groups[group] = list(set(self.atom_groups[group]))

    def load_patches(self):
        if not os.path.exists(self.segment_path):
            raise Exception(f'{self.segment_path} not found. Please check.')

        with open(self.segment_path, 'r') as file:
            lines = file.readlines()
            lines = [line.upper() for line in lines]

        for i in range(len(lines)):
            if lines[i].startswith("!") and lines[i].endswith("PATCHES\n"):
                resname_match = re.search(r'\((\w+)\)', string=lines[i])
                if resname_match:
                    resname = resname_match.group(1).upper()
                    self.residues.append(resname)
                    self.patches.setdefault(resname, [])
                    self.atom_groups.setdefault(resname, [])

                    for j in range(i+1, len(lines)):
                        if lines[j].startswith("!") and lines[j].endswith("PATCHES\n"):
                            break
                        if lines[j].startswith("PRES"):
                            patch_name = lines[j].split()[1].upper()
                            self.patches[resname].append(
                                patch_name) if patch_name not in self.patches[resname] else None
                            self.atom_groups.setdefault(patch_name, [])

                            for k in range(j+1, len(lines)):
                                if lines[k].startswith("PRES"):
                                    break
                                if "pka" in lines[k].lower():
                                    self.pka[patch_name] = lines[k].split('=')[
                                        1].strip()
                                if lines[k].startswith("ATOM"):
                                    atom = lines[k].split()[1].upper()
                                    self.atom_groups[patch_name].append(atom)
                                    if atom not in self.atom_groups[resname]:
                                        self.atom_groups[resname].append(atom)
                                        
    def load_default_patches(self):
        if not os.path.exists(self.topology_path):
            raise Exception(f'{self.topology_path} not found. Please check.')

        with open(self.topology_path, 'r') as file:
            lines = file.readlines()
            lines = [line.upper() for line in lines]

        for i in range(len(lines)):
            if lines[i].startswith("PRES"):
                patch_name = lines[i].split()[1].upper()
                self.default_patches.append(patch_name)
                self.atom_groups.setdefault(patch_name, [])

                for k in range(i+1, len(lines)):
                    if lines[k].startswith("PRES") or lines[k].startswith("RESI"):
                        break
                    if lines[k].startswith("ATOM"):
                        atom = lines[k].split()[1].upper()
                        self.atom_groups[patch_name].append(atom)

    def load_topology(self):
        if not os.path.exists(self.topology_path):
            return

        with open(self.topology_path, 'r') as file:
            lines = file.readlines()
            lines = [line.upper() for line in lines]

        for i in range(len(lines)):
            if lines[i].startswith("RESI"):
                resname = lines[i].split()[1].upper()
                if resname not in self.residues:
                    continue
                atoms = []
                for j in range(i+1, len(lines)):
                    if lines[j].startswith("RESI") or lines[j].startswith("PRES"):
                        break
                    if lines[j].startswith("ATOM"):
                        atoms.append(lines[j].split()[1])
                    if lines[j].startswith("BOND"):
                        bonds = lines[j].split()[1:]

                # Remove all atoms which are not in 'atoms' list for 'self.atom_groups[resname]'
                self.atom_groups[resname] = [
                    atom for atom in self.atom_groups[resname] if atom in atoms]
                

                # Look for bonds to find H atoms
                for k in range(0, len(bonds), 2):
                    if bonds[k] in self.atom_groups[resname] and not bonds[k].startswith('H'):
                        if bonds[k+1] not in self.atom_groups[resname] and bonds[k+1].startswith('H'):
                            self.atom_groups[resname].append(bonds[k+1])
                    if bonds[k+1] in self.atom_groups[resname] and not bonds[k+1].startswith('H'):
                        if bonds[k] not in self.atom_groups[resname] and bonds[k].startswith('H'):
                            self.atom_groups[resname].append(bonds[k])

    def print_atoms(self):
        for i in self.atom_groups.keys():
            if i in self.pka:
                print(f'{i}, pKa {self.pka[i]}: {self.atom_groups[i]}\n')
            else:
                print(f'{i}: {self.atom_groups[i]}\n')



# In[2]: Arguments
parser()
# Now you can use the variables in your code as they have been assigned globally
print(f"HMR Enabled: {hmr}")
print(f"Input Folder: {input_folder}")
print(f"Structure File: {structure_file}")

# Check that input files exist

required_files = [f'{structure_file}.crd', f'{structure_file}.psf']
for file in required_files:
    if not os.path.exists(os.path.join(input_folder, file)):
        print(f'{file} not found. Please regenerate {file}.')
        sys.exit()
    elif os.path.getsize(os.path.join(input_folder, file)) == 0:
        print(f'{file} is empty. Please regenerate {file}.')
        sys.exit()

# In[3]: CHARMM topology and parameter files


# check_charmm()
read_topology_files()


# In[5]: Create patches

# load topology and coordinates
read.psf_card(f'{input_folder}/{structure_file}.psf')
read.coor_card(f'{input_folder}/{structure_file}.crd')
uni = universe()
patches_topology = PatchParser(segment_path='toppar/my_files/titratable_residues.str', topology_path='toppar/top_all36_prot.rtf')
patches_topology.print_atoms()

titratable_dict, i = [], 0
for resname in patches_topology.residues:
    # Find all unique residues of resname
    residues = [(row['seg_id'], row['res_id'], row['res_name'])
                for _, row in uni.universe[uni.universe['res_name'] == resname][['seg_id', 'res_id', 'res_name']].drop_duplicates().iterrows()]

    # Special case for disulfide bonds in cysteine residues
    if resname == 'CYS':
        # Filter only cysteines that have SG atom type
        cys_universe = uni.universe[(uni.universe['res_name'] == 'CYS') & (
            uni.universe['atom_type'] == 'SG')]
        
        bonded_res = set()
        # If there are cysteines to consider
        if not cys_universe.empty:
            num_cys_sg = len(cys_universe)
            for i in range(num_cys_sg):
                cys1 = cys_universe.iloc[i]
                cys1_pos = np.array([cys1['x'], cys1['y'], cys1['z']])
                for j in range(i + 1, num_cys_sg): # Ensures j > i, so each pair is unique & no self-comparison of rows
                    cys2 = cys_universe.iloc[j]

                    # If two different SG atoms belong to the same residue (e.g. altlocs), they should not form a disulfide bond.
                    if cys1['res_id'] == cys2['res_id']:
                        continue

                    cys2_pos = np.array([cys2['x'], cys2['y'], cys2['z']])
                    distance = np.linalg.norm(cys1_pos - cys2_pos)

                    if distance < 2.6:  # Threshold for disulfide bond
                        print(
                            f'CYS {cys1["res_id"]} is disulfide bonded to {cys2["res_id"]}')
                        bonded_res.update([(cys1['seg_id'], cys1['res_id'], cys1['res_name']),
                                           (cys2['seg_id'], cys2['res_id'], cys2['res_name'])])
            # Filter out bonded cysteines from the residues list
            residues = [res for res in residues if res not in bonded_res]

            # Print statement for residues not involved in bonds if needed
            for res in residues:
                print(
                    f'Patching CYS {res[1]} not involved in a disulfide bond.')

    # Append non-bonded or non-CYS residues to titratable_list
    titratable_dict.extend(residues)



seg_ids = uni.universe['seg_id'].unique().tolist()
print(f'Structure segids: {seg_ids}')
for seg_id in seg_ids:
    if len(seg_ids) == 0:
        print(f'No segments found in the structure. Please check the input files.')
    elif seg_id == seg_ids[0]:
        print(f'Found {len(seg_ids)} segments in the structure. Creating temporary structure files to proceed patching')
        os.makedirs(f'{input_folder}/tmp', exist_ok=True)
    else:
        psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
        read.psf_card(f'{input_folder}/{structure_file}.psf')
        read.coor_card(f'{input_folder}/{structure_file}.crd')
    # selection = pycharmm.SelectAtoms().by_seg_id(seg_id).__invert__()
    # psf.delete_atoms(selection)
    if len(seg_ids) > 1:
        pycharmm.lingo.charmm_script(f'DELEte ATOMs SELEct .not. segid {seg_id} END')
    residues = [(row['seg_id'], row['res_id'], row['res_name'])
            for _, row in uni.universe[(uni.universe['res_name'] == 'HSD') & (uni.universe['seg_id'] == seg_id)][['seg_id', 'res_id', 'res_name']].drop_duplicates().iterrows()]
    residues += [(row['seg_id'], row['res_id'], row['res_name'])
                for _, row in uni.universe[(uni.universe['res_name'] == 'HSE') & (uni.universe['seg_id'] == seg_id)][['seg_id', 'res_id', 'res_name']].drop_duplicates().iterrows()]
    if residues:
        for res in residues:
            #rename in uni
            print(f'Patching HSPP {res[0]} {res[1]}')
            # pycharmm.charmm_script(f'patch HSPP {res[0]} {res[1]}')
            pycharmm.charmm_script(f'rename resn HSP sele segid {res[0]} .and. resid {res[1]} end')
            pycharmm.charmm_script(f'patch {res[2]}P {res[0]} {res[1]}')
            pycharmm.charmm_script(f'HBUILD sele segid {res[0]} .and. hydrogen end')
            titratable_dict.append([res[0], res[1], 'HSP'])
    pycharmm.charmm_script(f'HBuild')
    write.psf_card(f'{input_folder}/tmp/{seg_id}.psf')
    write.coor_card(f'{input_folder}/tmp/{seg_id}.crd')

# Filter titratable_dict based on selection criteria
filtered_titratable_dict = [
    item for item in titratable_dict 
    if should_patch_residue(item[0], item[1], item[2], parsed_selected_patching)
]
print(f"Filtered {len(titratable_dict) - len(filtered_titratable_dict)} residues based on selection criteria.")
titratable_dict = filtered_titratable_dict # Replace with the filtered list
# Print the final list of residues to be patched for user confirmation/logging
print("Final list of residues selected for patching:")
if titratable_dict:
    for seg, resid, resn in titratable_dict:
        print(f"  Segment: {seg}, Residue ID: {resid}, Residue Name: {resn}")
else:
    print("  No residues selected for patching based on criteria.")

f=open(f'{input_folder}/patches.dat', 'w')
f.write('SEGID,RESID,PATCH,SELECT,ATOMS,TAG\n')
f.close()    

# rename HSE and HSD to HSP in universe uni
uni.universe.loc[uni.universe['res_name'] == 'HSD', 'res_name'] = 'HSP'
uni.universe.loc[uni.universe['res_name'] == 'HSE', 'res_name'] = 'HSP'
    
def patching(input_folder, patches_topology, titratable_dict, current_titratable_residue_index, f):
    pycharmm.charmm_script('AUTO NOPAtch')
    for residue in titratable_dict:
        seg_id, resid, resname = residue
        current_titratable_residue_index += 1
    # if it first occurrence of seg_id in titratable_dict
        if [res[0] for res in titratable_dict].index(seg_id) == titratable_dict.index(residue):
            read.psf_card(f'{input_folder}/tmp/{seg_id}.psf')
            read.coor_card(f'{input_folder}/tmp/{seg_id}.crd')
    # Replicate residue
        charmm_command = f'REPLicate {resid} NREP {len(patches_topology.patches[resname])+1} SETUP -\n'
        select_command = ''
        for atom in patches_topology.atom_groups[resname]:
            if patches_topology.atom_groups[resname].index(atom) == 0:
                select_command = f'SELEct segid {seg_id} .and. resid {resid} .and. ('
            select_command += f'-\ntype {atom}'
            if patches_topology.atom_groups[resname].index(atom) == len(patches_topology.atom_groups[resname]) - 1:
                select_command += ') END \n'
            else:
                select_command += ' .or. '
        pycharmm.charmm_script(charmm_command + select_command)
    # Delete atoms in original residue
        charmm_command = f'DELEte ATOMs '
        pycharmm.charmm_script(charmm_command + select_command)
        j = 1
        for patch in [resname+'O']+patches_topology.patches[resname]:
        # Rename replica's resname to patch name
            charmm_command = f'! Working on {patch} {seg_id}:{resid}{j}:{resid}\n'
            if patch != resname+'O':
                charmm_command += f'PATCH {patch} {resid}{j} {resid} SETUP\n'
                # charmm_command += f'AUTO ANGLes DIHEdrals sele segid {resid}{j} end\n'
                # charmm_command += f'IC PARAM sele segid {resid}{j} .and. resid {resid}\n'
                # charmm_command += f'IC FILL sele segid {resid}{j} .and. resid {resid}\n'
                
                if len(patches_topology.atom_groups[patch]) - len(patches_topology.atom_groups[resname]) > 0:
                    f.write(f'{seg_id},{resid},{patch},s{current_titratable_residue_index}s{j},{" ".join(map(str,patches_topology.atom_groups[patch]))},UNEG {patches_topology.pka[patch]}\n')
                else:
                    f.write(f'{seg_id},{resid},{patch},s{current_titratable_residue_index}s{j},{" ".join(map(str,patches_topology.atom_groups[patch]))},UPOS {patches_topology.pka[patch]}\n') 
                if resname == 'GLU' or resname == 'ASP':
                    charmm_command += ''
                    # charmm_command += f'HBUILD SELE segid {resid}{j} .and. hydrogen .and. (type HE* .or. type HD*) END\n'
                #     f.write(f'{seg_id},{resid},{patch},s{i}s{j},UNEG {patches_topology.pka[patch]}\n')
                # else:
                #     f.write(f'{seg_id},{resid},{patch},s{i}s{j},UPOS {patches_topology.pka[patch]}\n')
            else:
                f.write(f'{seg_id},{resid},{patch},s{current_titratable_residue_index}s{j},{" ".join(map(str,patches_topology.atom_groups[patch[:-1]]))},NONE\n')
            charmm_command += f'RENAMe RESName {patch} SELE segid {resid}{j} END\n'
        # Join replica to original segment
        # ? RENUmber gives last free resid in PROA
            charmm_command += f'JOIN {seg_id} {resid}{j}\n'
            j += 1
            pycharmm.charmm_script(charmm_command)
        # ask user to continue
        pycharmm.charmm_script(f'AUTO ANGLES DIHEDRALS sele segid {seg_id} .and. resid {resid} end\n')
        pycharmm.charmm_script(f'IC PARAM sele segid {seg_id} .and. resid {resid} end\n')
        pycharmm.charmm_script(f'IC FILL PRESERVE sele segid {seg_id} .and. resid {resid} end\n')
        pycharmm.charmm_script(script='REPLIcate RESEt')
        combo = itertools.combinations([resname+'O']+patches_topology.patches[resname], 2)
        for pair in combo:
            pycharmm.charmm_script(
                f'DELETE CONN ATOMs SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[0]} END SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[1]} END\n')
        # u = universe().update_universe()
        # make selection
        # u = u.loc[(u['seg_id'] == seg_id) & (u['res_id'] == resid)]
        # print(u)
        # input('Press Enter to continue...')
        
    return current_titratable_residue_index

def deleteConnectingAtoms(patches_topology, titratable_dict):
    
    if not titratable_dict:
        return
    pycharmm.charmm_script(script='REPLIcate RESEt')
    for residue in titratable_dict:
            resid = residue[1]
            resname = residue[2]
            seg_id = residue[0]
            combo = itertools.combinations([resname+'O']+patches_topology.patches[resname], 2)
            for pair in combo:
                pycharmm.charmm_script(
                f'DELEte CONN ATOMS SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[0]} END SELEct segid {seg_id} .and. resid {resid} .and. resname {pair[1]} END\n')
    return
global_titratable_residue_idx = 0 
for seg_id in seg_ids:
    f = open(f'{input_folder}/patches.dat', 'a')
    psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
    read.psf_card(f'{input_folder}/tmp/{seg_id}.psf')
    read.coor_card(f'{input_folder}/tmp/{seg_id}.crd')
    sublist = [residue for residue in titratable_dict if residue[0] == seg_id]
    print(sublist)
    global_titratable_residue_idx = patching(input_folder, patches_topology, sublist, global_titratable_residue_idx, f)
    # deleteConnectingAtoms(patches_topology, sublist)
    ic.build()
    ic.prm_fill(replace_all=True)
    write.coor_card(f'{input_folder}/tmp/{seg_id}.crd')
    write.psf_card(f'{input_folder}/tmp/{seg_id}.psf')
    f.close()

psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())

for seg_id in seg_ids:
    if seg_ids.index(seg_id) == 0:
        read.psf_card(f'{input_folder}/tmp/{seg_id}.psf')
        read.coor_card(f'{input_folder}/tmp/{seg_id}.crd')
    else:
        read.psf_card(f'{input_folder}/tmp/{seg_id}.psf', append=True)
        read.coor_card(f'{input_folder}/tmp/{seg_id}.crd', append=True)

# Apply bonded disulfide bonds
print('Theese residues are bonded:')
# for bond in bonded_res:
    # lingo.charmm_script(f'patch DISU {bond[0]} {bond[1]} {bond[3]} {bond[4]}')
write.coor_pdb(f'{input_folder}/system.pdb')
write.psf_card(f'{input_folder}/system.psf')
write.coor_card(f'{input_folder}/system.crd') 
# Apply HMR
if hmr and hmr_waters:
    segids = psf.get_segid()
    print(segids)
    # for segid in segids:
        # psf.set_hmr(segid)
    psf.hmr(resnames_exclude=[])
    print('HMR enabled for system')
    write.psf_card(f'{input_folder}/system_hmr.psf')
    write.coor_card(f'{input_folder}/system_hmr.crd')
elif hmr and not hmr_waters:
    segids = psf.get_segid()
    #delete SOLV and IONS segments
    segids = [segid for segid in segids if segid not in ['SOLV', 'IONS']]
    print('HMR enabled for system without waters')
    psf.hmr()
    write.psf_card(f'{input_folder}/system_hmr.psf')
    write.coor_card(f'{input_folder}/system_hmr.crd')
    




# In[7]: Calculate size and FFT of system

# box size is in the second line of box.dat, it's three numbers separated by spaces
if not os.path.exists(f'{input_folder}/box.dat'):
    stats = pycharmm.coor.stat()
    xmax = stats['xmax']
    xmin = stats['xmin']
    ymax = stats['ymax']
    ymin = stats['ymin']
    zmax = stats['zmax']
    zmin = stats['zmin']
    Xinit = int((xmax - xmin))
    Yinit = int((ymax - ymin))
    Zinit = int((zmax - zmin))
    max_size = max(Xinit, Yinit, Zinit)
    box_size = [max_size, max_size, max_size]
    type = 'CUBic'
    angles = [90.0, 90.0, 90.0]
    with open(f'{input_folder}/box.dat', 'w') as f:
        f.write(f'{type}\n')
        f.write(f'{box_size[0]} {box_size[1]} {box_size[2]}\n')
        f.write(f'{angles[0]} {angles[1]} {angles[2]}\n')
    
box_size = open(f'{input_folder}/box.dat', 'r').readlines()[1].split()
box_size = [float(size) for size in box_size]
with open(f'{input_folder}/size.dat', 'w') as f:
    f.write(str(box_size[0]))
    


def fft_number(n):
    fft_numbers = [1]
    i2 = i3 = i5 = 0

    while fft_numbers[-1] < n:
        next_fft = min(fft_numbers[i2]*2, fft_numbers[i3]*3, fft_numbers[i5]*5)
        if divmod(next_fft, 2)[1] == 0:
            fft_numbers.append(next_fft)

        if next_fft == fft_numbers[i2]*2:
            i2 += 1
        if next_fft == fft_numbers[i3]*3:
            i3 += 1
        if next_fft == fft_numbers[i5]*5:
            i5 += 1

    for num in fft_numbers:
        if num >= n:
            return num
    return None     
fft = []
for i in box_size:
    fft.append(fft_number(i))

    

with open(f'{input_folder}/fft.dat', 'w') as f:
    f.write(' '.join([str(i) for i in fft]))


######



i = 0
f = open(f'{input_folder}/select.str', 'w')
for residue in titratable_dict:
    i+=1
    j=0
    seg_id = residue[0]
    resid = residue[1]
    resname = residue[2]
    patches = [resname+'O']+patches_topology.patches[resname]
    for patch in patches:
        j+=1
        f.write(f'DEFIne s{i}s{j} SELEct segid {seg_id} .and. resid {resid} .and. resname {patch} END\n')
f.close()

# read.stream(f'{input_folder}/select.str')


# In[8]: Save selections

def open_clog(log_unit=30, log_file=f'{input_folder}/tmp.clog'):
    clog = pycharmm.CharmmFile(
        file_name=log_file, file_unit=log_unit, read_only=False, formatted=True)
    pycharmm.charmm_script(f'OUTUnit {log_unit}')
    return clog


def flush_clog(log_file=f'{input_folder}/tmp.clog'):
    with open(log_file, 'w') as f:
        f.write('')


def close_clog(log_unit=30, log_file=f'{input_folder}/tmp.clog'):
    pycharmm.charmm_script(f'OUTUnit 6')
    pycharmm.charmm_script(f'CLOSE UNIT {log_unit}')


def read_clog(log_file=f'{input_folder}/tmp.clog'):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        lines_tmp = []
        for line in lines:
            if line.startswith('\x00') == False:
                lines_tmp.append(line)
        lines = lines_tmp
    return lines


def store_selection(rewrite=True):
    stored = select.get_stored_names()
    n_atoms = psf.get_natom()
    settings.set_verbosity(5)
    if rewrite:
        f = open(f'{input_folder}/selections.dat', 'w')
        f.close()
    for store in stored:
        selection = (False,) * n_atoms
        clog = open_clog()
        pycharmm.charmm_script(f'PRINt COOR SELEct {store} END')
        close_clog()
        clog.close()
        output = read_clog()
        flush_clog()
        for line in output:
            if len(line.split()) < 2:
                continue
            if line.split()[0].isdigit() and line.split()[1].isdigit():
                index = int(line.split()[0]) - 1
                selection = selection[:index] + (True,) + selection[index+1:]

        f = open(f'{input_folder}/selections.dat', 'a')
        selection_str = ''.join(['1' if s else '0' for s in selection])
        f.write(f'{store.lower()}: {selection_str}\n')
        print(f'Saved {store} selection')
        f.close()
        os.remove(f'{input_folder}/tmp.clog')


# store_selection(rewrite=True)


# remove tmp folder
shutil.rmtree(f'{input_folder}/tmp')

# move everything in the input_folder to input_folder/prep
os.makedirs(f'{input_folder}/prep', exist_ok=True)
for file in os.listdir(input_folder):
    # Skip the prep folder, any folders, and .out/.err files
    file_path = os.path.join(input_folder, file)
    if os.path.isdir(file_path):
        continue
    if file.endswith('.out') or file.endswith('.err'):
        continue
    shutil.move(file_path, os.path.join(input_folder, 'prep', file))
print('Patching completed. Files moved to input_folder/prep')
