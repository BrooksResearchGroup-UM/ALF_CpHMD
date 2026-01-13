# Copyright (c) 2024, Stanislav Cherepanov

import os
import random
import numpy as np
import pandas as pd
import pycharmm
import pycharmm.crystal
import pycharmm.nbonds
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.write as write
import pycharmm.energy as energy
import pycharmm.coor as coor
import pycharmm.psf as psf
import pycharmm.crystal as crystal
import pycharmm.nbonds as nbonds
import pycharmm.image as image
import pycharmm.minimize as minimize
import pycharmm.select as select
import operator
import argparse

# check if i have gpu
import torch
if torch.cuda.is_available():
    gpu_available = True
else:
    gpu_available = False

#######
filename = 'preparation/pdb/lys'
output = 'lys'
XTLtype = 'OCTAhedral'
pad = 10.0
salt_concentration = 0.10 # (M)
pos_ion = 'POT'
neg_ion = 'CLA'
skip_ions = False
T = 298.15 #(K)
min_ion_distance = 5  #(A) Minimum distance between ions, adjust as needed
ion_method = 'SLTCAP'   # default ion placement algorithm
topology_path = 'toppar'
topology_files = [
    'top_all36_prot.rtf',
    'par_all36m_prot.prm',
    'toppar_water_ions.str',
    'top_all36_na.rtf',
    'par_all36_na.prm'
]
#######


def parse_terminal_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some parameters for a simulation setup.')

    # Define the expected command line arguments
    parser.add_argument('-i', '--filename', type=str, default='preparation/pdb/glu', help='Path to the input file')
    parser.add_argument('-o', '--output', type=str, default='glu', help='Output file name')
    parser.add_argument('--XTLtype', type=str, default='OCTAhedral', help='Crystal type')
    parser.add_argument('--pad', type=float, default=10.0, help='Padding value')
    parser.add_argument('-s', '--salt_concentration', type=float, default=0.10, help='Salt concentration in mM')
    parser.add_argument('--pos_ion', type=str, default='POT', help='Positive ion type')
    parser.add_argument('--neg_ion', type=str, default='CLA', help='Negative ion type')
    parser.add_argument('-t', '--temperature', type=float, default=303, help='Temperature in Kelvin')
    parser.add_argument('--no-ions', default=False, action='store_true', help='Skip ionization step')
    parser.add_argument('--ion-method', type=str, default='SLTCAP',
                        choices=['AN', 'SLTCAP'],
                        help='Ion placement algorithm: AN (Add‑then‑Neutralize) or SLTCAP')

    # Parse the arguments
    args = parser.parse_args()

    # Assign global variables
    global filename, output, XTLtype, pad, salt_concentration, pos_ion, neg_ion, temperature, skip_ions, ion_method
    filename = args.filename
    output = args.output
    XTLtype = args.XTLtype
    pad = args.pad
    salt_concentration = args.salt_concentration
    pos_ion = args.pos_ion
    neg_ion = args.neg_ion
    temperature = args.temperature
    skip_ions = args.no_ions
    ion_method = args.ion_method.upper()

# Execute the parsing function only when the script is run as the main module
if __name__ == "__main__":
    parse_terminal_arguments()
    # Now you can use the variables in your code as they have been assigned globally
    print(f"Filename: {filename}")
    print(f"Output: {output}")
    print(f"XTLtype: {XTLtype}")
    print(f"Pad: {pad}")
    print(f"Salt Concentration: {salt_concentration} mM")
    print(f"Positive Ion: {pos_ion}")
    print(f"Negative Ion: {neg_ion}")
    print(f"Temperature: {temperature} K")
    print(f"Skip Ionization: {skip_ions}")
    


def water_density(temp):
    temp = temp - 273.15 # Convert to Celsius
    density = (999.8395 + 6.7914e-2 * temp - 9.0894e-3 * temp**2 +
               1.0171e-4 * temp**3 - 1.2846e-6 * temp**4 +
               1.1592e-8 * temp**5 - 5.0125e-11 * temp**6)
    return density


def get_box_parameters(XTLtype, stats, pad):
    xmax = stats['xmax']
    xmin = stats['xmin']
    ymax = stats['ymax']
    ymin = stats['ymin']
    zmax = stats['zmax']
    zmin = stats['zmin']

    Xinit = int((xmax - xmin) + 2 * pad) + 1
    Yinit = int((ymax - ymin) + 2 * pad) + 1
    Zinit = int((zmax - zmin) + 2 * pad) + 1

    XTLtype = XTLtype.upper()

    if XTLtype == 'CUBIC':
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 90.0
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif XTLtype == 'TETRAGONAL':
        A = B = max(Xinit, Yinit)
        C = Zinit
        Alpha = Beta = Gamma = 90.0
        BoxSizeX = BoxSizeY = A * 1.1
        BoxSizeZ = C * 1.1

    elif XTLtype == 'ORTHORHOMBIC':
        A = Xinit
        B = Yinit
        C = Zinit
        Alpha = Beta = Gamma = 90.0
        BoxSizeX = A * 1.1
        BoxSizeY = B * 1.1
        BoxSizeZ = C * 1.1

    elif XTLtype == 'MONOCLINIC':
        A = Xinit
        B = Yinit
        C = Zinit
        Alpha = Gamma = 90.0
        Beta = 70.0  # Assuming a typical value, adjust as needed
        BoxSizeX = A * 1.1
        BoxSizeY = B * 1.1
        BoxSizeZ = C * 1.1

    elif XTLtype == 'TRICLINIC':
        A = Xinit
        B = Yinit
        C = Zinit
        Alpha = 60.0  # Assuming typical values, adjust as needed
        Beta = 70.0
        Gamma = 80.0
        BoxSizeX = A * 1.1
        BoxSizeY = B * 1.1
        BoxSizeZ = C * 1.1

    elif XTLtype == 'HEXAGONAL':
        A = B = max(Xinit, Yinit)
        C = Zinit
        Alpha = Beta = 90.0
        Gamma = 120.0
        BoxSizeX = BoxSizeY = A * 1.1
        BoxSizeZ = C * 1.1

    elif XTLtype == 'RHOMBOHEDRAL':
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 67.0  # Assuming a typical value, adjust as needed
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif XTLtype == 'OCTAHEDRAL':
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Beta = Gamma = 109.4712206344907
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1

    elif XTLtype == 'RHDO':
        A = B = C = max(Xinit, Yinit, Zinit)
        Alpha = Gamma = 60.0
        Beta = 90.0
        BoxSizeX = BoxSizeY = BoxSizeZ = A * 1.1
        
    else:
        raise ValueError(f"Invalid XTLtype: {XTLtype}")

    # round to 3 decimal places
    
    BoxSizeX, BoxSizeY, BoxSizeZ = round(BoxSizeX, 3), round(BoxSizeY, 3), round(BoxSizeZ, 3)

    return A, B, C, Alpha, Beta, Gamma, BoxSizeX, BoxSizeY, BoxSizeZ


# def universe() -> pd.DataFrame:
#     try:
#         select_all = pycharmm.SelectAtoms().all_atoms()
#     except Exception as e:
#         raise ValueError('No atoms in system') from e
#     
#     crds = coor.get_positions()
#     charges = psf.get_charges()
#     
#     universe = pd.DataFrame({
#             'index': select_all._atom_indexes,
#             'atom_type': select_all._atom_types,
#             'res_name': select_all._res_names,
#             'res_id': select_all._res_ids,
#             'seg_id': select_all._seg_ids,
#             'chem_type': select_all._chem_types,
#             'x': crds['x'].values,
#             'y': crds['y'].values,
#             'z': crds['z'].values,
#             'charge': charges
#         })
#     
#     universe.set_index('index', inplace=True)
#     return universe


def read_topology_files(verbose=True):
    if not verbose:
        lingo.charmm_script('prnlev 0')
    
    prm_files = [file for file in topology_files if file.endswith('.prm')]
    rtf_files = [file for file in topology_files if file.endswith('.rtf')]
    str_files = [file for file in topology_files if file.endswith('.str')]
    
    if rtf_files:
        read.rtf(os.path.join(topology_path, rtf_files[0]))
        for file in rtf_files[1:]:
            read.rtf(os.path.join(topology_path, file), append=True)
            
    if prm_files:
        read.prm(os.path.join(topology_path, prm_files[0]), flex=True)
        for file in prm_files[1:]:
            read.prm(os.path.join(topology_path, file), flex=True, append=True)
            
    for file in str_files:
        lingo.charmm_script(f'stream {os.path.join(topology_path, file)}')
    
    if not verbose:
        lingo.charmm_script('prnlev 5')
    return None


######
#Create output directory
if not os.path.exists(output):
    os.makedirs(output)


read_topology_files()
lingo.charmm_script('IOFOrmat EXTEnded')

if gpu_available:
    lingo.charmm_script('blade on')

# Reading the structure
read.psf_card(f'{filename}.psf')
if os.path.exists(f'{filename}.crd'):
    read.coor_card(f'{filename}.crd')
else:
    read.pdb(f'{filename}.pdb', resid = True)
    write.coor_card(f'{filename}.crd')
# Calculate the molecule charge
charge = sum(psf.get_charges())

# Orient the structure
coor.orient()

#Calculate the box size
stats = coor.stat()

A, B, C, Alpha, Beta, Gamma, BoxSizeX, BoxSizeY, BoxSizeZ = get_box_parameters(XTLtype, stats, pad)
print('Box size:', BoxSizeX, BoxSizeY, BoxSizeZ)
print('Molecule size:', A, B, C)
xcen = ycen = zcen = 0.0
# Delete atoms
psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
L = 18.8560

# Number of boxes along XYZ-directions
Xnum = int(BoxSizeX / L) + 1
Ynum = int(BoxSizeY / L) + 1
Znum = int(BoxSizeZ / L) + 1


# read.sequence_string('216 TIP3')
lingo.charmm_script('read sequ TIP3 216')
gen.new_segment(seg_name='W000', first_patch='NONE', last_patch='NONE', setup_ic=True, noangle=True, nodihedral=True)
read.coor_card(os.path.join(topology_path, 'tip216.crd'))
stats = coor.stat(selection=pycharmm.SelectAtoms().by_atom_type('OH2'))


# Translate the water box
# move = (3*(L/2)**2)**0.5 
# lingo.charmm_script(f'coor translate xdir 1 ydir 1 zdir 1 dist {move}') 
lingo.charmm_script(f'coor translate xdir 1 dist {L/2}')
lingo.charmm_script(f'coor translate ydir 1 dist {L/2}')
lingo.charmm_script(f'coor translate zdir 1 dist {L/2}')           
stats = coor.stat(selection=pycharmm.SelectAtoms().by_atom_type('OH2'))

#Planar water box unit (XY) - variant 1
for J2 in range(1, Ynum + 1):
    for J1 in range(1, Xnum + 1):
        wsegid = str((J2 - 1) * Xnum + J1)
        # read.sequence_string('TIPS 216') # Doesn't work
        lingo.charmm_script('read sequ TIP3 216')
        gen.new_segment(seg_name=f'W{wsegid}', first_patch='NONE', last_patch='NONE', setup_ic=True, noangle=True, nodihedral=True)
        lingo.charmm_script(f'coor duplicate sele segid W000 end sele segid W{wsegid} end')
        X = L * (J1 - 1)
        Y = L * (J2 - 1)
        pycharmm.charmm_script(f'coor translate xdir {X} ydir {Y} sele segid W{wsegid} end')
        if J1 == 1 and J2 == 1:
            lingo.charmm_script(f'RENAme segid SOLV sele segid W{wsegid} end')
        else:
            lingo.charmm_script(f'JOIN SOLV W{wsegid} RENUmber')
        
if psf.get_natom == 0:
    raise Exception('No waters in the box')


psf.delete_atoms(pycharmm.SelectAtoms(seg_id='W000'))
########
# Working on the Z-direction - variant 1
N_water = psf.get_nres()
for J3 in range(2, Znum + 1):
    Z = L * (J3 - 1)
    lingo.charmm_script(f'read sequ TIP3 {N_water}')
    gen.new_segment(seg_name=f'W{J3}', first_patch='NONE', last_patch='NONE', setup_ic=True, noangle=True, nodihedral=True)
    lingo.charmm_script(f'coor duplicate sele segid SOLV .and. resid 1:{N_water} end sele segid W{J3} end')
    lingo.charmm_script(f'coor translate zdir {Z} sele segid W{J3} end')
    lingo.charmm_script(f'JOIN SOLV W{J3} RENUmber')
########  
    
########
# Working on the Z-direction - variant 2
# write.coor_card('pdb/water_tmp.crd') 
# write.coor_pdb('pdb/water_tmp.pdb')
# psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
# for J3 in range(1, Znum + 1):
#     read.sequence_coor('pdb/water_tmp.crd')
#     # read.sequence_pdb('pdb/water_tmp.pdb')
#     gen.new_segment(seg_name = f'W{J3}', first_patch='NONE', last_patch='NONE', setup_ic=True, noangle=True, nodihedral=True)
#     read.coor_card('pdb/water_tmp.crd', append=True) #! crd has limitation of 1944 water residues
#     # read.pdb('pdb/water_tmp.pdb', append=True)
#     Z = L * (J3 - 1)
#     pycharmm.charmm_script(f'coor translate zdir {Z} sele segid W{J3} end')
#     if J3 == 1:
#         lingo.charmm_script(f'RENAme segid SOLV sele segid W{J3} end')
#     else:
#         lingo.charmm_script(f'JOIN SOLV W{J3} RENUmber')
########

lingo.charmm_script(f'''
    delete atom sele .byres. ( ( type OH2 ) .and. -
        ( prop Z .gt. {BoxSizeZ} ) ) end
''')
# This can be an approach here, but works uncorrect and requires __invert__ method
Z_delete = pycharmm.SelectAtoms().by_property('Z',operator.gt,BoxSizeZ).__invert__()
# Z_delete = Z_delete & pycharmm.SelectAtoms().by_atom_type('OH2')
# Z_delete = Z_delete.whole_residues()
# psf.delete_atoms(Z_delete)

stats = coor.stat(selection=pycharmm.SelectAtoms().by_atom_type('OH2'))
coor.orient(noro=True)
stats = coor.stat(selection=pycharmm.SelectAtoms().by_atom_type('OH2'))



# Shaping the box
lingo.charmm_script(f'coor convert symmetric aligned {BoxSizeX} {BoxSizeY} {BoxSizeZ} {Alpha} {Beta} {Gamma}')
lingo.charmm_script('coor copy comp')

lingo.charmm_script(f'''crystal define {XTLtype} {BoxSizeX} {BoxSizeY} {BoxSizeZ} {Alpha} {Beta} {Gamma} {xcen} {ycen} {zcen}''')
crystal.build(2)

# pycharmm.charmm_script('open unit 90 write form name crystal.img')
# pycharmm.charmm_script('crystal write card unit 90')


image.setup_residue(0,0,0,'TIP3')
# pycharmm.charmm_script(f'image byres xcen {xcen} ycen {ycen} zcen {zcen}') - old


# nbonds.configure(cutnb=3, ctonnb=2, ctofnb=3, cutim=3)

lingo.charmm_script(f'nbonds ctonnb 2.0 ctofnb 3.0 cutnb 3.0 cutim 3.0 wmin 0.01 fswitch vswitch')

lingo.charmm_script('crystal free')

lingo.charmm_script('coor diff comp')

lingo.charmm_script('''
    delete atom sele .byres. ( ( prop Xcomp .ne. 0 ) .or. -
        ( prop Ycomp .ne. 0 ) .or. -
        ( prop Zcomp .ne. 0 ) ) end
''')
write.coor_card(os.path.join(output, 'waterbox.crd'))
write.psf_card(os.path.join(output, 'waterbox.psf'))
psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())
with open(os.path.join(output,'box.dat'), 'w', encoding='utf-8') as f:
    f.write(f'{XTLtype}\n')
    f.write(f'{BoxSizeX} {BoxSizeY} {BoxSizeZ}\n')
    f.write(f'{Alpha} {Beta} {Gamma}\n')

read.psf_card(os.path.join(filename+'.psf'))
read.coor_card(os.path.join(filename+'.crd'))
coor.orient()
stats = coor.stat()
N_molecule = psf.get_nres()
write.psf_card(f'{output}/molecule.psf', title='Molecule with Minimization (part with waterbox.*)', select='segid MOL end')
read.psf_card(os.path.join(output, 'waterbox.psf'), append=True)
read.coor_card(os.path.join(output, 'waterbox.crd'), append=True)

molecule = pycharmm.SelectAtoms().by_seg_id('SOLV').__invert__()
water = pycharmm.SelectAtoms().by_seg_id('SOLV') & pycharmm.SelectAtoms().by_atom_type('OH2')
water = (water & molecule.around(2.6)).whole_residues()
psf.delete_atoms(water)


# OLD WAY
# pycharmm.charmm_script(' define MOL sele .not. segid SOLV end')
# pycharmm.charmm_script('coor stat sele MOL end')
# pycharmm.charmm_script('coor stat sele type OH2 .and. segid SOLV end')
# pycharmm.charmm_script('delete atom sele .byres. ( ( type OH2 .and. segid SOLV) .and. (MOL .around. 2.6) ) end')
# pycharmm.charmm_script('join SOLV RENUmber')
# pycharmm.charmm_script('coor stat sele type OH2 .and. segid SOLV end')
N_water = psf.get_nres() - N_molecule

########
#ionization
########
if not skip_ions:
    print(f'Number of waters: {N_water}')
    print(f'Total of solute charge: {charge}')

    lingo.charmm_script('crystal free')
    lingo.charmm_script(f'''crystal define {XTLtype} {A} {B} {C} {Alpha} {Beta} {Gamma} {xcen} {ycen} {zcen}''')
    crystal.build(pad)


    # Calculate ions
    M_water = 18.01528  # g/mol
    Rho_water = water_density(T)  # g cm‑3
    print(f'Water density: {Rho_water} g/cm^3')

    # Number of ion pairs that would be present at bulk concentration
    N_0 = (N_water * M_water * salt_concentration) / Rho_water

    if ion_method == 'AN':          # Add‑then‑Neutralize
        N_pos = round(N_0)
        N_neg = round(N_0 + charge)
    elif ion_method == 'SLTCAP':    # Screening Layer Tally by Container Average Potential
        factor = (1 + (charge / (2 * N_0))) ** 0.5
        N_pos = round(N_0 * factor - charge / 2)
        N_neg = round(N_0 * factor + charge / 2)
    else:
        raise ValueError(f'Unknown ion_method: {ion_method}')

    N_ion = N_pos + N_neg
    print(f'Ion placement algorithm: {ion_method}')

    # Generate co-ions
    if N_ion > 0:
        read.sequence_string(f'{pos_ion} ' *N_pos + f'{neg_ion} ' * N_neg)
        gen.new_segment(seg_name='IONS', first_patch='NONE', last_patch='NONE', setup_ic=True, noangle=True, nodihedral=True)

    # Find N_ion random positions in water_coords which (pandas with X,Y,Z) are away from any atom in molecule_coords (pandas with X,Y,Z) 
    for i in range(1, N_ion + 1):
        search = True
        while search:
            # Get current system state directly instead of using universe()
            all_crds_df = coor.get_positions() # Returns DF with 1-based 'atom_index', 'x', 'y', 'z'
            
            sel_all_for_psf = pycharmm.SelectAtoms().all_atoms()
            if not sel_all_for_psf._atom_indexes:
                print("Warning: No atoms found in PSF for ion placement. Stopping ion placement.")
                search = False # Stop trying to place this ion
                N_ion = i -1 # Adjust N_ion to ions successfully placed so far
                break # Break from while search loop

            current_psf_df = pd.DataFrame({
                'atom_index': sel_all_for_psf._atom_indexes, # 1-based
                'res_name': sel_all_for_psf._res_names,
                'res_id': sel_all_for_psf._res_ids,
                'atom_type': sel_all_for_psf._atom_types,
                'seg_id': sel_all_for_psf._seg_ids
            })
            
            # Merge PSF info with coordinates. all_crds_df.index is 1-based 'atom_index'
            system_df = pd.merge(current_psf_df, all_crds_df, left_on='atom_index', right_index=True)

            water_oh2_for_resids = system_df[
                (system_df['seg_id'] == 'SOLV') &
                (system_df['res_name'] == 'TIP3') &
                (system_df['atom_type'] == 'OH2')
            ]
            if water_oh2_for_resids.empty:
                print("Warning: No TIP3 OH2 atoms found in SOLV segment for ion placement. Stopping ion placement for this ion.")
                search = False # Stop trying for this ion
                N_ion = i-1
                break # Break from while search

            water_resids = water_oh2_for_resids['res_id'].unique()
            if not water_resids.size:
                 print("Warning: No water residues available to replace with ions. Stopping ion placement for this ion.")
                 search = False # Stop trying for this ion
                 N_ion = i-1
                 break # from while search

            # Molecule coordinates (any segment not SOLV or IONS)
            # Assuming ions generated so far are in 'IONS' seg_id
            molecule_seg_ids = [sid for sid in system_df['seg_id'].unique() if sid not in ['SOLV', 'IONS']]
            if molecule_seg_ids:
                molecule_sel_df = system_df[system_df['seg_id'].isin(molecule_seg_ids)]
                molecule_coords_np = molecule_sel_df[['x', 'y', 'z']].values
            else:
                molecule_coords_np = np.empty((0,3))

            # Existing ion coordinates
            ion_sel_df = system_df[system_df['seg_id'] == 'IONS']
            if not ion_sel_df.empty:
                # Exclude the current ion being placed if it has placeholder coords already
                # Assuming resid 'i' in 'IONS' is the current ion being placed.
                # Its coords might be 0,0,0 or from a previous failed attempt.
                # For distance check, we only care about *other* already placed ions.
                other_ions_sel_df = ion_sel_df[ion_sel_df['res_id'] != i] # i is the current ion's residue ID
                if not other_ions_sel_df.empty:
                    ion_coords_np = other_ions_sel_df[['x', 'y', 'z']].values
                else:
                    ion_coords_np = np.empty((0,3))
            else:
                ion_coords_np = np.empty((0,3))


            random_water_res_id = random.choice(water_resids)
            # Get all atoms of the chosen water residue
            chosen_water_atoms_df = system_df[
                (system_df['seg_id'] == 'SOLV') & 
                (system_df['res_id'] == random_water_res_id)
            ]
            
            if chosen_water_atoms_df.empty: # Should not happen if random_water_res_id is valid
                continue

            # Use the geometric center of the chosen water molecule as the target position for the ion
            target_ion_pos_xyz = chosen_water_atoms_df[['x', 'y', 'z']].mean().values

            # Check distance from molecule atoms to the target ion position
            if molecule_coords_np.shape[0] > 0:
                distances_to_mol = np.linalg.norm(molecule_coords_np - target_ion_pos_xyz, axis=1)
                if np.any(distances_to_mol <= 2.6): # Original distance threshold
                    continue # Too close to molecule, try another water

            # Check distance from existing *other* ions to the target ion position
            if ion_coords_np.shape[0] > 0:
                distances_to_ions = np.linalg.norm(ion_coords_np - target_ion_pos_xyz, axis=1)
                if np.any(distances_to_ions < min_ion_distance):
                    continue # Too close to another already placed ion, try another water
            
            # If all checks pass:
            x_place, y_place, z_place = target_ion_pos_xyz
            # Place the current ion (residue 'i' in segment 'IONS')
            lingo.charmm_script(f'coor set xdir {x_place:.4f} ydir {y_place:.4f} zdir {z_place:.4f} sele segid IONS .and. resid {i} end')
            
            # Delete the chosen water molecule
            # Ensure res_id is treated as a string for SelectAtoms if necessary, CHARMM command is more direct
            lingo.charmm_script(f'dele atom sele resid {random_water_res_id} .and. segid SOLV end')
            # sel_water_to_delete = pycharmm.SelectAtoms(res_id=str(random_water_res_id), seg_id="SOLV")
            # pycharmm.psf.delete_atoms(sel_water_to_delete) # This is also a good option

            search = False # Successfully placed this ion
        
        if search : # This means the while loop broke due to no waters/PSF, and 'search' is still true
            print(f"Could not place ion {i}. Stopping further ion placement.")
            break # Break from the main for loop over ions


    lingo.charmm_script('JOIN SOLV RENUmber')
    # nbonds.configure(ctonnb=10, ctofnb=12, cutnb=14, cutim=14)
    lingo.charmm_script('nbonds ctonnb 10 ctofnb 12 cutnb 14 cutim 14 wmin 1.0 fswitch vswitch')
    lingo.charmm_script(' define MOL sele .not. (segid SOLV .or. segid IONS) end')
energy.show()
energy_ = energy.get_total()
for force in [100,50,25,5]:
    pycharmm.charmm_script(f'cons harm force {force} sele MOL .and. (.not. hydrogen) end')
    minimize.run_sd(nstep=50)
    minimize.run_abnr(nstep=100)
    lingo.charmm_script('cons harm clear')
minimize.run_sd(nstep=1000)
new_energy = energy.get_total()
delta_energy = energy_ - new_energy
print(f'Energy before minimization: {energy_} kcal/mol')
print(f'Energy after minimization: {new_energy} kcal/mol')
print(f'Energy change after minimization: {delta_energy} kcal/mol')
# minimize.run_abnr(nstep=1000)

write.coor_card(f'{output}/solvated.crd')
write.psf_card(f'{output}/solvated.psf')
write.coor_pdb(f'{output}/solvated.pdb')
write.coor_card(f'{output}/molecule.crd', title='Molecule with Minimization (part with waterbox.*)', select='.not. (segid SOLV .or. segid IONS) end')
psf.delete_atoms(molecule)
write.coor_card(f'{output}/waterbox.crd', title=f'{XTLtype} Waterbox with box size {BoxSizeX}:{BoxSizeY}:{BoxSizeZ}', select='segid SOLV .or. segid IONS end')
write.psf_card(f'{output}/waterbox.psf', title=f'{XTLtype} Waterbox with box size {BoxSizeX}:{BoxSizeY}:{BoxSizeZ}', select='segid SOLV .or. segid IONS end')

print('Solvation completed')