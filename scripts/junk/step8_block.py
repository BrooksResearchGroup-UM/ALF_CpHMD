import os
import sys
import itertools
import pandas as pd
import argparse

# ----------------------------------------
# Command-line argument parsing
# ----------------------------------------
parser = argparse.ArgumentParser(
    description="Generate MSLD block and restraint files from patches.œdat"
)
parser.add_argument(
    '-i', '--input',
    required=True,
    help="Path to your input folder (must contain prep/patches.dat)"
)
parser.add_argument(
    '--restrain-type',
    choices=['SCAT', 'NOE'],
    default='NOE',
    help="Type of restraint to build (default: NOE)"
)
parser.add_argument(
    '-H', '--hydrogens',
    action='store_true',
    help="Include hydrogens in the restraint definitions"
)
parser.add_argument('-e', '--electrostatics', default='pmeex', 
                    choices=['pmeon', 'pmenn', 'pmeex', 'fshift', 'fswitch', 'pme_on', 'pme_nn', 'pme_ex'],
                    help="Electrostatics method (default: pme)")

args = parser.parse_args()

input_folder = args.input
restrain_type = args.restrain_type
restrain_hydrogens = args.hydrogens
temperature = 298.15

# Load patches.dat from the prep folder.
patches_file = os.path.join(input_folder, 'prep', 'patches.dat')
if not os.path.isfile(patches_file):
    print("Error: patches.dat not found in prep folder!")
    sys.exit(1)
patch_info = pd.read_csv(patches_file, sep=',')


# Optionally extract site information from the SELECT string (e.g. s1s2)
patch_info[['site', 'sub']] = patch_info['SELECT'].str.extract(r's(\d+)s(\d+)')

# Initialize BLOCK components
BLOCK = {
    'block': 1,
    'call': '',
    'excl': '',
    'ldin': '',
    'msld': 'msld 0',
    'ldbi': '',
    'ldbv': ''
}
BLOCK['excl'] = '!---------------------------------------------------------------\n! l-exclusions\n!---------------------------------------------------------------\n\n'
# Initial LDIN line (common to all)
BLOCK['ldin'] += 'LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0, 'NONE')



counter = 0
for site, group in patch_info.groupby('site', sort=False):
    resname = group['PATCH'].iloc[0][0:3]
    resname_lower = resname.lower()
    segid = group['SEGID'].iloc[0]
    resid = group['RESID'].iloc[0]
    # Load the variables file for this resname
    var_file_txt = os.path.join('variables', f'var-{resname_lower}.txt')
    var_file_inp = os.path.join('variables', f'var-{resname_lower}.inp')
    if os.path.isfile(var_file_txt):
        var_df = pd.read_csv(var_file_txt, sep=',', header=None,
                             names=['variable', 'value']).apply(lambda x: x.str.strip()).set_index('variable')
    elif os.path.isfile(var_file_inp):
        var_df = pd.DataFrame([line.split('=') for line in open(var_file_inp)
                               if line.strip().startswith('set')],
                              columns=['variable', 'value']).apply(lambda x: x.str.strip()).assign(
                                  variable=lambda x: x['variable'].str.split().str[-1]).set_index('variable')
    else:
        print('No variable file found for %s' % resname)
        sys.exit(1)
    
    BLOCK['ldin'] += (
        '!---------------------------------------------------------------\n'
        f'! Lambda Initialization for {segid} {resid} {resname} | SITE {site}\n'
        '!---------------------------------------------------------------\n'
    )
    BLOCK['ldbv'] += (
        '!---------------------------------------------------------------\n'
        f'! Biasing Potential for {segid} {resid} {resname} | SITE {site}\n'
        '!---------------------------------------------------------------\n'
    )
    BLOCK['call'] += (
        '!---------------------------------------------------------------\n'
        f'! CALL selection for {segid} {resid} {resname} | SITE {site}\n'
        '!---------------------------------------------------------------\n'
    )
    
    # Process each patch in this group.
    # (We assume each row in the group is one patch to be handled.)
    # The number of patches in the group will determine weights (1/len(group))
    counter += 1
    num_patches = len(group)
    # It is important to use the group order so that variable keys like lams1s1, lams1s2, etc. work.
    for idx, row in group.iterrows():
        patch = row['PATCH']
        utag = patch_info.at[idx, 'TAG']
        # For this example we use the "site" value from the patches file as the segment identifier.
        # Increase the block counter and add a CALL statement using the SELECT field.
        BLOCK['block'] += 1
        
        # BLOCK['call'] += "CALL {:>4} SELECT {} END\n".format(BLOCK['block'], row["SELECT"])
        BLOCK['call'] += "CALL {:>4} SELEct segid {:>4} .and. resid {:>4} .and. resname {:>4} end\n".format(BLOCK['block'], segid, resid, patch)
        BLOCK['msld'] += f' {counter}'
        
        # Determine the patch index within this group (starting at 1)
        group_indices = list(group.index)
        patch_index = group_indices.index(idx) + 1
        
        
        # Add an LDIN line. For the first patch, no additional tag is needed.
        if patch_index == 1:
            BLOCK['ldin'] += "LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n".format(
                BLOCK['block'], str(round(1/num_patches, 2)), 0.0, 12.0,
                var_df.loc[f'lams1s{patch_index}', 'value'], 5.0, 'NONE')
        else:
            # For subsequent patches, include the utag and a pka value (or similar) from the variable file.
            # (Adjust the key for pka if needed; here we assume the variable file contains a key named exactly as the patch index.)
            BLOCK['ldin'] += "LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n".format(
                BLOCK['block'], str(round(1/num_patches, 2)), 0.0, 12.0,
                var_df.loc[f'lams1s{patch_index}', 'value'], 5.0, f'{utag}' )
    
    BLOCK['msld'] += ' -\n'
    # Generate exclusions and bias potential lines (the following loops use combinations of block indices in this group)
    # Note: The exact formatting for keys such as 'cs1s...', 'ss1s...', and 'xs1s...' must match what is in your variable files.
    inner_group = list(itertools.combinations(range(BLOCK['block'] - num_patches + 1, BLOCK['block'] + 1), 2))
    for sub_group in inner_group:
        BLOCK['excl'] += 'adexcl ' + (' '.join(str(x) for x in sub_group)) + '\n'
        # Quadratic bias
        key = 'cs1s%ss1s%s' % (str(sub_group[0] - (BLOCK['block'] - num_patches)), str(sub_group[1] - (BLOCK['block'] - num_patches)))
        BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(
            str(BLOCK['ldbv'].count('LDBV') + 1),
            str(sub_group[0]),
            str(sub_group[1]),
            6,
            0.0,
            var_df.loc[key, 'value'],
            0)
    BLOCK['ldbv'] += '\n'
    inner_group = list(itertools.permutations(range(BLOCK['block'] - num_patches + 1, BLOCK['block'] + 1), 2))
    for sub_group in inner_group:
        # End-Point Potential
        key = 'ss1s%ss1s%s' % (str(sub_group[0] - (BLOCK['block'] - num_patches)), str(sub_group[1] - (BLOCK['block'] - num_patches)))
        BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(
            str(BLOCK['ldbv'].count('LDBV') + 1),
            str(sub_group[0]),
            str(sub_group[1]),
            8,
            0.017,
            var_df.loc[key, 'value'],
            0)
    BLOCK['ldbv'] += '\n'
    for sub_group in inner_group:
        # Skew Potential
        key = 'xs1s%ss1s%s' % (str(sub_group[0] - (BLOCK['block'] - num_patches)), str(sub_group[1] - (BLOCK['block'] - num_patches)))
        BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(
            str(BLOCK['ldbv'].count('LDBV') + 1),
            str(sub_group[0]),
            str(sub_group[1]),
            10,
            -5.56,
            var_df.loc[key, 'value'],
            0)
    BLOCK['ldbv'] += '\n'

# Set LDBI based on the total number of LDBV lines
BLOCK['ldbi'] = 'LDBI ' + str(BLOCK['ldbv'].count('LDBV')) + '\n'

# Assemble the full block string
BLOCK_str = '!---------------------------------------------------------------\n'
BLOCK_str += '! Set up l-dynamics by setting BLOCK parameters\n'
BLOCK_str += '!---------------------------------------------------------------\n\n'
BLOCK_str += 'BLOCK ' + str(BLOCK['block']) + ' NREP @nrep \n'
BLOCK_str += BLOCK['call'] + '\n' + BLOCK['excl'] + '\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '!QLDM turns on lambda-dynamics option\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'qldm theta\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '!LANGEVIN turns on the langevin heatbath\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'lang temp ' + str(temperature) + '\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '!Setup initial pH from PHMD\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'phmd pH 7\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '!Soft Core Potential\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'soft on\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! lambda-dynamics energy constrains (from ALF)\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += BLOCK['ldin'] + '\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! All bond/angle/dihe terms treated at full str (no scaling),\n'
BLOCK_str += '! prevent unphysical results\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'rmla bond thet impr\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! Selects MSLD, the numbers assign each block to the specified site on the core\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += BLOCK['msld'] + 'fnex 5.5\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! Constructs the interaction matrix and assigns lambda & theta values for each block\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += 'msma\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! PME electrostatics\n'
BLOCK_str += '!------------------------------------------\n\n'
if args.electrostatics == 'pmeon' or args.electrostatics == 'pme_on':
    BLOCK_str += 'pmel on\n\n'
elif  args.electrostatics == 'pmenn' or args.electrostatics == 'pme_nn':
    BLOCK_str += 'pmel nn\n\n'
elif  args.electrostatics == 'pmeex' or args.electrostatics == 'pme_ex':
    BLOCK_str += 'pmel ex\n\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! Enables bias potential on lambda variables\n'
BLOCK_str += '! INDEX, I,J(Bias between I & J)), CLASS, REF, CFORCE, NPOWER, Identity flag\n'
BLOCK_str += '! CLASS: Functional Form of bias, REF: Cut off for physical lambda states\n'
BLOCK_str += '! NPOWER: Power of functional form, CFORCE: kbias on Fvar, residue specific value\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += BLOCK['ldbi']
BLOCK_str += BLOCK['ldbv']
BLOCK_str += 'END'

# Write the block command to the output file
with open(os.path.join(input_folder, 'prep', 'block.str'), 'w') as f:
    f.write(BLOCK_str)

# Build restraint commands using SCAT logic.
restrains = ''
if restrain_type == 'SCAT':
    restrains += 'BLOCK\n'
    restrains += 'scat on \n'
    restrains += f'scat k {temperature}\n'
    # For each patch group, build restraints using the ATOMS field if available.
    for site in patch_info['site'].unique():
            atoms = patch_info.loc[patch_info['site'] == site]['ATOMS']
            # get all unique atoms in the site
            atoms = set([atom for atom in atoms.str.split().sum()])
            h_atoms = [atom for atom in atoms if atom.startswith('H')]
            atoms = [atom for atom in atoms if atom.startswith('H') == False]
            for atom in atoms:
                restrains += f'cats SELE type {atom} .and. ({" .or. ".join(map(str, patch_info.loc[patch_info["site"] == site]["SELECT"]))}) END\n'
            if restrain_hydrogens:
                for atom in h_atoms:
                    restrains += f'cats SELE type {atom} .and. ({" .or. ".join(map(str, patch_info.loc[patch_info["site"] == site]["SELECT"]))}) END\n'
                
    restrains += 'END\n'
elif restrain_type == 'NOE':
    restrains += '! Small minimization in case of atoms at same position\n'
    # all unique patches
    restrains += 'cons fix sele .not. (resn %%%%) .or. resn TIP3 end \n'
    restrains += 'mini sd nstep 2 step 0.005 \n'
    restrains += '! NOE restrains\n'
    restrains += 'NOE\n'
    index = 1
    # Group by site (from patches.dat) and process each site
    for site, group in patch_info.groupby('site', sort=False):
        segid = group['SEGID'].iloc[0]
        resid = group['RESID'].iloc[0]
        resname = group['PATCH'].iloc[0][0:3]  # Assuming first 3 chars of PATCH is resname
        patches = group['PATCH'].tolist()
        
        # Get atom names from the ATOMS column for this site
        atoms = group['ATOMS'].str.split().explode().dropna().unique()
        
        # Filter atoms based on restrain_hydrogens
        if not restrain_hydrogens:
            atoms = [atom for atom in atoms if not atom.startswith('H')]
        
        # Count occurrences of each atom name to find repeats
        atom_counts = group['ATOMS'].str.split().explode().value_counts()
        repeats = atom_counts[atom_counts > 1].index.tolist()
        
        if repeats:  # Only proceed if there are repeated atom names
            restrains += f'!---------------------------------------------------------------\n! Restrains for {segid} {resname} {resid}, SITE {site}, GROUP {index}\n!---------------------------------------------------------------\n'
            index += 1
            for repeat_atom in repeats:
                # Get all patches containing this atom
                atom_patches = group[group['ATOMS'].str.contains(repeat_atom, na=False)]
                if not restrain_hydrogens and repeat_atom.startswith('H'):
                    atom_patches = atom_patches.iloc[0:0]
                if len(atom_patches) > 1:  # Ensure more than one occurrence for restraint
                    for i1, i2 in itertools.combinations(atom_patches.index, 2):
                        patch1 = atom_patches.loc[i1, 'PATCH']
                        patch2 = atom_patches.loc[i2, 'PATCH']
                        restrains += (
                            f'assign sele segid {segid} .and. resid {resid} .and. resn {patch1} .and. type {repeat_atom} end '
                            f'sele segid {segid} .and. resid {resid} .and. resn {patch2} .and. type {repeat_atom} end -\n'
                            'kmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0\n'
                        )
    restrains += 'END\n'
    restrains += 'cons fix sele none end \n'

with open(os.path.join(input_folder, 'prep', 'restrains.str'), 'w') as f:
    f.write(restrains)