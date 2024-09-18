import itertools
import os
import re
import sys
import time

import pandas as pd



# Process the input arguments
import sys

input_dir = sys.argv[1]


restrain_type = 'SCAT'
restrain_hydrogens = False


# Set pandas data frame for the PDB file
pdb_df = pd.DataFrame()
colspecs = [(0, 6), (6, 11), (12, 16), (17, 21), (22, 27),
            (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (70, 76)]  # define the columns to read
names = ['ATOM', 'index', 'name', 'resname', 'resid',
        'x', 'y', 'z', 'occupancy', 'charge', 'segment']  # column names

# Read the PDB
pdb_df = pd.read_fwf(input_dir+'/'+'prep/'+'system.pdb', colspecs=colspecs, names=names)
# clean text unrelated to atoms
pdb_df = pdb_df.loc[pdb_df['ATOM'] == 'ATOM']  
pdb_df.drop(['ATOM', 'occupancy', 'charge', 'x', 'y', 'z'], axis=1, inplace=True)
# pdb_df['resid'] = pdb_df['resid'].astype(int)  # convert residue number to int
pdb_df['index'] = pdb_df['index'].astype(int)  # convert index to int
pdb_df.set_index('index', inplace=True)  # set index according to PDB index

#Prepare BLOCK function
BLOCK = {'block': 1, 'call': '', 'excl': '', 'ldin': '',
            'msld': 'msld 0', 'ldbi': '', 'ldbv': ''} # create empty dictionary for BLOCK
BLOCK['excl'] = '!---------------------------------------------------------------\n! l-exclusions\n!---------------------------------------------------------------\n\n'

pka = {'ASP': 3.67,
       'GLU': 4.25,
       'HSP': [7.08, 6.69],
       'LYS': 10.40,
       'TYR': 9.84,
       'ARG': 12.5}

pka = {'ASH1': 3.37, 'ASH2': 3.37,
       'GLH1': 3.95, 'GLH2': 3.95,
       'HSPD': 7.08, 'HSPE': 6.69, 
       'LYSU': 10.4, 'TYRU': 9.84, 
       'ARU1': 12.8,'ARU2': 12.8,
       'CYSD': 8.55, 'SERD': 12.8} 

temperature = 298.15
counter = 0

# Find all resnames which are different but have the same resid
# e.g. resid 8 has resname ARG and ARGO later in the PDB and provide a list of them
grouped = pdb_df.groupby(['resid', 'segment'])['resname'].apply(list).reset_index()
resname_resid_dict = {}
for _, row in grouped.iterrows():
    resid, segment, resname_list = row['resid'], row['segment'], row['resname']
    if len(set(resname_list)) > 1:
        unique_resnames = list(dict.fromkeys(resname_list))
        resname = ' '.join(unique_resnames)
        key = f"{resname} {segment}"
        if key not in resname_resid_dict:
            resname_resid_dict[key] = []
        resname_resid_dict[key].append(resid)


BLOCK['ldin'] += 'LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n'.format(1, 1, 0.0, 12.0, 0.0, 5.0, 'NONE')
for group in resname_resid_dict.keys():
    parts = group.split()
    resname = parts[0]
    patches = parts[1:-1]  # Everything except the first and last elements
    segment = parts[-1]
    resids = resname_resid_dict[group]
    print(f'Processing {segment} {resname} {resids}')
    if os.path.isfile('variables/var-'+ resname.lower() +'.txt'):
        # print('found .txt')
        # var_df = pd.DataFrame(columns=['variable', 'value'])
        var_df = pd.read_csv('variables/var-'+ resname.lower() +'.txt', sep=',', header=None, names=['variable', 'value']).apply(lambda x: x.str.strip()).set_index('variable')
    elif os.path.isfile('variables/var-'+ resname.lower() +'.inp'):
        # print('found .inp')
        var_df = pd.DataFrame([line.split('=') for line in open('variables/var-'+ resname.lower() +'.inp') if line.strip().startswith('set')], columns=['variable', 'value']).apply(lambda x: x.str.strip()).assign(variable=lambda x: x['variable'].str.split().str[-1]).set_index('variable')
    else:
        print('No variable file found for %s' % resname)
        sys.exit()  
    
    if resname == 'ASP' or resname == 'GLU':
        utag = 'UNEG'
    else: 
        utag = 'UPOS'
    
    for resid in resids:
        counter += 1
        BLOCK['call'] += f'!---------------------------------------------------------------\n! CALL selections for {segment} {resname} {resid}, GROUP {counter}\n!---------------------------------------------------------------\n'
        BLOCK['ldin'] += f'!---------------------------------------------------------------\n! Initialize Fixed Bias dG for {segment} {resname} {resid}, GROUP {counter}\n!---------------------------------------------------------------\n\n'
        BLOCK['excl'] += f'!---------------------------------------------------------------\n! l-exclusions for {segment} {resname} {resid}, GROUP {counter}\n!---------------------------------------------------------------\n'
        BLOCK['ldbv'] += f'!---------------------------------------------------------------\n! Bias Potentials for {segment} {resname} {resid}, GROUP {counter}\n!---------------------------------------------------------------\n'
       
        for patch in patches:
            BLOCK['block'] += 1
            # BLOCK['call'] += f"CALL {BLOCK['block']} SELEct resid {resid} .and. resname '{patch}' end\n"
            BLOCK['call'] += "CALL {:>4} SELEct segid {:>4} .and. resid {:>4} .and. resname {:>4} end\n".format(BLOCK['block'], segment, resid, patch)
            BLOCK['msld'] += f' {counter}'
            patch_index = patches.index(patch) + 1
            if patch_index == 1:
                BLOCK['ldin'] += "LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n".format(BLOCK['block'], str(round(1/len(patches),2)), 0.0, 12.0, str(var_df.loc[f'lams1s{patch_index}','value']), 5.0, 'NONE')
            else:
                BLOCK['ldin'] += "LDIN {:<6} {:<6} {:<6} {:<7} {:>8} {:>7} {:>12}\n".format(BLOCK['block'], str(round(1/len(patches),2)), 0.0, 12.0, str(var_df.loc[f'lams1s{patch_index}','value']), 5.0, f'{utag} {pka[patch]}')
        BLOCK['ldbv'] 
        
        inner_group = list(itertools.combinations(range(BLOCK['block']-len(patches)+1, BLOCK['block']+1), 2))
        for sub_group in inner_group:
            BLOCK['excl'] += 'adexcl ' + (' '.join(str(x) for x in sub_group)) + '\n'
            # Quadratic bias
            BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(str(BLOCK['ldbv'].count('LDBV')+1), str(sub_group[0]), str(sub_group[1]), 6, 0.0,var_df.loc['cs1s%ss1s%s' % (str(sub_group[0]-BLOCK['block']+len(patches)), str(sub_group[1]-BLOCK['block']+len(patches))), 'value'], 0)
        BLOCK['ldbv'] += '\n'
        for sub_group in inner_group:
            #End-Point Potential
            BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(str(BLOCK['ldbv'].count('LDBV')+1), str(sub_group[0]), str(sub_group[1]), 8, 0.017,var_df.loc['ss1s%ss1s%s' % (str(sub_group[0]-BLOCK['block']+len(patches)), str(sub_group[1]-BLOCK['block']+len(patches))), 'value'], 0)
            BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(str(BLOCK['ldbv'].count('LDBV')+1), str(sub_group[1]), str(sub_group[0]), 8, 0.017,var_df.loc['ss1s%ss1s%s' % (str(sub_group[1]-BLOCK['block']+len(patches)), str(sub_group[0]-BLOCK['block']+len(patches))), 'value'], 0)
        BLOCK['ldbv'] += '\n'
        for sub_group in inner_group:
            # Skew Potential
            BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(str(BLOCK['ldbv'].count('LDBV')+1), str(sub_group[0]), str(sub_group[1]), 10, -5.56,var_df.loc['xs1s%ss1s%s' % (str(sub_group[0]-BLOCK['block']+len(patches)), str(sub_group[1]-BLOCK['block']+len(patches))), 'value'], 0)
            BLOCK['ldbv'] += 'LDBV {:<3} {:>4} {:>4} {:>4} {:>8} {:>10} {:>5}\n'.format(str(BLOCK['ldbv'].count('LDBV')+1), str(sub_group[1]), str(sub_group[0]), 10, -5.56,var_df.loc['xs1s%ss1s%s' % (str(sub_group[1]-BLOCK['block']+len(patches)), str(sub_group[0]-BLOCK['block']+len(patches))), 'value'], 0)

        # BLOCK['call'] += '\n'
        # BLOCK['ldin'] += '\n'
        # BLOCK['excl'] += '\n'
        BLOCK['msld'] += ' -\n'
    
BLOCK['ldbi'] = 'LDBI '+ str(BLOCK['ldbv'].count('LDBV')) + '\n'  
        

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
BLOCK_str += 'pmel ex ! ADDED\n'
BLOCK_str += '!------------------------------------------\n'
BLOCK_str += '! Enables bias potential on lambda variables\n'
BLOCK_str += '! INDEX, I,J(Bias between I & J)), CLASS, REF, CFORCE, NPOWER, Identity flag\n'
BLOCK_str += '! CLASS: Functional Form of bias, REF: Cut off for physical lambda states\n'
BLOCK_str += '! NPOWER: Power of functional form, CFORCE: kbias on Fvar, residue specific value\n'
BLOCK_str += '!------------------------------------------\n\n'
BLOCK_str += BLOCK['ldbi']
BLOCK_str += BLOCK ['ldbv'] 
BLOCK_str += 'END'

file = open(f'{input_dir}/prep/block.str', 'w')
file.write(BLOCK_str)
file.close()


restrains = ''
if restrain_type == 'NOE':
    restrains += 'cons fix resname TIP3.or. resname POT .or. resname CLA end \n'
    restrains += 'mini sd nstep 100 nprint 100 step 0.005 \n'
elif restrain_type == 'SCAT':
    restrains += 'BLOCK\n'
    restrains += 'scat on \n'
    restrains += f'scat k {temperature}\n'
index = 1
for group in resname_resid_dict.keys():
    resname = group.split()[0]
    patches = group.split()[1:]
    resids = resname_resid_dict[group]
    for resid in resids:
        if restrain_type == 'NOE':
            restrains += 'NOE\n'
        # find segments which starts with resid in pdb_df
        sub_df = pdb_df[pdb_df['resid'] == resid]
        sub_df = sub_df[sub_df['resname'] != resname]
        restrains += f'!---------------------------------------------------------------\n! Restrains for {segment} {resname} {resid}, GROUP {index}\n!---------------------------------------------------------------\n'
        index += 1
        # find names which repeat more than once
        repeats = sub_df['name'].value_counts()[sub_df['name'].value_counts() > 1].index.tolist()
        if restrain_hydrogens is False:
            repeats = list(filter(lambda x: not x.startswith('H'), repeats))
        for repeat_atom in repeats:
            select = sub_df[sub_df['name'] == repeat_atom]
            if restrain_type == 'NOE':
                for i1, i2 in itertools.combinations(select.index, 2):
                    restrains += 'assign sele {:>4} .and. resid {:>4} .and. resn {:>4} .and. type {:>4} end sele {:>4} .and. resid {:>4} .and. resn {:>4} .and. type {:>4} end -\nkmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0\n'.format(i1, select.loc[i1, "resid"], select.loc[i1, "resname"], select.loc[i1, "name"], i2, select.loc[i2, "resid"], select.loc[i2, "resname"], select.loc[i2, "name"])
                    # restrains += f'assign sele segid {select.loc[i1, "segment"]} .and. resid {select.loc[i1, "resid"]} .and. resn {select.loc[i1, "resname"]} .and. type {select.loc[i1, "name"]} end sele segid {select.loc[i1, "segment"]} .and. resid {select.loc[i2, "resid"]} .and. resn {select.loc[i2, "resname"]} .and. type {select.loc[i2, "name"]} end -\nkmin 100.0 rmin 0.0 kmax 100.0 rmax 0.0 fmax 2.0 rswitch 99999 sexp 1.0\n'
            elif restrain_type == 'SCAT':
                restrains += f'cats sele ('
                for segment in select['segment'].unique():
                    restrains += 'segid {:>4} '.format(segment)
                    # restrains += f'segid {segment} '
                    #if segment is not the last one
                    if segment != select['segment'].unique()[-1]:
                        restrains += '.or. '
                    else: restrains += '.and. ' 
                restrains += 'resid {:>4} .and. '.format(resid)
                # restrains+= f'resid {resid} .and. '    
                
                # for patch in patches:
                #     restrains += f'resname {patch}'
                #     if patch != patches[-1]:
                #         restrains += ' .or. '
                #     else:
                #         restrains += ' .and. '
                        
                # restrains += f'type {repeat_atom}) end\n'
                restrains += 'type {:>4}) end\n'.format(repeat_atom)
        if restrain_type == 'NOE':
            restrains += 'END\n'

if restrain_type == 'SCAT':
    restrains += 'END'

if restrain_type == 'NOE':
    restrains += 'cons fix sele none end \n'
    
file = open(f'{input_dir}/prep/restrains.str', 'w')
file.write(restrains)
file.close()