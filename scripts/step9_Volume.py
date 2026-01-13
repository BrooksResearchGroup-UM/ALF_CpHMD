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
selected_patching = []

import argparse

def parser():
    # Create the parser
    global hmr, input_folder, structure_file, hmr_waters, selected_patching
    parser = argparse.ArgumentParser(description='Process additional parameters for the simulation.')

    # Define the expected command line arguments
    parser.add_argument('-hmr','--hmr', type=bool, default=hmr, help='Whether to use hydrogen mass repartitioning (True/False)',action=argparse.BooleanOptionalAction)
    parser.add_argument('-hmrw','--hmr_waters', type=bool, default=hmr_waters, help='Whether to use hydrogen mass repartitioning for water molecules (True/False)',action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--input_folder', type=str, default=input_folder, help='Name of the input folder')
    parser.add_argument('-f', '--file', type=str, default=structure_file, help='Name of the structure file')
    parser.add_argument('-s', '--selected_patching', type=str, nargs='+', default=selected_patching, help='Selected residues for patching')

    # Parse the arguments
    args = parser.parse_args()

    # Assign global variables
    hmr = args.hmr
    hmr_waters = args.hmr_waters
    input_folder = args.input_folder
    structure_file = args.file
    selected_patching = args.selected_patching
    selected_patching = [patch.upper() for patch in selected_patching]




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


check_charmm()
read_topology_files()
parser()
read.psf_card(f'{input_folder}/prep/{structure_file}.psf')
read.coor_card(f'{input_folder}/prep/{structure_file}.crd')

psf.delete_atoms(pycharmm.SelectAtoms.by_res_name('TIP3')._invert)
lingo.charmm_script('''
calc S1 = ( INT( ?XMAX - ?XMIN ) * INxT( ?YMAX - ?YMIN ) * INT( ?ZMAX - ?ZMIN ) * 1000000 ) ** 3
format (I12)
calc S2 = INT( @S1 )
format
SCALar WMAIn = RADIus
SCALar WMAIn STORE 1
SCALAr WMAIn ADD 1.6
SCALar WMAIn STORE 2
coor volu SPACE @S2 hole sele segi PROA end                 
                    
                    '''
)