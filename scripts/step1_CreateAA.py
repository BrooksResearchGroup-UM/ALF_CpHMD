import os
import sys
import pycharmm
import pycharmm.read as read
import pycharmm.lingo as lingo
import pycharmm.generate as gen
import pycharmm.write as write
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.psf as psf

# Specify the template
seq_template = 'ALA ALA X ALA ALA'


lingo.charmm_script('prnlev 0')
read.rtf('../toppar/top_all36_prot.rtf')
read.prm('../toppar/par_all36m_prot.prm', flex = True)
lingo.charmm_script('stream ../toppar/toppar_water_ions.str')
lingo.charmm_script('prnlev 5')


amino_acids = ['HSP', 'LYS', 'ARG', 'ASP', 'GLU', 'TYR', 'SER', 'CYS']

if not os.path.isdir('pdb'): 
    os.system('mkdir pdb')

print('Amino acids to create: {}'.format(amino_acids))

for amino_acid in amino_acids:
    seq = seq_template.replace('X', amino_acid)
    if os.path.isfile('pdb/{}.pdb'.format(amino_acid.lower())):
        print('File pdb/{}.pdb already exists, skipping'.format(amino_acid.lower()))
        continue
    read.sequence_string(seq)
    gen.new_segment(seg_name = 'PROA', first_patch='ACE', last_patch='CT3', setup_ic = True)
    ic.prm_fill(replace_all=False)
    ic.seed(res1=1, atom1='CAY', res2=1, atom2='CY', res3=1, atom3='N')
    ic.build()
    coor.orient(by_rms=False,by_mass=False,by_noro=False)
    coor.show()

    write.coor_card('pdb/{}.crd'.format(amino_acid.lower()))
    write.coor_pdb('pdb/{}.pdb'.format(amino_acid.lower()))
    write.psf_card('pdb/{}.psf'.format(amino_acid.lower()))

    #  Delete atoms for the next run
    psf.delete_atoms(pycharmm.SelectAtoms().all_atoms())