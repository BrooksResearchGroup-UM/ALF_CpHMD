#! /usr/bin/env python


def GetFreeEnergy5(alf_info,ms,msprof, cutb=2.0, cutc=8.0, cutx=2.0, cuts=1.0, cutc2=1.0, cutx2=0.5, cuts2=0.5):
  import sys, os
  import numpy as np

  kT=0.001987*alf_info['temp']
  krest=1

  Emid=np.arange(1.0/800,1,1.0/400)
  Emid2=np.arange(1.0/40,1,1.0/20)

  nsubs=alf_info['nsubs']
  nblocks=alf_info['nblocks']

  b_prev=np.loadtxt('b_prev.dat')
  c_prev=np.loadtxt('c_prev.dat')
  x_prev=np.loadtxt('x_prev.dat')
  s_prev=np.loadtxt('s_prev.dat')

  b=np.zeros((1,nblocks))
  c=np.zeros((nblocks,nblocks))
  x=np.zeros((nblocks,nblocks))
  s=np.zeros((nblocks,nblocks))

  nparm=0
  for isite in range(0,len(nsubs)):
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2;
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        nparm+=n1+5*n2
      elif ms==1:
        nparm+=5*n3
      elif ms==2:
        nparm+=n3

  cutlist=np.zeros((nparm,))
  reglist=np.zeros((nparm,))
  n0=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2;
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        cutlist[n0:n0+n1]=cutb
        n0+=n1
        cutlist[n0:n0+n2]=cutc
        n0+=n2
        cutlist[n0:n0+2*n2]=cutx
        n0+=2*n2
        cutlist[n0:n0+2*n2]=cuts
        n0+=2*n2
      elif ms==1:
        cutlist[n0:n0+n3]=cutc2
        n0+=n3

        ind=n0
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            reglist[ind]=-x_prev[iblock+i,jblock+j]
            ind+=1
            reglist[ind]=-x_prev[jblock+j,iblock+i]
            ind+=1
        cutlist[n0:n0+2*n3]=cutx2
        n0+=2*n3

        ind=n0
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            reglist[ind]=-s_prev[iblock+i,jblock+j]
            ind+=1
            reglist[ind]=-s_prev[jblock+j,iblock+i]
            ind+=1
        cutlist[n0:n0+2*n3]=cuts2
        n0+=2*n3
      elif ms==2:
        cutlist[n0:n0+n3]=cutc
        n0+=n3
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  if not os.path.exists('multisite/C.dat'):
    print('Error, %s/multisite/C.dat does not exist, RunWham.py probably failed, check %s/output and %s/error for clues' % (os.getcwd(),os.getcwd(),os.getcwd()))
  C=np.loadtxt('multisite/C.dat')
  for i in range(0,n0):
    C[i,i]+=krest*cutlist[i]**-2
  for i in range(0,np.shape(C)[0]):
    if C[i,i]==0:
      C[i,i]=1;
  V=np.loadtxt('multisite/V.dat')
  for i in range(0,n0):
    # Add a harmonic restraint to the total value of the x and s cross terms
    V[i]+=(krest*cutlist[i]**-2)*reglist[i]

  # coeff=C^-1*V;
  coeff=np.linalg.solve(C,V)

  coeff_orig=coeff

  scaling=1.5/np.max(np.abs(coeff[0:n0]/cutlist))
  if scaling>1:
    scaling=1
  coeff*=scaling

  print("scaling is:")
  print(scaling)

  ind=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite==jsite:
        for i in range(0,nsubs[isite]):
          b[0,iblock+i]=coeff[ind]
          ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(i+1,nsubs[isite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[isite]):
            if i != j:
              x[iblock+i,jblock+j]=coeff[ind]
              ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[isite]):
            if i != j:
              s[iblock+i,jblock+j]=coeff[ind]
              ind+=1
      elif ms==1:
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            x[iblock+i,jblock+j]=coeff[ind]
            ind+=1
            x[jblock+j,iblock+i]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            s[iblock+i,jblock+j]=coeff[ind]
            ind+=1
            s[jblock+j,iblock+i]=coeff[ind]
            ind+=1
      elif ms==2:
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite!=jsite:
        for i in range(0,nsubs[isite]):
          b[0,iblock+i]+=c[iblock+i,jblock]
          c[iblock+i,jblock:jblock+nsubs[jsite]]-=c[iblock+i,jblock]
        for j in range(0,nsubs[jsite]):
          b[0,jblock+j]+=c[iblock,jblock+j]
          c[iblock:iblock+nsubs[isite],jblock+j]-=c[iblock,jblock+j]
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  iblock=0
  for isite in range(0,len(nsubs)):
    b[0,iblock:iblock+nsubs[isite]]-=b[0,iblock]
    iblock+=nsubs[isite]


  np.savetxt('b.dat',b,fmt=' %7.2f')
  np.savetxt('c.dat',c,fmt=' %7.2f')
  np.savetxt('x.dat',x,fmt=' %7.2f')
  np.savetxt('s.dat',s,fmt=' %7.2f')
# import sys, os
# import numpy as np
# from scipy import stats
# from scipy.signal import savgol_filter
# from scipy.optimize import minimize
# from sklearn.linear_model import HuberRegressor
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# import pymc as pm

# def GetFreeEnergy5(alf_info, ms, msprof, cutb=2.0, cutc=8.0, cutx=2.0, cuts=1.0, cutc2=1.0, cutx2=0.5, cuts2=0.5, verbose=False):
#     def vprint(*args, **kwargs):
#         if verbose:
#             print(*args, **kwargs)

#     vprint("Starting GetFreeEnergy5 calculation")

#     kT = 0.001987 * alf_info['temp']
#     krest = 1

#     Emid = np.arange(1.0/800, 1, 1.0/400)
#     Emid2 = np.arange(1.0/40, 1, 1.0/20)

#     nsubs = alf_info['nsubs']
#     nblocks = alf_info['nblocks']

#     vprint(f"Temperature: {alf_info['temp']}, kT: {kT}")
#     vprint(f"Number of subsystems: {len(nsubs)}, Total blocks: {nblocks}")

#     b_prev = np.loadtxt('b_prev.dat')
#     c_prev = np.loadtxt('c_prev.dat')
#     x_prev = np.loadtxt('x_prev.dat')
#     s_prev = np.loadtxt('s_prev.dat')

#     vprint("Loaded previous data files")

#     b = np.zeros((1, nblocks))
#     c = np.zeros((nblocks, nblocks))
#     x = np.zeros((nblocks, nblocks))
#     s = np.zeros((nblocks, nblocks))

#     nparm = 0
#     for isite in range(0, len(nsubs)):
#         n1 = nsubs[isite]
#         n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
#         for jsite in range(isite, len(nsubs)):
#             n3 = nsubs[isite] * nsubs[jsite]
#             if isite == jsite:
#                 nparm += n1 + 5 * n2
#             elif ms == 1:
#                 nparm += 5 * n3
#             elif ms == 2:
#                 nparm += n3

#     vprint(f"Total number of parameters: {nparm}")

#     cutlist = np.zeros((nparm,))
#     reglist = np.zeros((nparm,))
#     n0 = 0
#     iblock = 0
#     for isite in range(0, len(nsubs)):
#         jblock = iblock
#         n1 = nsubs[isite]
#         n2 = nsubs[isite] * (nsubs[isite] - 1) // 2
#         for jsite in range(isite, len(nsubs)):
#             n3 = nsubs[isite] * nsubs[jsite]
#             if isite == jsite:
#                 cutlist[n0:n0+n1] = cutb
#                 n0 += n1
#                 cutlist[n0:n0+n2] = cutc
#                 n0 += n2
#                 cutlist[n0:n0+2*n2] = cutx
#                 n0 += 2*n2
#                 cutlist[n0:n0+2*n2] = cuts
#                 n0 += 2*n2
#             elif ms == 1:
#                 cutlist[n0:n0+n3] = cutc2
#                 n0 += n3

#                 ind = n0
#                 for i in range(0, nsubs[isite]):
#                     for j in range(0, nsubs[jsite]):
#                         reglist[ind] = -x_prev[iblock+i, jblock+j]
#                         ind += 1
#                         reglist[ind] = -x_prev[jblock+j, iblock+i]
#                         ind += 1
#                 cutlist[n0:n0+2*n3] = cutx2
#                 n0 += 2*n3

#                 ind = n0
#                 for i in range(0, nsubs[isite]):
#                     for j in range(0, nsubs[jsite]):
#                         reglist[ind] = -s_prev[iblock+i, jblock+j]
#                         ind += 1
#                         reglist[ind] = -s_prev[jblock+j, iblock+i]
#                         ind += 1
#                 cutlist[n0:n0+2*n3] = cuts2
#                 n0 += 2*n3
#             elif ms == 2:
#                 cutlist[n0:n0+n3] = cutc
#                 n0 += n3
#             jblock += nsubs[jsite]
#         iblock += nsubs[isite]

#     vprint("Initialized cutlist and reglist")

#     if not os.path.exists('multisite/C.dat'):
#         print('Error, %s/multisite/C.dat does not exist, RunWham.py probably failed, check %s/output and %s/error for clues' % (os.getcwd(),os.getcwd(),os.getcwd()))
#     C = np.loadtxt('multisite/C.dat')
#     V = np.loadtxt('multisite/V.dat')

#     vprint("Loaded C and V matrices")
÷
    # np.savetxt('b.dat',b,fmt=' %7.2f')
    # np.savetxt('c.dat',c,fmt=' %7.2f')
    # np.savetxt('x.dat',x,fmt=' %7.2f')
    # np.savetxt('s.dat',s,fmt=' %7.2f')
