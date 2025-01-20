# #! /usr/bin/env python

# def GetEnergy(alf_info,Fi,Ff,skipE=1):

#   import sys, os
#   import numpy as np

#   # Fi=int(sys.argv[1])
#   # Ff=int(sys.argv[2])
#   NF=Ff-Fi+1;

#   # if len(sys.argv)>3:
#   #   skipE=int(sys.argv[3])
#   # else:
#   #   skipE=1

#   alphabet='abcdefghijklmnopqrstuvwxyz'

#   if os.path.isdir('../run'+str(Ff)):
#     production=False
#     ndupl=1
#   else:
#     production=True
#     ndupl=0
#     for i in range(0,len(alphabet)):
#       if os.path.isdir('../run'+str(Ff)+alphabet[i]):
#         ndupl+=1
#     if ndupl==0:
#       print("Error, not flattening or production")
#       quit()

#   nblocks=alf_info['nblocks']
#   nsubs=alf_info['nsubs']
#   nreps=alf_info['nreps']
#   ncentral=alf_info['ncentral']

#   b_shift=np.loadtxt('../nbshift/b_shift.dat')
#   c_shift=np.loadtxt('../nbshift/c_shift.dat')
#   x_shift=np.loadtxt('../nbshift/x_shift.dat')
#   s_shift=np.loadtxt('../nbshift/s_shift.dat')

#   Lambda=[]
#   b=[]
#   c=[]
#   x=[]
#   s=[]
#   for i in range(0,NF):
#     DIR='../analysis'+str(Fi+i)
#     for j in range(0,ndupl):
#       for k in range(0,nreps):
#         if production:
#           if os.path.isfile(DIR+'/data/Lambda.'+str(j)+'.'+str(k)+'.dat'):
#             Lambda.append(np.loadtxt(DIR+'/data/Lambda.'+str(j)+'.'+str(k)+'.dat')[(skipE-1)::skipE,:])
#         else:
#           if os.path.isfile(DIR+'/data/Lambda.'+str(j)+'.'+str(k)+'.dat'):
#             Lambda.append(np.loadtxt(DIR+'/data/Lambda.'+str(j)+'.'+str(k)+'.dat'))
#         b.append(np.loadtxt(DIR+'/b_prev.dat')+b_shift*(k-ncentral))
#         c.append(np.loadtxt(DIR+'/c_prev.dat')+c_shift*(k-ncentral))
#         x.append(np.loadtxt(DIR+'/x_prev.dat')+x_shift*(k-ncentral))
#         s.append(np.loadtxt(DIR+'/s_prev.dat')+s_shift*(k-ncentral))

#   if not os.path.isdir('Lambda'):
#     os.mkdir('Lambda')
#   if not os.path.isdir('Energy'):
#     os.mkdir('Energy')

#   E=[]
#   for i in range(0,NF*ndupl*nreps):
#     E.append([])
#     for j in range(0,NF*ndupl*nreps):
#       bi=b[i]
#       ci=c[i]
#       xi=x[i]
#       si=s[i]
#       Lj=Lambda[j]
#       E[-1].append(np.reshape(np.dot(Lj,-bi),(-1,1)))
#       E[-1][-1]+=np.sum(np.dot(Lj,-ci)*Lj,axis=1,keepdims=True)
#       E[-1][-1]+=np.sum(np.dot(1-np.exp(-5.56*Lj),-xi)*Lj,axis=1,keepdims=True)
#       E[-1][-1]+=np.sum(np.dot(Lj/(Lj+0.017),-si)*Lj,axis=1,keepdims=True)

#   for i in range(0,NF*ndupl*nreps):
#     Ei=E[nreps*(NF*ndupl-1)+ncentral][i]
#     for j in range(0,NF*ndupl*nreps):
#       Ei=np.concatenate((Ei,E[j][i]),axis=1)
#     np.savetxt('Energy/ESim'+str(i+1)+'.dat',Ei,fmt='%12.5f')

#   for i in range(0,NF*ndupl*nreps):
#     Li=Lambda[i]
#     np.savetxt('Lambda/Lambda'+str(i+1)+'.dat',Li,fmt='%10.6f')

#! /usr/bin/env python
def GetEnergy(alf_info, Fi, Ff, skipE=1):
    import sys, os
    import numpy as np

    NF = Ff - Fi + 1

    nblocks = alf_info['nblocks']
    nsubs = alf_info['nsubs']
    nreps = alf_info['nreps']
    ncentral = alf_info['ncentral']

    try:
        b_shift = np.loadtxt('../nbshift/b_shift.dat')
        c_shift = np.loadtxt('../nbshift/c_shift.dat')
        x_shift = np.loadtxt('../nbshift/x_shift.dat')
        s_shift = np.loadtxt('../nbshift/s_shift.dat')
    except FileNotFoundError as e:
        print(f"Error loading shift files: {e}")
        return

    Lambda = []
    b = []
    c = []
    x = []
    s = []

    for i in range(NF):
        analysis_dir = f'../analysis{Fi+i}'
        data_dir = os.path.join(analysis_dir, 'data')
        
        if not os.path.isdir(data_dir):
            print(f"Warning: Directory {data_dir} not found")
            continue

        # print(f"Checking directory: {data_dir}")

        lambda_files = [f for f in os.listdir(data_dir) if f.startswith('Lambda.') and f.endswith('.dat')]
        lambda_files.sort()

        for lambda_file in lambda_files:
            file_path = os.path.join(data_dir, lambda_file)
            print(f"Loading file: {file_path}")
            try:
                Lambda.append(np.loadtxt(file_path)[(skipE-1)::skipE,:])
                
                # Extract j and k from the filename (assuming format Lambda.j.k.dat)
                j, k = map(int, lambda_file.split('.')[1:3])
                
                b.append(np.loadtxt(os.path.join(analysis_dir, 'b_prev.dat')) + b_shift * (k - ncentral))
                c.append(np.loadtxt(os.path.join(analysis_dir, 'c_prev.dat')) + c_shift * (k - ncentral))
                x.append(np.loadtxt(os.path.join(analysis_dir, 'x_prev.dat')) + x_shift * (k - ncentral))
                s.append(np.loadtxt(os.path.join(analysis_dir, 's_prev.dat')) + s_shift * (k - ncentral))
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not os.path.isdir('Lambda'):
        os.mkdir('Lambda')
    if not os.path.isdir('Energy'):
        os.mkdir('Energy')

    total_simulations = len(Lambda)
    # print(f"Total simulations: {total_simulations}")

    if total_simulations == 0:
        print("Error: No Lambda files found.")
        return

    E = [[] for _ in range(total_simulations)]

    try:
        for i in range(total_simulations):
            for j in range(total_simulations):
                bi, ci, xi, si = b[i], c[i], x[i], s[i]
                Lj = Lambda[j]
                E[i].append(np.reshape(np.dot(Lj,-bi),(-1,1)))
                E[i][-1] += np.sum(np.dot(Lj,-ci)*Lj,axis=1,keepdims=True)
                E[i][-1] += np.sum(np.dot(1-np.exp(-5.56*Lj),-xi)*Lj,axis=1,keepdims=True)
                E[i][-1] += np.sum(np.dot(Lj/(Lj+0.017),-si)*Lj,axis=1,keepdims=True)
    except Exception as e:
        print(f"Error during energy calculation: {e}")
        return

    for i in range(total_simulations):
        Ei = E[total_simulations-1][i]
        for j in range(total_simulations):
            Ei = np.concatenate((Ei,E[j][i]),axis=1)
        np.savetxt(f'Energy/ESim{i+1}.dat', Ei, fmt='%12.5f')

    for i in range(total_simulations):
        Li = Lambda[i]
        np.savetxt(f'Lambda/Lambda{i+1}.dat', Li, fmt='%10.6f')

    # print("Energy calculation completed successfully.")