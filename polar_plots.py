import numpy as np
import os
import matplotlib.pyplot as plt

# configurations : VV i.e. same polarisation of incoming and outgoing light
# configurations : HV i.e. pi/2 angle between polarisations of incoming and outgoing light
material = 'IBAPbBr3'
config = 'VV'
E_pol_plane_axes = [0,1] # 0,1,2 for x, y, z
dir_propogation = '001' # To be used while saving file
offset_angle = np.pi *0.5 # To account for the arbitrary choice of 0 degrees in expt
raman_tensor_file = 'raman_tensor.npy'
phonon_energies_file = 'phonon_energies.npy'
raman_int_out_file=f'summed_intensity_{config}_{dir_propogation}_{material}.dat' # file for raman intensity summed over all angles
dump_int_summed_over_pol_angles=True # True : Store Raman intensity summed over angles
stretch_factor_phonon_freq = 1.0  # Phonon energies will be multiplied by this factor

# Choose energy window for phonons
start_e = 25 # in cm-1
end_e = 175
e_pad = 10 # To account for broadening from phonons outside the window
gamma = 2.5 # Broadening in cm-1

#####################################
# Defining Gaussian lineshape to use for broadening
def gauss(energy,strength,gamma,energy_grid):
    lineshape = strength * np.exp(-0.5*((energy_grid-energy)**2/gamma**2))
    return(lineshape)

##### Loading previously calculated Raman tensor and phonon energies
raman_tensor = np.load(raman_tensor_file,allow_pickle=True)
phonon_energies = np.load(phonon_energies_file,allow_pickle=True) #* 1.05

# Use the subset of phonons belonging to energy window
subset = np.argwhere(np.logical_and(phonon_energies > start_e - e_pad, phonon_energies < end_e + e_pad ))
subset = subset[:,0]

raman_tensor = raman_tensor[subset,...]
phonon_energies = phonon_energies[subset]

### Defining an angular grid
angle_grid = np.linspace(0,2*np.pi,1000)

e_field_in = np.zeros((len(angle_grid),3))  # Defining dimensionality of arrays for light pol
e_field_out = np.zeros((len(angle_grid),3))

e_field_in[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle)  # Defining incoming light pols
e_field_in[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle)

if config == 'VV':
    # VV configuration
    # Incoming and outgoing light have the same polarization
    e_field_out[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle) # VV configuration
    e_field_out[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle)
else:
    e_field_out[:,E_pol_plane_axes[0]] = np.sin(angle_grid+offset_angle + np.pi*0.5) # HV configuration
    e_field_out[:,E_pol_plane_axes[1]] = np.cos(angle_grid+offset_angle + np.pi*0.5)

# Matrix multiplication and squaring to get Raman intensity
raman_int = np.abs(np.einsum('tm,pmn,tn -> pt',e_field_in,raman_tensor,e_field_out))**2

# Create folder to dump polar plots
if os.path.exists('./polar_plots'):
    pass
else:
    os.system("mkdir polar_plots")

# Loop over phonon energies
for i in range(len(phonon_energies)):
    # Initiate a polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(3,3))
    ax.set_rticks([]) # Removing radial ticks
    # Normalizing Raman intensity for polar plots
    if np.max(raman_int[i,:]) > 0.001:
        raman_int_norm = raman_int[i,:] / np.max(raman_int[i,:])
    else:
        raman_int_norm = raman_int[i,:] # No normalization if intensity is negligible (due to diverging denominator)
    # Plotting
    plt.plot(angle_grid,raman_int_norm,linewidth=2)
    #ax.set_ylim(bottom=-0.01,top=1.1)
    # Assign A or B symmetry label to file based on if Raman tensor is diagonal or not
    #if np.sum(np.abs(np.diagonal(raman_tensor[i,...]))) > 0.0001:
    #    plt.savefig(f'polar_plots/polar_plot_{i}_A_{config}_{dir_propogation}.png',bbox_inches='tight')
    #else:
    #    plt.savefig(f'polar_plots/polar_plot_{i}_B_{config}_{dir_propogation}.png',bbox_inches='tight')
    plt.close()

# Dumping Raman intensity summed over all pol angles
if dump_int_summed_over_pol_angles==True:
    sum_intensity = np.sum(raman_int,axis=1) # Summed over all angles, new dim : phonon, x, y
    # Define energy grid
    energy_grid = np.linspace(start_e,end_e,1000) # in cm-1
    # Sum Gaussian lineshapes of Raman intensities of all phonon modes
    intensity=0.0
    for p, energy in enumerate(phonon_energies):
        intensity += gauss(energy,sum_intensity[p],gamma,energy_grid)
    # Plotting the spectra
    fig, ax = plt.subplots(figsize=(4,3))
    plt.plot(energy_grid,intensity,linestyle='solid',linewidth=2,color='darkblue')
    plt.bar(phonon_energies,sum_intensity,width=1,color='darkgrey',label='Mode\nintensity' )
    # In this part, we identify the intense modes (top 20)
    # We highlight them and assign symmetry labe in plot based on Raman tensor structure
    intense_indices = np.argsort(sum_intensity)
    intense_indices = intense_indices[::-1][:30]
    k_label = 0
    j_label = 0
    for j in intense_indices:
        if np.sum(np.abs(np.diagonal(raman_tensor[j,...]))) > 0.00001:
            k_label +=1
            #plt.text(phonon_energies[j],np.max(intensity)*1.075,'A',ha='center', va='center')
            if k_label == 1:
                plt.axvline(phonon_energies[j],color='green',zorder=-10,alpha=0.3,label='A modes')
            else:
                plt.axvline(phonon_energies[j],color='green',zorder=-10,alpha=0.3,linestyle='solid')
        else:
            j_label += 1
            #plt.text(phonon_energies[j],np.max(intensity)*1.125,'B',ha='center', va='center')
            if j_label == 1:
                plt.axvline(phonon_energies[j],color='red',zorder=-10,alpha=0.3,linestyle='dashed',label='B modes')
            else:
                plt.axvline(phonon_energies[j],color='red',zorder=-10,alpha=0.3,linestyle='dashed')
    plt.xlim(start_e,end_e)
    plt.xlabel(r'Raman shift (cm$^{-1}$)')
    plt.ylabel('Raman intensity (arb. u.)')
    plt.legend(loc=3,fontsize=8,handlelength=1.5,bbox_to_anchor=(0.825,0.65))
    plt.savefig(f'polar_plots/summed_intensity_{config}_{dir_propogation}_{material}.png',dpi=400,bbox_inches ='tight') # Saving plot
    np.savetxt(raman_int_out_file,np.c_[energy_grid,intensity]) # Saving intensity


#intense_indices = np.argsort(sum_intensity)
#intense_indices = intense_indices[::-1][:20]

#  for n in intense_indices:
#      if sum_intensity[n] > 1000:
#          if np.sum(np.abs(np.diagonal(raman_tensor[n,...]))) > 0.000001:
#              sym_label = "A"
#          elif np.abs(raman_tensor[n,0,1]) > 0.000001:
#              sym_label = "B1"
#          elif np.abs(raman_tensor[n,0,2]) > 0.000001:
#              sym_label = "B2"
#          elif np.abs(raman_tensor[n,1,2]) > 0.000001:
#              sym_label = "B3"
#          else:
#              sym_label = "N/A"
#          print(n,phonon_energies[n],sum_intensity[n],sym_label)
# for p,ene in enumerate(phonon_energies):
#     if sum_intensity[p] > 1000:
#         if np.sum(np.abs(np.diagonal(raman_tensor[p,...]))) > 0.000001:
#             sym_label = "A"
#         elif np.abs(raman_tensor[p,0,1]) > 0.000001:
#             sym_label = "B1"
#         elif np.abs(raman_tensor[p,0,2]) > 0.000001:
#             sym_label = "B2"
#         elif np.abs(raman_tensor[p,1,2]) > 0.000001:
#             sym_label = "B3"
#         else:
#             sym_label = "N/A"
#         print(p, phonon_energies[p],"   ",sum_intensity[p],"   ",sym_label)
#plt.bar(phonon_energies,sum_intensity,width=0.5,color='black')
#plt.plot(energy_grid,intensity_hv,linestyle='dashed')
