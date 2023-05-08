import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reading the file and extracting data
file = 'AllSets.xlsx'
DF = pd.read_excel(file, sheet_name='Sheet1')  # Data is of type 'Dataframe'
num_samples = DF.shape[0]
b = DF['b'].tolist()    # The strings are defined in the excel sheet
AR = DF['AR'].tolist()
TR = DF['TR'].tolist()
Vf = DF['Vf'].tolist()
r1 = DF['r1'].tolist()
Rp = DF['Rp'].tolist()
Rvi = DF['Rvi'].tolist()
Rhi = DF['Rhi'].tolist()
CompAvgStr = DF['CompAvgStress'].tolist()
CompStrength = DF['CompStrength'].tolist()

# Classification into good and bad designs based on the stress ratios
Good_Design = [[b[i], AR[i], TR[i], Vf[i], r1[i], Rp[i], Rvi[i], Rhi[i], CompAvgStr[i], CompStrength[i]] for i in range(num_samples) if max(Rp[i], Rvi[i], Rhi[i])==Rvi[i]]
Bad_Design = [[b[i], AR[i], TR[i], Vf[i], r1[i], Rp[i], Rvi[i], Rhi[i], CompAvgStr[i], CompStrength[i]] for i in range(num_samples) if max(Rp[i], Rvi[i], Rhi[i])!=Rvi[i]]

b_g, AR_g, TR_g, Vf_g, r1_g, Rp_g, Rvi_g, Rhi_g, CompAvgStr_g, CompStrength_g = zip(*Good_Design)
b_b, AR_b, TR_b, Vf_b, r1_b, Rp_b, Rvi_b, Rhi_b, CompAvgStr_b, CompStrength_b = zip(*Bad_Design)

# Creating Dataframes for Good and Bad designs for Regression analysis
nd_var = {'b':b, 'AR':AR, 'TR':TR, 'Vf':Vf, 'r1':r1}  # No distinction
nd_out = {'Strength':CompStrength}

gd_var = {'b':b_g, 'AR':AR_g, 'TR':TR_g, 'Vf':Vf_g, 'r1':r1_g}    # Good Design
gd_out = {'Strength':CompStrength_g}

bd_var = {'b':b_b, 'AR':AR_b, 'TR':TR_b, 'Vf':Vf_b, 'r1':r1_b}    # Bad Design
bd_out = {'Strength':CompStrength_b}

ND_var_DF = pd.DataFrame(nd_var)
ND_out_DF = pd.DataFrame(nd_out)

GD_var_DF = pd.DataFrame(gd_var)
GD_out_DF = pd.DataFrame(gd_out)

BD_var_DF = pd.DataFrame(bd_var)
BD_out_DF = pd.DataFrame(bd_out)

#**************************************************************************************************
# Plots
#**************************************************************************************************

# FIGURE-1
params = [('b', 'AR'), ('b', 'TR'), ('b', 'Vf'), ('b', 'r1'),
          ('AR', 'TR'), ('AR', 'Vf'), ('AR', 'r1'),
          ('TR', 'Vf'), ('TR', 'r1'),
          ('Vf', 'r1')]


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 40))
fig.subplots_adjust(hspace=0.24, wspace=0.3)

minval = ND_out_DF['Strength'].min()
maxval = ND_out_DF['Strength'].max()
cmap = plt.cm.get_cmap('jet')

for i, param in enumerate(params):
    ax = axes[i%2, i//2]
    ax.scatter(ND_var_DF[param[0]], ND_var_DF[param[1]], c=ND_out_DF['Strength']/maxval, cmap=cmap, marker='.')
    ax.set_xlabel('{}'.format(param[0]))
    ax.set_ylabel('{}'.format(param[1]))

sm = plt.cm.ScalarMappable(cmap=cmap)
sm.set_array(ND_out_DF['Strength']/maxval)
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', aspect=50)
cbar.ax.set_xlabel('Normalized Composite Strength')
plt.show()

# FIGURE-2
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 40))
fig.subplots_adjust(hspace=0.24, wspace=0.8)

for i, param in enumerate(params):
    ax = axes[i%2, i//2]
    ax.scatter(GD_var_DF[param[0]], GD_var_DF[param[1]], color='blue', marker='.')
    ax.scatter(BD_var_DF[param[0]], BD_var_DF[param[1]], color='red', marker='.')
    ax.set_xlabel('{}'.format(param[0]))
    ax.set_ylabel('{}'.format(param[1]))
    handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper center')
plt.show()

# FIGURE-3
Design = []
for i in range(len(b)):
    if max(Rvi[i], Rp[i], Rhi[i])==Rvi[i]:
        Design.append('Good Design')
    else:
        Design.append('Bad Design')

DF['DesignClass'] = Design


design_parameters = ['b', 'AR', 'TR', 'Vf', 'r1']
sns.set(style="ticks")
fig, axs = plt.subplots(ncols=5, figsize=(15, 5))
fig.subplots_adjust(hspace=0.5, wspace=0.4)
for i, parameter in enumerate(design_parameters):
    sns.boxplot(x='DesignClass', y=parameter, data=DF, ax=axs[i])
plt.show()

plt.subplot2grid((2,6),(0,0),colspan=2)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
p1 = sns.boxplot(x='DesignClass', y=b, data=DF)
p1.set(xlabel=None)
p1.set(ylabel='b')

plt.subplot2grid((2,6),(0,2),colspan=2)
p1 = sns.boxplot(x='DesignClass', y=AR, data=DF)
p1.set(xlabel=None)
p1.set(ylabel='AR')

plt.subplot2grid((2,6),(0,4),colspan=2)
p1 = sns.boxplot(x='DesignClass', y=TR, data=DF)
p1.set(xlabel=None)
p1.set(ylabel='TR')

plt.subplot2grid((2,6),(1,1),colspan=2)
p1 = sns.boxplot(x='DesignClass', y=Vf, data=DF)
p1.set(xlabel=None)
p1.set(ylabel='Vf')

plt.subplot2grid((2,6),(1,3),colspan=2)
p1 = sns.boxplot(x='DesignClass', y=r1, data=DF)
p1.set(xlabel=None)
p1.set(ylabel='r1')
plt.show()

# FIGURE-4
Failure = []
plt_counter = 0
vi_counter = 0
hi_counter = 0
for i in range(len(b)):
    if max(Rp[i], Rvi[i], Rhi[i])==Rp[i]:
        Failure.append('Platelet Failure')
        plt_counter += 1
    elif max(Rp[i], Rvi[i], Rhi[i])==Rvi[i]:
        Failure.append('VI Failure')
        vi_counter += 1
    else:
        Failure.append('HI Failure')
        hi_counter += 1

DF['FailureMode']:Failure
total_count = plt_counter + vi_counter + hi_counter
percentages = np.array([100*plt_counter/total_count, 100*vi_counter/total_count, 100*hi_counter/total_count])
labels = ['Platelet Failure: '+str(100*plt_counter/total_count)+'%', 'VI Failure: '+str(100*vi_counter/total_count)+'%', 'HI Failure: '+str(100*hi_counter/total_count)+'%']
c = ['red', 'green', 'blue']
explode = [0, 0, 0.2]
plt.pie(percentages, labels=labels, colors=c, shadow=False, explode=explode, textprops={'fontsize':18})
plt.show()
