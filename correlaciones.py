import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import scipy.stats as stats

cloze1_online = pd.read_csv('1_online.csv')
cloze1_inlab = pd.read_csv('1_presencial.csv')
cloze2_toma1 = pd.read_csv('2_toma1.csv')
cloze2_toma2 = pd.read_csv('2_toma2.csv')
cloze3_full = pd.read_csv('3_full.csv')
cloze3_orac = pd.read_csv('3_orac.csv')

recat = pd.read_csv('recategorizoOraciones.csv')

show = False
size_letters = 'xx-large'


plt.rcParams['figure.figsize'] = [30 / 2.54, 30/ 2.54]

cloze1 = pd.merge(cloze1_inlab, cloze1_online, on=['id_orac_online', 'palNum'])
cloze1 = cloze1.loc[cloze1['tipo_orac_x'] != 3]  # Filtro para sacar tipo 3
cloze1.to_csv('1_TODO.csv', index=False)

cloze2 = pd.merge(cloze2_toma1, cloze2_toma2, left_on=['oracID', 'palNum'], right_on=['oracID_orig', 'palNum'])
cloze2.to_csv('2_TODO.csv', index=False)

cloze3 = pd.merge(cloze3_full, cloze3_orac, left_on=['id_text', 'palNum'], right_on=['id_text', 'palNumGobal'])
cloze3.to_csv('3_TODO.csv', index=False)

####Ruidos########

mu, sigma = 0, 0.001

noise1_x = np.random.normal(mu, sigma, [897])
noise1_y = np.random.normal(mu, sigma, [897])

noise2_x = np.random.normal(mu, sigma, [2260])
noise2_y = np.random.normal(mu, sigma, [2260])

noise3_x = np.random.normal(mu, sigma, [1323])
noise3_y = np.random.normal(mu, sigma, [1323])

cloze1['pred_noise_x'] = cloze1['pred_x'] + noise1_x
cloze1['pred_noise_y'] = cloze1['pred_y'] + noise1_y

cloze2['pred_noise_x'] = cloze2['pred_x'] + noise2_x
cloze2['pred_noise_y'] = cloze2['pred_y'] + noise2_y

cloze3['pred_noise_x'] = cloze3['pred_x'] + noise3_x
cloze3['pred_noise_y'] = cloze3['pred_y'] + noise3_y

############ Scatters Bruno ############

plt.scatter(cloze1.pred_x, cloze1.pred_y)
plt.plot([0, 1], [0, 1])
plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
plt.title('Proverbs 1')
plt.xlabel('In Lab')

plt.ylabel('Online')
plt.savefig('scatter_1.svg', transparent=True)
if show:
    plt.show()
plt.close()

plt.scatter(cloze2.pred_x, cloze2.pred_y)
plt.plot([0, 1], [0, 1])
plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
plt.title('Proverbs 2')
plt.xlabel('Toma 1')
plt.ylabel('Toma 2')
plt.savefig('scatter_2.svg', transparent=True)
if show:
    plt.show()
plt.close()

plt.scatter(cloze3.pred_x, cloze3.pred_y)
plt.plot([0, 1], [0, 1])
plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
plt.title('Stories')
plt.xlabel('Contextualizado')
plt.ylabel('Oraciones Aisladas')
plt.savefig('scatter_3.svg', transparent=True)
if show:
    plt.show()
plt.close()



####Scatters con hues ######



####SACAR LABELS


# Proverbs 1
plt.rcParams.update({'figure.figsize': (10, 10), 'figure.dpi': 100})

lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="tipo_orac_new",
                   palette="crest", data=cloze1)
#plt.title('Proverbs 1')
plt.xlabel('In-Lab', fontsize=size_letters)
plt.ylabel('Online', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('hue_1.svg', transparent=True)
if show:
    plt.show()
plt.close()

#lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="palNum", palette="crest", data=cloze1,legend='brief')
#plt.title('Proverbs 1')
#leg = lena._legend
#leg.set_bbox_to_anchor([0.4, 0.9])
#plt.xlabel('In Lab', fontsize=size_letters)
#plt.ylabel('Online', fontsize=size_letters)
#plt.xticks(fontsize=size_letters)
#plt.yticks(fontsize=size_letters)
#plt.tight_layout()
#plt.savefig('hue_11.pdf', transparent=True)
#if show:
#    plt.show()
#plt.close()

# Cloze2
lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="tipo_x", palette='crest', data=cloze2,)
#plt.title('Proverbs 2')

plt.xlabel('Cohort 1', fontsize=size_letters)
plt.ylabel('Cohort 2', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('hue_2.svg', transparent=True)
if show:
    plt.show()
plt.close()

#lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="palNum", palette='crest', data=cloze2,legend='brief')
#plt.title('Proverbs 2')
#leg = lena._legend
#leg.set_bbox_to_anchor([0.4, 0.9])
#plt.xlabel('Toma 1', fontsize=size_letters)
#plt.ylabel('Toma 2', fontsize=size_letters)
#plt.xticks(fontsize=size_letters)
#plt.yticks(fontsize=size_letters)
#plt.tight_layout()
#plt.savefig('hue_22.pdf', transparent=True)
#if show:
#    plt.show()
#plt.close()

# Cloze3


####CAMBIAR COLOR DE LOS CUENTOS QUE SEA TIPO ESCALA
lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="id_text", data=cloze3)
#plt.title('Stories')

plt.xlabel('Contextualized', fontsize=size_letters)
plt.ylabel('Isolated', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)
plt.savefig('hue_3.svg', transparent=True)
if show:
    plt.show()
plt.close()

#lena = sns.relplot(x="pred_noise_x", y="pred_noise_y", hue="palNum_y", palette="crest", data=cloze3,legend='brief')
#plt.title('Stories')
#leg = lena._legend
#leg.set_bbox_to_anchor([0.4, 0.9])
#plt.xlabel('Contextualizado', fontsize=size_letters)
#plt.ylabel('Oraciones Aisladas', fontsize=size_letters)
#plt.xticks(fontsize=size_letters)
#plt.yticks(fontsize=size_letters)
#plt.tight_layout()
#plt.savefig('hue_33.pdf', transparent=True)
#if show:
#    plt.show()
#plt.close()

############### 2D histogram matplotlib ###############

tmp = cloze1[["pred_x", "pred_y"]].dropna()

[h, x, y, _] = plt.hist2d(tmp.pred_x, tmp.pred_y,
                          cmap="Greens", bins=10,
                          norm=LogNorm())
plt.plot([0, 1], [0, 1], 'g')

#plt.title(label="Proverbs 1")
plt.xlabel(xlabel='In-Lab', fontsize=size_letters)
plt.ylabel(ylabel='Online', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)
plt.savefig('hist2d_1.svg', transparent=True)
if show:
    plt.show()
plt.close()

tmp = cloze2[["pred_x", "pred_y"]].dropna()
[h, x, y, _] = plt.hist2d(tmp.pred_x, tmp.pred_y,
                          cmap="Greens", bins=20,
                          norm=LogNorm())
plt.plot([0, 1], [0, 1], 'g')

#plt.title(label="Proverbs 2")
plt.xlabel(xlabel='Cohort 1', fontsize=size_letters)
plt.ylabel(ylabel='Cohort 2', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)
plt.savefig('hist2d_2.svg', transparent=True)
if show:
    plt.show()
plt.close()

tmp = cloze3[["pred_x", "pred_y"]].dropna()
[h, x, y, _] = plt.hist2d(tmp.pred_x, tmp.pred_y,
                          cmap="Greens", bins=20,
                          norm=LogNorm())
plt.plot([0, 1], [0, 1], 'g')

#plt.title(label="Stories")
plt.xlabel(xlabel='Contextualized', fontsize=size_letters)
plt.ylabel(ylabel='Isolated', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)
plt.savefig('hist2d_3.svg', transparent=True)
if show:
    plt.show()
plt.close()

######Scatters posicion/predictibilidad#########


#############SACAR LABELS PONER A MANO, CAMBIAR COLORES OJO DALTONICOS


# Cloze 1
groupped1 = (
    cloze1.groupby('palNum').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std']})).dropna()

groupped1['error_x'] = (groupped1['pred_x']['std']) / np.sqrt(groupped1['pred_x']['count'])
groupped1['error_y'] = (groupped1['pred_y']['std']) / np.sqrt(groupped1['pred_y']['count'])

plt.errorbar(groupped1.index + 0.1, groupped1['pred_x']['mean'], yerr=groupped1['error_x'],linewidth=3, label='In Lab')
plt.errorbar(groupped1.index, groupped1['pred_y']['mean'], yerr=groupped1['error_y'],linewidth=3, label='Online')

#plt.title(label="Proverbs 1")
plt.xlabel(xlabel='Word Position', fontsize=size_letters)
plt.ylabel(ylabel='Predictability', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('line_1.svg', transparent=True)
if show:
    plt.show()
plt.close()

# Cloze2

groupped2 = (
    cloze2.groupby('palNum').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std']})).dropna()

groupped2['error_x'] = (groupped2['pred_x']['std']) / np.sqrt(groupped2['pred_x']['count'])
groupped2['error_y'] = (groupped2['pred_y']['std']) / np.sqrt(groupped2['pred_y']['count'])

plt.errorbar(groupped2.index + 0.1, groupped2['pred_x']['mean'], yerr=groupped2['error_x'],linewidth=3, label='Toma 1')
plt.errorbar(groupped2.index, groupped2['pred_y']['mean'], yerr=groupped2['error_y'],linewidth=3, label='Toma 2')

#plt.title(label="Proverbs 2")
plt.xlabel(xlabel='Word Position', fontsize=size_letters)
plt.ylabel(ylabel='Predictability', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('line_2.svg', transparent=True)
if show:
    plt.show()
plt.close()

# Cloze3

groupped3 = (
    cloze3.groupby('palNum_y').agg({'pred_x': ['mean', 'count', 'std'], 'pred_y': ['mean', 'count', 'std']})).dropna()

groupped3['error_x'] = (groupped3['pred_x']['std']) / np.sqrt(groupped3['pred_x']['count'])
groupped3['error_y'] = (groupped3['pred_y']['std']) / np.sqrt(groupped3['pred_y']['count'])

plt.errorbar(groupped3.index + 0.1, groupped3['pred_x']['mean'], yerr=groupped3['error_x'],linewidth=3, label='Contextualizada')
plt.errorbar(groupped3.index, groupped3['pred_y']['mean'], yerr=groupped3['error_y'],linewidth=3, label='Oraciones Aisladas')

#plt.title(label="Stories")
plt.xlabel(xlabel='Word Position', fontsize=size_letters)
plt.ylabel(ylabel='Predictability', fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('line_3.svg', transparent=True)
if show:
    plt.show()
plt.close()

##### Histogramas #####

# Cloze1
c1 = sns.distplot(cloze1["pred_x"], color="navy", bins=10, norm_hist=True, label="In Lab", kde=False)
c1 = sns.distplot(cloze1["pred_y"], color="darkorange", bins=10, norm_hist=True, label="Online", kde=False)
c1.set_yscale('log')
#plt.title(label='Proverbs 1')
plt.xlabel(xlabel="predictibilidad", fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('hist_1.svg', transparent=True)
if show:
    plt.show()
plt.close()

# Cloze2
c2 = sns.distplot(cloze2["pred_x"], color="navy", bins=20, norm_hist=True, label="Toma 1", kde=False)
c2 = sns.distplot(cloze2["pred_y"], color="darkorange", bins=20, norm_hist=True, label="Toma 2", kde=False)
c2.set_yscale('log')
#plt.title(label='Proverbs 2')
plt.xlabel(xlabel="predictibilidad", fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('hist_2.svg', transparent=True)
if show:
    plt.show()
plt.close()

# Cloze3
c3 = sns.distplot(cloze3["pred_x"], color="navy", bins=20, norm_hist=True, label="Contextualizado", kde=False)
c3 = sns.distplot(cloze3["pred_y"], color="darkorange", bins=20, norm_hist=True, label="Oraciones Aisladas", kde=False)
c3.set_yscale('log')
#plt.title(label='Stories')
plt.xlabel(xlabel="predictibilidad", fontsize=size_letters)
plt.xticks(fontsize=size_letters)
plt.yticks(fontsize=size_letters)

plt.savefig('hist_3.svg', transparent=True)
if show:
    plt.show()
plt.close()


#Ttests

agrupado = pd.concat([groupped1, groupped2, groupped3], axis=1)



##Virtud online --> Online tiene una resolucion casi continua,por n.Definicion es 1/n

