# -*- coding: utf-8 -*-
###############################################################################
import os
import numpy as np
import pandas as pd
import seaborn as sns
from glob import iglob
import matplotlib as mpl
from scipy.io import mmread
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
import csv

os.chdir('C:\Users\Analyzer\Desktop\Bcell')

#Prepare enviroment
mpl.rcParams['axes.titlesize'] = 19
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 13
mpl.rcParams['legend.markerscale'] = 4
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['figure.dpi'] = 100

sns.set_style('ticks')
sns.set_palette('Set1')

###############################################################################
#QUALITY CONTROL ANALYSIS
###############################################################################  

###############################################################################
#lOAD QC MATRIX ------ NEEDS TO BE DONE
#sample_info = pd.read_csv('mSp_scATAC-seq/qc_all.csv', index_col=0)
#sample_info.head(2)

###############################################################################
#Load count table from MTX file
count = mmread('mSp_scATAC_count_matrix_over_aggregate.mtx')
idxs = [i.strip() for i in open('mSp_scATAC_count_matrix_over_aggregate.rownames')]
cols = [i.strip() for i in open('mSp_scATAC_count_matrix_over_aggregate.colnames')]

#Change cols names
#def remove_cruft(s):
#    return s[-7:]
#cols = [remove_cruft(col) for col in cols]

###############################################################################
#Generating the matrix
sc_count = pd.DataFrame(data=count.toarray(),
                        index=idxs,
                        columns=cols)

#bigdata=sc_count.join(sc_count_2)

plate_qc = pd.read_table('QC_matrix.txt', sep='\t');
plate_qc.index=plate_qc['Sample']

#sequencing_depth
sequencing_depth = plate_qc['fastaq']
#mapping_rate
mapping_rate = ((plate_qc['mapped']*100)/plate_qc['fastaq'])
#dup_level
dup_level = plate_qc['dupp']
#uniq_frags
uniq_frags = plate_qc['rm_d_qc']

#REINDEX
dup_level.index = plate_qc.index
sequencing_depth.index = plate_qc.index
mapping_rate.index = plate_qc.index
uniq_frags.index =  plate_qc.index

#Dataframe with some qc's obtained
#REINDEX table of QC
plate_qc['sequencing_depth'] = sequencing_depth
plate_qc['mapping_rate'] = mapping_rate
plate_qc['dup_level'] = dup_level
plate_qc['uniq_frags'] = uniq_frags

#Change cols names
new_list=plate_qc['Sample'].tolist()
def remove_cruft(s):
    return s[:-7]
new_cols = [remove_cruft(col) for col in new_list]
sc_count = sc_count[new_cols]

#frip
frip = sc_count.sum()
plate_qc['frip'] = frip.values

#Redu frip for %
plate_qc['frip'] = (plate_qc['frip'])/plate_qc['uniq_frags']

plate_qc.head()
plate_qc.median()

#Empty cells
empty_cells = ['esplenocitosplateA_5ulTN5_singlecell__10uM_human_H12_nocells_S87_R1_001', 'esplenocitosplateA_5ulTN5_singlecell__10uM_human_A12_nocells_S84_R1_001']

#PLOT 1 FOR QC
g = sns.FacetGrid(plate_qc.drop(labels=empty_cells), size=6)
g.map(plt.scatter, 'sequencing_depth', 'mapping_rate', s=5)
g.ax.set_xscale('log')
g.ax.set_xlim(10, 5e7)

g.ax.scatter(plate_qc.loc[empty_cells, 'sequencing_depth'],
             plate_qc.loc[empty_cells, 'mapping_rate'],
             label='Empty wells', c='#f0f0f0',
             s=20, edgecolors='k')

g.ax.legend(bbox_to_anchor=[1.5, .5])
g.ax.axhline(y=90, xmin=0, xmax=1, ls='dashed', c='k', lw=1.)
g.ax.axvline(x=1e4, ymin=0, ymax=1, ls='dashed', c='k', lw=1.)

g.ax.set_xlabel('Sequencing depth')
g.ax.set_ylabel('Overall alignment rate%')
plt.savefig('qc_sequencing_depth_vs_mapping_rate.pdf',
            bbox_inches='tight', transparent=True)
            
#PLOT 2 FOR QC
plate_qc['dupp'] = plate_qc['dupp'].str.replace(',','.').astype(np.float64)
plate_qc['dup_level'] = plate_qc['dup_level'].str.replace(',','.').astype(np.float64)

g = sns.FacetGrid(plate_qc.drop(labels=empty_cells), size=6)
g.map(plt.scatter, 'dup_level', 'sequencing_depth', s=5)
g.ax.set_yscale('log')
g.ax.set_ylim(10, 5e7)

g.ax.scatter(plate_qc.loc[empty_cells, 'dup_level'],
             plate_qc.loc[empty_cells, 'sequencing_depth'],
             label='Empty wells', c='#f0f0f0',
             s=20, edgecolors='k')

g.ax.legend(bbox_to_anchor=[1.5, .5])

g.ax.set_xlabel('Duplication level')
g.ax.set_ylabel('Sequencing depth')

plt.savefig('qc_sequencing_depth_vs_duplication_level.pdf',
            bbox_inches='tight', transparent=True)   
            
#PLOT 3 FOR QC
g = sns.FacetGrid(plate_qc.drop(labels=empty_cells), size=6)
g.map(plt.scatter, 'sequencing_depth', 'uniq_frags', s=5)
g.ax.set_xscale('log')
g.ax.set_yscale('log')
g.ax.set_xlim(10, 5e7)
g.ax.set_ylim(10, 5e6)

g.ax.scatter(plate_qc.loc[empty_cells, 'sequencing_depth'],
             plate_qc.loc[empty_cells, 'uniq_frags'],
             label='Empty wells', c='#f0f0f0',
             s=20, edgecolors='k')

g.ax.legend(bbox_to_anchor=[1.5, .5])

g.ax.set_xlabel('Sequencing depth')
g.ax.set_ylabel('# of unique fragments')

plt.savefig('qc_sequencing_depth_vs_uniq_frags.pdf',
            bbox_inches='tight', transparent=True)
            
#PLOT 4 FOR QC            
g = sns.FacetGrid(plate_qc.drop(labels=empty_cells), size=6, palette='Set1')
g.map(plt.scatter, 'uniq_frags', 'frip', s=5)
g.ax.set_xscale('log')
g.ax.set_xlim(10, 1e6)
g.ax.set_ylim(0, 1)

g.ax.scatter(plate_qc.loc[empty_cells, 'uniq_frags'],
             plate_qc.loc[empty_cells, 'frip'],
             label='Empty wells', c='#f0f0f0',
             s=20, edgecolors='k')

g.ax.legend(bbox_to_anchor=[1.5, .5])

g.ax.set_xlabel('# of unique fragments')
g.ax.set_ylabel('Fraction of reads in peak (FRiP)')

plt.savefig('uniq_frags_vs_frip.pdf',
            bbox_inches='tight', transparent=True)

#Downsampling analysis to look at # of unique fragments
downsampling = pd.read_table('downsampling.txt', sep='\t');

downsampling = downsampling.loc[3:]
#downsampling = downsampling.loc[[0]]
#downsampling = downsampling.loc[[1]]
#downsampling = downsampling.loc[[2]]

downsampling.index=downsampling['cell']
downsampling = downsampling.drop(['cell'], axis=1)

plt.figure(figsize=(9, 6))
plt.plot(downsampling.columns.astype(float), downsampling.median(), 'r.-', ms=15)
plt.xticks(downsampling.columns.astype(float), downsampling.columns.values)
plt.ylim(0,)

plt.xlabel('Fraction of original depth')
plt.ylabel('Median # unique fragments mapped to nuclear genome')

plt.savefig('subsampling_unique_fragments_median_50KBulk.pdf',
            bbox_inches='tight', transparent=True)

#Violin plots for previous efficiency data
violin_data = pd.melt(downsampling, value_vars=['0.2', '0.4', '0.6', '0.8', '1'])

fig, ax = plt.subplots(figsize=(12, 6.5))

sns.violinplot(x='variable', y='value', data=violin_data, ax=ax)
ax.set_xlabel('Fraction of original depth')
ax.set_ylabel('# unique fragments mapped to nuclear genome')
#ax.set_ylim(-15000, 100000)

plt.savefig('subsampling_unique_fragments_violin.pdf',
            bbox_inches='tight', transparent=True)
   
###############################################################################
#EXPLORATORY ANALYSIS
###############################################################################         

###############################################################################
#Elements to be dropped ------ NEEDS TO BE DONE
#to_drop = pd.concat([pd.read_csv('mSp_scATAC-seq/qc_bad_cells.csv', index_col=0),
#                     pd.read_csv('mSp_scATAC-seq/qc_possible_doublets.csv', index_col=0)])

#sc_count.drop(labels=to_drop.index, axis=1, inplace=True)
###############################################################################

###############################################################################
# binarise the data
bin_sc_count = sc_count.where(sc_count < 1, 1)
bin_sc_count.shape

###############################################################################
#Filter cells and peaks
#Counts to use as filter
np.percentile(bin_sc_count.sum(), 5)
# at least 247.5 peaks detected in the cell
bin_sc_count = bin_sc_count.loc[:,bin_sc_count.sum() > 442.4] # 3 bad cells and 2 nocells removed = 5
# at least two cells have the peak
bin_sc_count = bin_sc_count.loc[bin_sc_count.sum(1) >= 2,] #341 removed

#sample_info = sample_info.loc[bin_sc_count.columns]
#sc_count = sc_count.loc[bin_sc_count.index, bin_sc_count.columns]

bin_sc_count.shape

###############################################################################
#Perform Latent Semantic Indexing Analysis
# get TF-IDF matrix
tfidf = TfidfTransformer(norm='l2', sublinear_tf=True)
normed_count = tfidf.fit_transform(bin_sc_count.T)

# perform SVD on the sparse matrix
lsi = TruncatedSVD(n_components=50, random_state=42)
lsi_r = lsi.fit_transform(normed_count)

lsi.explained_variance_ratio_

plate_qc.index = new_cols
plate_qc = plate_qc.loc[bin_sc_count.columns]
for i in range(3):
    plate_qc['LSI Dimension {}'.format(i+1)] = lsi_r[:, i]
plate_qc = plate_qc.loc[sorted(plate_qc.index)]

###############################################################################
#Look the first dimension and the sequencing depth
g = sns.FacetGrid(plate_qc, size=6)
g.map(plt.scatter, 'LSI Dimension 1', 'sequencing_depth', s=5)
g.ax.legend(bbox_to_anchor=[1.5, .5])
g.ax.set_yscale('log')
g.ax.set_ylim(1e4, 2e7)
g.ax.set_ylabel('Sequencing depth')

plt.savefig('LSI_1st_dimension_vs_sequencing_depth.pdf',
            bbox_inches='tight', transparent=True)

###############################################################################
#Look the first dimension and the uniq frags
g = sns.FacetGrid(plate_qc, size=6)
g.map(plt.scatter, 'LSI Dimension 1', 'uniq_frags', s=5)
g.ax.legend(bbox_to_anchor=[1.5, .5])
g.ax.set_yscale('log')
g.ax.set_ylim(1e2, 5e5)
g.ax.set_ylabel('Number of uniqe fragments')

plt.savefig('LSI_1st_dimension_vs_uniq_frags.pdf',
            bbox_inches='tight', transparent=True)

###############################################################################
#Look the first dimension and the uniq frags
g = sns.FacetGrid(plate_qc, size=6)
g.map(plt.scatter, 'LSI Dimension 2', 'LSI Dimension 3', s=5)
g.ax.legend(bbox_to_anchor=[1.5, .5])

plt.savefig('LSI_2nd_dimension_vs_3rd_dimension.pdf',
            bbox_inches='tight', transparent=True)

###############################################################################
##Check peaks linked to some marker genes (by annotatePeaks.pl from the HOMER suite)
#homer = pd.read_table('cmp_to_immgen/homer_annotation_spleen_union_peaks_no_black_list.txt', index_col=0)
#marker_genes = ['Bcl11a', 'Bcl11b', 'Cd3e', 'Cd4',
#                'Cd8a', 'Cd19', 'Ms4a1', 'Ebf1',
#                'Tcf7', 'Gzma', 'Lrg1']
#
#marker_p2g = homer[homer['Gene Name'].isin(marker_genes)]['Gene Name']
#marker_count = pd.concat([sc_count, marker_p2g], axis=1, join='inner')
#marker_count.head(2)
#
#marker_sum = marker_count.groupby('Gene Name').sum()
#marker_sum
#
##PLot markers in TSNE projection
#for g in marker_genes:
#    plate_qc[g] = marker_sum.loc[g]
#
##some cells showed high counts around both T-cell and B-cell markers
##these are possible doublets
##if this is the case, they might have more reads/fragments than the others
#
#tcf7_high = marker_sum.columns[marker_sum.loc['Tcf7'] > 10]
#ebf1_high = marker_sum.columns[marker_sum.loc['Ebf1'] > 10]
#bcl11a_high = marker_sum.columns[marker_sum.loc['Bcl11a'] > 10]
#bcl11b_high = marker_sum.columns[marker_sum.loc['Bcl11b'] > 10]
#
#dbs1 = (set(tcf7_high) & set(ebf1_high))
#dbs2 = (set(bcl11a_high) & set(bcl11b_high))
#dbs = dbs1 | dbs2
#
#fig, ax = plt.subplots(figsize=(3,5))
#
#offset1 = np.random.normal(scale=0.1, size=plate_qc.drop(labels=dbs).shape[0])
#ax.scatter(1 + offset1, plate_qc.drop(labels=dbs).uniq_frags, s=8, c='k')
#
#offset2 = np.random.normal(scale=0.05, size=len(dbs))
#ax.scatter(2 + offset2, plate_qc.loc[dbs].uniq_frags, s=3, c='k', alpha=.5)
#ax.set_yscale('log')
#
#ax.set_xticks(range(1,3))
#ax.set_xticklabels(['The reset', 'Possible\ndoublets'])
#ax.set_xlim(0.5,2.5)

#len(dbs)
#plate_qc.loc[dbs].to_csv('mSp_scATAC-seq/qc_possible_doublets.csv')
#plate_qc.drop(labels=dbs, inplace=True)

###############################################################################
## get the size - migration time of the ladder from bioanalzyer trace
#ladder_info = pd.read_csv('bioanalyzer_results/Ladder.csv', encoding = 'ISO-8859-1')
#ladder_info.head(2)
#
## plot bioanlayzer traces of each library
#fig, ax = plt.subplots(figsize=(9,4.5))
#samples = iglob('bioanalyzer_results/Rep*.csv')
#
#for s in samples:
#    ax.cla()
#    sn = s.split('/')[-1][:-4]
#    df = pd.read_csv(s, skiprows=17)
#    df = df.iloc[:-1,:]
#    df = df.astype(float)
#    ax.plot(df.Time, df.Value, color='#d7301f', lw=2.)
#    ax.set_xticks(ladder_info['Aligned Migration Time [s]'])
#    ax.set_xticklabels(['35', '', '', '150', '', '300', '', '500', '', '',
#                        '1000', '', '', '', '       10380 [bp]'])
#    ax.set_xlim(35, 130)
#    ax.set_xlabel("Size")
#    ax.set_ylabel('[FU]')
#    plt.savefig('figures/{}_bioanlayzer.pdf'.format(sn), bbox_inches='tight', transparent=True)


###############################################################################
#Since we know that the first LSI dimension is related to sequencing depth
#We just ignore the first dimension since, and only pass the 2nd dimension and onwards for t-SNE
X_lsi = lsi_r[:, 1:]
tsne = TSNE(n_components=2,
            learning_rate=200,
            random_state=42,
            n_jobs=10).fit_transform(X_lsi)

plate_qc['t-SNE Dimension 1'] = tsne[:, 0]
plate_qc['t-SNE Dimension 2'] = tsne[:, 1]

#zeros=np.zeros(75)
#unos=np.ones(384)
#nuevo=np.concatenate((zeros, unos), axis=0)
#plate_qc['color'] = nuevo

#siguiente linea para dar colores , hue="color"
g = sns.FacetGrid(plate_qc, size=6, palette='Set1')
g.map(plt.scatter, 't-SNE Dimension 1', 't-SNE Dimension 2', s=2)

g.ax.spines['top'].set_visible(True)
g.ax.spines['right'].set_visible(True)
g.ax.tick_params(left=False, bottom=False)
g.ax.set_xticks([])
g.ax.set_yticks([])

g.ax.legend(bbox_to_anchor=[1.5, .5])
plt.savefig('tSNE_colored_by_batch.pdf',
             bbox_inches='tight', transparent=True)

###############################################################################
#We ignore the first LSI dimension by setting it to zero
#Then we invert it back to its original space
lsi_r[:, 0] = 0
lsi_r[:, 1] = 0
matrix = lsi.inverse_transform(lsi_r)
matrix.shape

#Perform spectral clustering on cells using the new matrix
cluster = SpectralClustering(n_clusters=12,
                             n_jobs=-1,
                             affinity='nearest_neighbors',
                             random_state=42, n_neighbors=15,
                             assign_labels='discretize',
                             n_init=50).fit(scale(matrix))

plate_qc['cluster_labels'] = cluster.labels_
plate_qc.cluster_labels.value_counts()

color12 = ['#a6cee3', '#333333', '#b2df8a', '#33a02c',
           '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00',
           '#cab2d6', '#6a3d9a', '#1f78b4', '#b15928']

g = sns.FacetGrid(plate_qc, hue='cluster_labels', size=6, palette=sns.color_palette(color12))
g.map(plt.scatter, 't-SNE Dimension 1', 't-SNE Dimension 2', s=2)

g.ax.spines['top'].set_visible(True)
g.ax.spines['right'].set_visible(True)
g.ax.tick_params(left=False, bottom=False)
g.ax.set_xticks([])
g.ax.set_yticks([])

###############################################################################




















