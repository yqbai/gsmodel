import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
from collections import Counter
import torch

rename2 = pd.read_csv('anno.264clusters.renameV1212.txt', sep="\t")
rename2.set_index('anno2', inplace=True)

mat_marker = pd.read_csv('topct_model/mat_marker.csv', sep = ',')

mat_columnanno = pd.read_csv('data2024/column2annot_man_res_corrected.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
seqlen = 20

def sum_expr(sampletag, n_extend_plan = 50, min_d = 5, seqlen = 20, topn = 30, filter = ''):
    top_features = pd.read_csv('data2024/topct_model/top_features.inter30.csv')
    if filter != '':
        if filter == 'EX_L23':
            top_features = top_features[top_features['subtype'].isin(['L2','L2/3'])]
        if filter == 'EX_L4':
            top_features = top_features[top_features['subtype'].isin(['L4/5'])]
        if filter == 'EX_L6':
            top_features = top_features[top_features['subtype'].isin(['L6'])]
        if filter == 'IN':
            top_features = top_features[top_features['class'].isin(['GABA'])]
        if filter == 'NonNeuron':
            top_features = top_features[top_features['class'].isin(['NonNeuron'])]
    genelist = []
    for item in top_features.index:
        item2 = rename2.loc[rename2['FullName'] == item, 'Plot_Old']
        genes = mat_marker.loc[mat_marker['FullName'] == item, 'gene']
        genelist.extend(genes[:min(topn, len(genes))])
    genelist = list(set(genelist))
    print(len(genelist))
    print(sampletag)
    mat_cl = pd.read_csv(f'spa_matrix/cell-meta-matrix_1105/cell-meta-matrix.{sampletag}.tsv', sep="\t")
    mat_cl['layer'] = mat_cl['gene_area'].apply(lambda x: x.split('-')[2])
    mat_cl['region'] = mat_cl['gene_area'].apply(lambda x: x.split('-')[1])
    mat_cl['ry'] = -mat_cl['ry']
    mat_cl = mat_cl[mat_cl['cluster'].isin(rename2.index)]
    mat_cl['class'] = mat_cl['cluster'].map(lambda x: rename2.loc[x, 'FullName'].split(' ')[0])
    mat_cl['subclass'] = mat_cl['cluster'].map(lambda x: rename2.loc[x, 'FullName'].split(' ')[1].split('.')[0])
    mat_cl['celltype'] = mat_cl['cluster'].map(lambda x: rename2.loc[x, 'FullName'])
    mat_column = pd.read_csv(f'data2024/cell_column/{sampletag}_cell2column.csv')
    mat_cl = mat_cl.merge(mat_column, on='cell_id')
    print(mat_cl.shape)
    mat_cl.rename(columns={'column_id': 'column_id_ori'}, inplace=True)
    unique_columns = sorted(mat_cl['column_id_ori'].unique())
    column_map = {col: idx + 1 for idx, col in enumerate(unique_columns)}
    mat_cl['column_id'] = mat_cl['column_id_ori'].map(column_map)
    mat_cl['cell_id'] = mat_cl['cell_id'].apply(lambda x: str(int(float(x))) if pd.notnull(x) else None)
    sob = sc.read_h5ad(f'spa_matrix/sct_h5ad/spa_matrix.{sampletag}.h5ad')
    sob = sob[sob.obs.index.isin(mat_cl['cell_id'])]
    mat_cl = mat_cl[mat_cl['cell_id'].isin(sob.obs.index)]
    sob.obs = mat_cl.set_index('cell_id')
    mat_columnanno = pd.read_csv('data2024/column2annot_man_res_corrected.csv')
    mat_columnanno = mat_columnanno[mat_columnanno['section'] == sampletag].sort_values('column')
    if len(mat_columnanno) == 1:
        n_extend = n_extend_plan
    else:
        n_extend = min(
            [mat_columnanno.iloc[i]['column'] - mat_columnanno.iloc[i - 1]['column'] for i in range(1, len(mat_columnanno))]
        ) // 2
        n_extend = min(n_extend, n_extend_plan)
    print(n_extend)
    sob.obs['gs'] = np.nan
    sob.obs['area'] = np.nan
    features = []
    labels = []
    samplenames = []
    N = 0
    for _, row in mat_columnanno.iterrows():
        column = row['column']
        if column in mat_cl['column_id_ori'].values:
            column_mid = mat_cl.loc[mat_cl['column_id_ori'] == column, 'column_id'].iloc[0]
        else:
            d = abs(mat_cl['column_id_ori'] - column)
            if d.min() <= min_d:
                column_mid = mat_cl.loc[d.idxmin(), 'column_id']
            else:
                print(f"Error! min(d) is {d.min()}")
                continue
        start, end = column_mid - n_extend, column_mid + n_extend
        sob.obs.loc[(sob.obs['column_id'] >= start) & (sob.obs['column_id'] <= end), 'gs'] = row['g/s']
        sob.obs.loc[(sob.obs['column_id'] >= start) & (sob.obs['column_id'] <= end), 'area'] = row['area']
        subset_sob = sob[sob.obs['column_id'].between(start, end)]
        subset_sob = subset_sob[:, subset_sob.var_names.isin(genelist)]
        t = subset_sob.to_df()
        if t is None:
            continue
        t.columns = [col.replace('-', '_') for col in t.columns]
        t['column_id'] = subset_sob.obs['column_id'].values
        t_grouped = t.groupby('column_id').sum()
        t_grouped.index = [f"{sampletag}_{col}" for col in t_grouped.index]
        N += 1
        for i in range(t_grouped.shape[0] - seqlen):
            window = t_grouped.iloc[i:(i+seqlen),:]
            window_tensor = torch.tensor(window.values, dtype=torch.float32)
            features.append(window_tensor.unsqueeze(0))
            labels += [row['g/s']]
            samplenames += [t_grouped.index[i]]
    if features:
        features = torch.cat(features, dim=0)
    else:
        features = None
    print(N, features.shape, len(labels))
    return features, labels, samplenames

def prepare_input_expr(samplelist, seqlen = 20, topn = 30, filter = ''):
    Features = []
    Labels = []
    Samplenames = []
    for sampletag in samplelist:
        features, labels, samplenames = sum_expr(sampletag, n_extend_plan = 50, min_d = 5, seqlen = seqlen, topn = topn, filter = filter)
        if features is not None:
            Features.append(features)
            Labels += labels
            Samplenames += samplenames
    if Features:
        Features = torch.cat(Features, dim=0)  # Concatenate tensors
    else:
        Features = None
    return Features, Labels, Samplenames

def encode_onehot(labels):
    labels2 = []
    for item in labels:
        if item == '1':
            labels2 += ['Gyrus']
        else:
            labels2 += ['Sulcus']
    classes_dict = {'Sulcus': np.array([1., 0.]), 'Gyrus': np.array([0., 1.])}
    labels_onehot = list(map(classes_dict.get, labels2))
    return labels_onehot

topn = 50
filter = ''
#filter = ['','EX_L23','EX_L4','EX_L6','IN','NonNeuron']
tag = f"topmarker_{topn}"
print(tag)
features_np, labels, samplenames = prepare_input_expr(samplelist, seqlen, topn, filter = filter)
new_features_np, new_labels, _ = prepare_input_expr(samplelist2, seqlen, topn, filter = filter)
labels_np = torch.FloatTensor(encode_onehot(labels))
new_labels_np = torch.FloatTensor(encode_onehot(new_labels))

