import os
import pandas as pd
import numpy as np
import torch

os.chdir('data2024.v2')

n_extend_plan = 50
seqlen = 20

mat_columnanno = pd.read_csv('data2024/column2annot_man_res_corrected.csv')
samplelist_all = np.unique(mat_columnanno['section'])
samplelist2 = list(pd.read_csv('samplelist_test.txt').loc[:,'x'])
samplelist = np.setdiff1d(samplelist_all, samplelist2 + ['T114','T145'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64

samplelist_replicate2 = pd.read_csv('data2024/maca_repeat/column2annot_man_res.csv').section.unique()[:18]
samplelist_replicate3 = pd.read_csv('data2024/maca_repeat/column2annot_man_res.csv').section.unique()[18:][:13]

def get_input_by_sample(sampletag, n_extend_plan, seqlen, macaque = 1, min_d = 5):
    print(sampletag)
    if macaque == 1:
        mat_cl = pd.read_csv(f'spa_matrix/cell-meta-matrix_1105/cell-meta-matrix.{sampletag}.tsv', sep='\t')
        mat_column = pd.read_csv(f'data2024/cell_column/{sampletag}_cell2column.csv')
        mat_columnanno = pd.read_csv('data2024/column2annot_man_res_corrected.csv')
    if macaque == 2:
        mat_cl = pd.read_csv(f'spa_matrix_1106/cell-meta-matrix2.{sampletag}.tsv', sep='\t')
        mat_column = pd.read_csv(f'data2024/maca_repeat/cell_column/{sampletag}_cell2column.csv')
        mat_columnanno = pd.read_csv('data2024/maca_repeat/column2annot_man_res.csv')
    if macaque == 3:
        mat_cl = pd.read_csv(f'spa_matrix_1118/cell-meta-matrix2.{sampletag}.tsv', sep='\t')
        mat_column = pd.read_csv(f'data2024/maca_repeat/cell_column/{sampletag}_cell2column.csv')
        mat_columnanno = pd.read_csv('data2024/maca_repeat/column2annot_man_res.csv')
    rename2 = pd.read_csv('anno.264clusters.renameV1212.txt', sep='\t')
    rename2.set_index('anno2', inplace=True)
    mat_cl2 = pd.merge(mat_cl, mat_column, on='cell_id')
    mat_cl2.rename(columns={mat_cl2.columns[-1]: 'column_id_ori'}, inplace=True)
    a = pd.DataFrame({
        'reid': range(1, len(mat_cl2['column_id_ori'].unique()) + 1)
    }, index=sorted(mat_cl2['column_id_ori'].unique()))
    mat_cl2['column_id'] = mat_cl2['column_id_ori'].map(a['reid'])
    mat_columnanno = mat_columnanno[mat_columnanno['section'] == sampletag]
    mat_columnanno = mat_columnanno.sort_values(by='column')
    if len(mat_columnanno) > 1:
        n_extend = int(min([(mat_columnanno.iloc[i]['column'] - mat_columnanno.iloc[i-1]['column']) for i in range(1, len(mat_columnanno))]) / 2)
        n_extend = min(n_extend, n_extend_plan)
    else:
        n_extend = n_extend_plan
    print(n_extend)
    def apply_function(entry):
        if entry['column'] in mat_cl2['column_id_ori'].values:
            column_mid = mat_cl2.loc[mat_cl2['column_id_ori'] == entry['column'], 'column_id'].values[0]
        else:
            d = abs(mat_cl2['column_id_ori'] - entry['column'])
            if d.min() <= min_d:
                column_mid = mat_cl2.loc[mat_cl2['column_id_ori'] == mat_cl2['column_id_ori'].iloc[d.idxmin()], 'column_id'].values[0]
            else:
                print(f"Error! min(d) is {d.min()}")
                return
        start = int(column_mid) - n_extend
        end = int(column_mid) + n_extend
        if start < 1:
            start = 1  
        if end > mat_cl2['column_id'].max():
            end = mat_cl2['column_id'].max()
        filtered_mat_cl2 = mat_cl2[(mat_cl2['column_id'] >= start) & (mat_cl2['column_id'] <= end)]
        pivot_table = filtered_mat_cl2.pivot_table(index='cluster', columns='column_id', aggfunc='size', fill_value=0)
        pivot_table = pivot_table.reindex(rename2.index, fill_value=0)
        pivot_table.columns = [f'{sampletag}_{entry["area"]}_{col}_{entry["g/s"]}' for col in pivot_table.columns]
        return pivot_table.T
    res = []
    curve = []
    symm = []
    samplename = []
    for i in range(mat_columnanno.shape[0]):
        entry = mat_columnanno.iloc[i,:]
        result = apply_function(entry)
        if result is not None:
            res.append(result)
            curve.append(entry['curve_ratio'])
            symm.append(entry['symm'])
            samplename.append(result.index)
    from collections import Counter
    def mode(data):
        counter = Counter(data)
        max_count = max(counter.values())
        return [key for key, count in counter.items() if count == max_count]
    def most_frequent(list):
      return max(set(list), key=list.count)
    features = []
    labels_gs = []
    labels_region = []
    labels_curve = []
    labels_symm = []
    labels_samplename = []
    for item_i in range(len(res)):
        item = res[item_i]
        for i in range(item.shape[0] - seqlen):
            features += [item.iloc[i:(i+seqlen),:]]
            labels_gs += [item.index[i].split('_')[-1]]
            columnid_i = int(item.index[i].split('_')[2])
            filtered_mat_cl2 = mat_cl2[(mat_cl2['column_id'] >= columnid_i) & (mat_cl2['column_id'] <= columnid_i+seqlen)] #****
            labels_region += [most_frequent([item.split('-')[1] for item in filtered_mat_cl2['gene_area']])]
            labels_curve += [curve[item_i]]
            labels_symm += [symm[item_i]]
            labels_samplename += [samplename[item_i][i]]
    tensor_list = [torch.tensor(item.values, dtype=torch.float32) for item in features]
    print(torch.stack(tensor_list).shape, len(labels_gs), len(labels_region))
    return tensor_list, labels_gs, labels_region, labels_curve, labels_symm, labels_samplename
