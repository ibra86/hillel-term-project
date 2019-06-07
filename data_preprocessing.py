import os
import ast
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from constants import DATA_DIR, DATA_RAW_FILE_NAME

print('data preprocessing...')
# -- Loadning/data-preparation
df = pd.read_csv(os.path.join(DATA_DIR, DATA_RAW_FILE_NAME),
                 converters={'var_vec_0': ast.literal_eval, 'var_vec_1': ast.literal_eval})

var_enum_col_list = [f for f in df.columns if 'enum' in f]

df.var_id = df.var_id.apply(lambda x: x[:2] + '%05d' % int(x[2:]))
df[var_enum_col_list] = df[var_enum_col_list].applymap(lambda x: 'V%s' % x)
df = df.sort_values(by=['var_id']).reset_index(drop=True)

cols_reorder = ['var_id'] + var_enum_col_list + [f for f in df.columns if f not in (['var_id'] + var_enum_col_list)]
df = df[cols_reorder]

oh_var_enum = OneHotEncoder(sparse=False).fit_transform(df[var_enum_col_list])

df['var_enum_v'] = oh_var_enum.tolist()
df['var_features'] = df.apply(lambda x: x.var_enum_v + x.var_vec_0 + x.var_vec_1, axis=1)
df = df[['var_id', 'var_features', 'var_obs', 'var_target']]

# assuming targets with 60+ values are abnomal being outliers
df = df[df.var_target < 60].reset_index(drop=True)
print('data preprocessing completed')

# df.to_csv(os.path.join(DATA_DIR, DATA_FILE_NAME), index=False)
