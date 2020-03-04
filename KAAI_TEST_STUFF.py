import pandas as pd
from utils import composition
from utils import composition_CBFV

df = pd.read_csv('data/material_properties/ael_bulk_modulus_vrh/train.csv')
df.columns = ['formula', 'target']
df['formula'] = df['formula'].str.split('_ICSD').str[0]

comps = composition.generate_features(df)
comps_cbfv =

X, y, formula, skipped = composition_CBFV.generate_features(df)

X_new = X.values.reshape(X.shape[0], 5, -1)

len0 = comps_cbfv[comps_cbfv.index.str.contains('elem0')].shape
len1 = comps_cbfv[comps_cbfv.index.str.contains('elem1')].shape
len2 = comps_cbfv[comps_cbfv.index.str.contains('elem2')].shape
len3 = comps_cbfv[comps_cbfv.index.str.contains('elem3')].shape

225 / 4
