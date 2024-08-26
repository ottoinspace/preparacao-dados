import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head(15))

# Codificao one-hot para 'estado_civil'
df = pd.concat([df, pd.get_dummies(df['estado_civil'], prefix='estado_civil')], axis=1)

print("\nDataFrame apos codificao one-hot para 'estado_civil':\n", df.head())

# Codificao ordinal para 'nivel_educacao'
educacao_ordem = {'Ensino Fundamental': 1, 'Ensino Médio': 2, 'Graduação': 3, 'Pós-graduação': 4, 'Mestrado': 5, 'Doutorado': 6}
df['nivel_educacao_ordinal'] = df['nivel_educacao'].map(educacao_ordem)

print("\nDataFrame apos codificao ordinal para 'ordinal':\n", df.head())

# Transformar 'area_atuacao' em categorias codificadas usando o metodo .cat.codes
df['area_atuacao_cod'] = df['area_atuacao'].astype('category').cat.codes

print("\nDataFrame apos transformar 'area_atuacao' em codigo numericos:\n", df.head())

# LabelEncoder para 'estado'
# LabelEncoder converte cada valor unico em numeros de 0 a n_classes-1
label_encoder = LabelEncoder()
df['estado_cod'] = label_encoder.fit_transform(df['estado'])

print("\nDataFrame apos aplicar LabelEncoder em 'estado':\n", df.head())
