import pandas as pd
import numpy as np
from scipy import stats

pd.set_option('display.width', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

# Transformacao logaritmica
df['salario_log'] = np.log1p(df['salario'])  # log1p e usado para evitar problemas com valores zero

print("\nDataFrame apos transformacao logaritimca no 'salario':\n", df.head())

# Transformacao Box-cox
df['salario_boxcox'], _ = stats.boxcox(df['salario'] + 1)  # + 1 para nao ter valores negativos

print("\nDataFrame apos transformacao Box-Cox no 'salario':\n", df.head())

# Codifiao da frequencia para 'estado
estado_freq = df['estado'].value_counts() / len(df)
df['estado_freq'] = df['estado'].map(estado_freq)

print("\nDataFrame apos codificao de frequencias para 'estado':\n", df.head())

# Interacoes
df['interacao_idade_filho'] = df['idade'] * df['numero_filhos']

print("\nDataFrame apos criacao de interacao entre 'idade' e 'numero_filhos':\n", df.head())