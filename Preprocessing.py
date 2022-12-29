import numpy as np
import pandas as pd
from sklearn import preprocessing
# python pip install -U scikit-learn pandas numpy
class CleanData:
    _le = preprocessing.LabelEncoder()
    '''
    CleanData: Clase para limpiar DataFrame y prepararlos para el entrenamiento de un modelo.
    Autor: Mateo Jesus Cabello Rodriguez
    '''
    def clean(df,drop_columns=None,delete_null_values=False,fill_null_values=False):
        '''
        Prepara un DataFrame para ser usado en predicciones.
        df: 'Requiere de un objecto DataFrame'
        drop_columns: 'Listado de columnas que se quieren eliminar del DataFrame'
        delete_null_values: 'Boolean que indica si se borran filas/tuplas con valores nulos'
        fill_null_values: 'Boolean que indica si se rellena con el valor de la media las filas/tuplas que tengan valores nulos'
        '''
        if(df.__class__ != pd.DataFrame):
            print("Se esperaba un DataFrame:",df.__class__)
            return None
        if(fill_null_values):
            df = CleanData.fill_numeric_null_rows(df)
        if(delete_null_values):
            df = CleanData.delete_null_rows(df)
        df = CleanData.clean_columns(df)
        df = CleanData.convert_types(df)
        if(drop_columns != None and type(drop_columns) == list):
            df = CleanData.drop_colums(df,drop_columns)
        return df
    def delete_null_rows(df):
        """
        Borra las filas/tuplas del dataframe que contengan algun valor nulo.
        df: 'Requiere de un objecto DataFrame'
        """
        return df.copy().dropna(inplace = True)
    def fill_numeric_null_rows(df):
        """
        Rellena con la media las filas/tuplas del dataframe que contengan algun valor nulo.
        df: 'Requiere de un objecto DataFrame'
        """
        df = df.copy()
        numeric_columns = CleanData.get_numeric_columns(df)
        for column in numeric_columns:
            df.fillna(int(df[column].mean()), inplace = True)
        return df
        
    def drop_colums(df,drop_columns):
        """
        Borra las columnas del dataframe.
        df: 'Requiere de un objecto DataFrame'
        drop_columns: 'Listado de columnas que se quieren eliminar del DataFrame'
        """
        for column in drop_columns:
            df.drop(column.lower(), axis=1, inplace = True)
        return df
    def clean_columns(df):
        '''
        Transforma el nombre de las columnas a lowercase.
        df: 'Requiere de un objecto DataFrame'
        '''
        df.columns = map(str.lower, df.columns)
        return df
    def convert_types(df):
        '''
        Convierte las columnas del DataFrame en sus tipos correspondientes.
        Los tipos son numeric, date y str.
        df: 'Requiere de un objecto DataFrame'
        '''
        df = CleanData.convert_numeric_types(df)
        df = CleanData.convert_date_types(df)
        df = CleanData.convert_str_types(df)
        return df
    def convert_numeric_types(df):
        '''
        Convierte las columnas del DataFrame de tipo numerico en int o float.
        df: 'Requiere de un objecto DataFrame'
        '''
        numeric_columns = CleanData.get_numeric_columns(df)
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column],errors='coerce')
        return df
    def convert_date_types(df):
        '''
        Convierte las columnas del DataFrame de tipo fecha en datetime.
        df: 'Requiere de un objecto DataFrame'
        '''
        date_columns = CleanData.get_date_columns(df)
        for column in date_columns:
            df[column] = pd.to_datetime(df[column],errors='coerce')
        return df
    def convert_str_types(df):
        '''
        Convierte las columnas del DataFrame de tipo cadena en str.
        Limpia los espacios de las cadenas y los reemplaza por _ .
        df: 'Requiere de un objecto DataFrame'
        '''
        df= df.applymap(lambda s:s.lower().strip().replace(' ','_') if type(s) == str else s)
        return df
    def get_categorical_columns(df):
        '''
        Devuelve las columnas que son de tipo categórico.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.dtypes[df.dtypes == 'object'].to_dict().keys()
    def get_numeric_columns(df):
        '''
        Devuelve las columnas que son de tipo numerico.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.select_dtypes(include=np.number).columns.tolist()
    def get_date_columns(df):
        '''
        Devuelve las columnas que son de tipo datetime.
        df: 'Requiere de un objecto DataFrame'
        '''
        return df.select_dtypes(include=np.datetime64).columns.tolist()
    def transform_categorical_to_numeric(df):
        '''
        Convierte las columnas del DataFrame que son categóricas en int.
        df: 'Requiere de un objecto DataFrame'
        '''
        categorical = CleanData.get_categorical_columns(df)
        for column in categorical:
            df[column] = CleanData._le.fit_transform(df[column].values)
        return df