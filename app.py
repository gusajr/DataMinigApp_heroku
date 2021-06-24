import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.proto.DataFrame_pb2 import DataFrame   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib

#import pandas as pd                                   # Para la manipulación y análisis de datos
#import numpy as np                                    # Para crear vectores y matrices n dimensionales
#import matplotlib.pyplot as plt                       # Para la generación de gráficas a partir de los datos
#import seaborn as sns                                 # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import io 
from PIL import Image

#sudo apt-get install python3-tk

PAGES = (
    "Inicio",
    "Análisis exploratorio de datos",#: src.pages.home,
    "Análisis de componentes principales",
    #"Acerca"#: src.pages.resources,
    #"Gallery",#: src.pages.gallery.index,
    #"Vision",#: src.pages.vision,
    #"Acerca"#: src.pages.about,
)

def pagina_inicio():

    st.title('Bienvenid@ a mi Pequeña Herramienta de Inteligencia Artificial (PHIA)')
    st.image(Image.open("ia.png"))
    st.subheader("Esta página fue creada con ❤️ por Gustavo Jiménez, un alumno de la Facultad de Ingeniería de la Universidad Nacional Autónoma de México.")
    st.write("_Para comenzar el análisis de datos puedes dar clic en la flecha de la esquina superior izquierda para poder escoger alguna de las opciones._")

def pagina_EDA():
    #%matplotlib inline                
    # Para generar imágenes dentro del cuaderno

    st.title('Análisis Exploratorio de Datos (EDA)')

    #st.write(pd.DataFrame({
    #    'first column': [1, 2, 3, 4],
    #    'second column': [10, 20, 30, 40]
    #}))

    #option1 = st.selectbox(
    #    'Which number do you like best?',
    #    [1,2,3,4,5]
    #)

    #st.write('You selected option:', option1)

    st.header("**Importación de datos**")
    st.write("**1. Lectura de datos**")

    datosEDA_subido = st.file_uploader("Escoge el archivo que quieres analizar: ")

    if(datosEDA_subido is not None):
        with st.spinner('Procesando el archivo...'):
            datosEDA = pd.DataFrame(pd.read_csv(datosEDA_subido))
            st.write("**Datos leídos**")
            st.write(datosEDA)
            #my_bar = st.progress(0)
            #for percent_complete in range(100):
            #    time.sleep(0.01)
            #    my_bar.progress(percent_complete + 1)
        st.success('¡Hecho!')

        st.header("**Descripción de la estructura de los datos**")
        st.write("**1. Dimensiones de la data**")
        datosEDA.shape
        st.write("**2. Tipos de dato de las variables**")
        datosEDA.dtypes

        st.header("**Identificación de datos faltantes**")
        buffer = io.StringIO() 
        datosEDA.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)
        #with open("df_info.txt", "w", encoding="utf-8") as f:
        #    f.write(info) 

        st.header("**Detección de valores atípicos**")
        st.write("**1. Distribución de variables numéricas**")
        datosEDA.hist(figsize=(14,14), xrot=45)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.write("**2. Resumen estadístico de variables numéricas**")
        st.write(datosEDA.describe())

        st.write("**3. Diagramas para detectar posibles valores atípicos**")
        VariablesValoresAtipicos = ['Price', 'Landsize', 'BuildingArea', 'YearBuilt']
        #fig, axes = plt.subplots(1,1, figsize=(20,20))
        #fig.suptitle("valores atípicos")
        #sns.set(rc={'figure.figsize':(10,8)})
        with st.spinner('Procesando diagrama de caja...'):
            for col in VariablesValoresAtipicos:
                sns.boxplot(col, data=datosEDA)
                plt.show()
                st.pyplot()
        st.success('¡Hecho!')
        
        st.write("**4. Distribución de variables categóricas**")
        st.write(datosEDA.describe(include='object'))
        
        with st.spinner('Procesando histograma...'):
            for col in datosEDA.select_dtypes(include='object'):
                if datosEDA[col].nunique()<10:
                    sns.countplot(y=col, data=datosEDA)
                    plt.show()
                    st.pyplot()
        st.success('¡Hecho!')

        for col in datosEDA.select_dtypes(include='object'):
            if datosEDA[col].nunique() < 10:
                st.write((datosEDA.groupby(col).agg(['mean'])))
        
        st.header("**Identificación de relaciones entre pares de variables**")
        st.write("**1. Matriz de correlaciones**")
        st.write(datosEDA.corr())
        st.write("**2. Mapa de calor de la matriz de correlaciones**")
        plt.figure(figsize=(14,14))
        sns.heatmap(datosEDA.corr(), cmap='RdBu_r', annot=True)
        plt.show()
        st.pyplot()

    #plt.show()
    #if(DatosEDA is not None):
    #    DatosEDA

def pagina_analisisComponentesPrincipales():
    #"""
    # Componentes principales
    #En esta página se visualizan las componentes principales de un set de datos.
    #"""

    st.title('Componentes principales (PCA)')
    #st.write("En esta página se visualizan las componentes principales de un set de datos.")

    st.header("**Importación de datos**")
    st.write("**1. Lectura de datos**")

    datosPCA_subido = st.file_uploader("Escoge el archivo que quieres analizar: ")

    if(datosPCA_subido is not None):
        with st.spinner('Procesando el archivo...'):
            datosPCA = pd.DataFrame(pd.read_csv(datosPCA_subido))
            st.write("**Datos leídos**")
            st.write(datosPCA)
            #my_bar = st.progress(0)
            #for percent_complete in range(100):
            #    time.sleep(0.01)
            #    my_bar.progress(percent_complete + 1)
        st.success('¡Hecho!')
        st.header("**Estandarización de los datos**")
        normalizar = StandardScaler()
        normalizar.fit(datosPCA)
        datosPCA_normalizada = normalizar.transform(datosPCA)
        #st.write(pd.DataFrame(datosPCA_normalizada))
        st.write(datosPCA_normalizada.shape)
        st.write(pd.DataFrame(datosPCA_normalizada, columns=datosPCA.columns))
        
        st.header("**Matriz de covarianzas y correlaciones, varianza y componentes**")
        Componentes = PCA(n_components=10)
        Componentes.fit(datosPCA_normalizada)
        X_Comp = Componentes.transform(datosPCA_normalizada)
        st.write(pd.DataFrame(X_Comp)) #Componentes.components_

        st.header("**Elección del número de componentes principales (eigen-vectores)**")
        Varianza = Componentes.explained_variance_ratio_
        st.write("Eigenvalues:",Varianza)
        st.write("Varianza acumulada: ",sum(Varianza[0:5]))

        plt.plot(np.cumsum(Componentes.explained_variance_ratio_))
        plt.xlabel("Número de componentes")
        plt.ylabel("Varianza acumulada")
        plt.grid()
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        st.header("Análisis de proporción de relevancias (cargas)")
        st.write(pd.DataFrame(abs(Componentes.components_)))
        CargasComponentes = pd.DataFrame(Componentes.components_, columns=datosPCA.columns)
        st.write(CargasComponentes)
        CargasComponentes = pd.DataFrame(abs(Componentes.components_), columns=datosPCA.columns)
        st.write(CargasComponentes)

def main():

    st.set_page_config(
        page_title="PHIA",
        page_icon="🤖",
        layout="centered",
        #initial_sidebar_state="expanded",
    )

    #"""Main function of the App"""
    st.sidebar.title("Comencemos...")
    selection = st.sidebar.radio("Da clic en la función que te gustaría utilizar: ", PAGES)

    df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

    # df

    #if st.checkbox('Show dataframe'):
    #    chart_data = pd.DataFrame(
    #       np.random.randn(20, 3),
    #       columns=['a', 'b', 'c'])

    #    chart_data

    #option2 = st.sidebar.selectbox(
    #    'Which number do you like best?',
    #    df['first column']
    #    )

    #st.sidebar.write('You selected:', option2)

    if(selection == "Análisis exploratorio de datos"):
        pagina_EDA()

    elif(selection == "Inicio"):
        pagina_inicio()

    elif(selection == "Análisis de componentes principales"):
        pagina_analisisComponentesPrincipales()
        
if __name__ == "__main__":
    main()

# st.dataframe() st.table()