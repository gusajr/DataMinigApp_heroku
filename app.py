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

import SessionState
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from apyori import apriori


#sudo apt-get install python3-tk

PAGES = (
    "Inicio",
    "Análisis exploratorio de datos",#: src.pages.home,
    "Correlaciones",
    "Análisis de componentes principales",
    "Métricas de similitud",
    "Clustering particional",
    "Algoritmo apriori",
    "Pruebas"
    #"Acerca"#: src.pages.resources,
    #"Gallery",#: src.pages.gallery.index,
    #"Vision",#: src.pages.vision,
    #"Acerca"#: src.pages.about,
)

session_state = SessionState.get(datosEDA = 0)
session_state_1 = SessionState.get(datosPCA = 0)

def pagina_inicio():

    st.title('Bienvenid@ a mi Pequeña Herramienta de Inteligencia Artificial (PHIA)')
    st.image(Image.open("ia.png"))
    st.subheader("Esta página fue creada con ❤️ por Gustavo Jiménez, un alumno de la Facultad de Ingeniería de la Universidad Nacional Autónoma de México.")
    st.write("_Para comenzar el análisis de datos puedes dar clic en la **flecha de la esquina superior izquierda** y escoger alguna de las opciones._")

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
    datosEDA = None

    if(st.radio("¿Contiene Header?",("Si","No"))=="No"):
        datosEDA_header = None
    else:
        datosEDA_header = 0

    datosEDA_sep = st.text_input("Separación de los datos: ", value=",")

    datosEDA_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['csv', 'xlsx', 'xls'])

    if(datosEDA_subido is not None):
        if(datosEDA is None):
            with st.spinner('Procesando datos...'):
                datosEDA = pd.DataFrame(pd.read_csv(datosEDA_subido, header=datosEDA_header, sep=datosEDA_sep))
                st.write("**Datos leídos**")
                st.write(datosEDA)
                #my_bar = st.progress(0)
                #for percent_complete in range(100):
                #    time.sleep(0.01)
                #    my_bar.progress(percent_complete + 1)
            st.success('¡Hecho!')

        session_state.datosEDA = datosEDA

        columnasEDA = st.multiselect(
            "Escoja las columnas de su elección: ", list(datosEDA.columns), list(datosEDA.columns)
        )
        if not columnasEDA:
            st.error("Por favor escoja al menos una columna a analizar")
        else:
            datosEDA = datosEDA[columnasEDA]
            VariableValoresAtipicos = st.multiselect(
                    "Escoja las columnas de su elección para visualizar valores atípicos: ", list(datosEDA.columns)
                )
            if not VariableValoresAtipicos:
                st.error("Por favor escoja al menos una columna a analizar para valores atípicos")
            else:
                if st.button("Iniciar ejecución"):
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
                    
                    #VariableValoresAtipicos = columnasEDA_atipicos
                    #fig, axes = plt.subplots(1,1, figsize=(20,20))
                    #fig.suptitle("valores atípicos")
                    #sns.set(rc={'figure.figsize':(10,8)})
                    with st.spinner('Procesando diagrama de caja...'):
                        for col in VariableValoresAtipicos:
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

    #plt.show()
    #if(DatosEDA is not None):
    #    DatosEDA

def pagina_correlaciones():
    try:
        st.header("**Identificación de relaciones entre pares de variables**")
        st.write("**1. Matriz de correlaciones**")
        st.write(session_state.datosEDA.corr())
        st.write("**2. Mapa de calor de la matriz de correlaciones**")
        plt.figure(figsize=(14,14))
        sns.heatmap(session_state.datosEDA.corr(), cmap='RdBu_r', annot=True)
        plt.show()
        st.pyplot()
    except:
        st.write("**Datos no cargados o incompatibles, por favor dirígete a la pestaña de Análisis Exploratorio de Datos para cargar los datos.**")

def pagina_analisisComponentesPrincipales():
    #"""
    # Componentes principales
    #En esta página se visualizan las componentes principales de un set de datos.
    #"""

    st.title('Análisis de Componentes principales (PCA)')
    #st.write("En esta página se visualizan las componentes principales de un set de datos.")

    st.header("**Importación de datos**")
    st.write("**1. Lectura de datos**")

    if(st.radio("¿Contiene Header?",("Si","No"))=="No"):
        datosPCA_header = None
    else:
        datosPCA_header = 0

    datosPCA_sep = st.text_input("Separación de los datos: ", value=",")

    datosPCA_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['csv', 'xlsx', 'xls'])

    if(datosPCA_subido is not None):
        with st.spinner('Procesando datos...'):
            datosPCA = pd.DataFrame(pd.read_csv(datosPCA_subido, header=datosPCA_header, sep=datosPCA_sep))
            st.write("**Datos leídos**")
            st.write(datosPCA)
            #my_bar = st.progress(0)
            #for percent_complete in range(100):
            #    time.sleep(0.01)
            #    my_bar.progress(percent_complete + 1)
        st.success('¡Hecho!')

        session_state_1.datosPCA = datosPCA

        columnasPCA = st.multiselect(
            "Escoja las columnas de su elección: ", list(datosPCA.columns), list(datosPCA.columns)
        )
        if not columnasPCA:
            st.error("Por favor escoja al menos una columna a analizar")
        else:
            if st.button("Iniciar ejecución"):
                
                datosPCA = datosPCA[columnasPCA]
                st.header("**Estandarización de los datos**")
                normalizar = StandardScaler()
                normalizar.fit(datosPCA)
                datosPCA_normalizada = normalizar.transform(datosPCA)
                #st.write(pd.DataFrame(datosPCA_normalizada))
                st.write(datosPCA_normalizada.shape)
                st.write(pd.DataFrame(datosPCA_normalizada, columns=datosPCA.columns))
                
                st.header("**Matriz de covarianzas y correlaciones, varianza y componentes**")
                Componentes = PCA(n_components=len(datosPCA.columns))
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

def pagina_similitudes():
    st.title('Métricas de similitud')
    try:
        metrica = st.radio("Elija la métrica de similitud de su elección: ", ("Euclideana", "Chebyshev", "Manhattan","Minkowski"))
        separador = ""
        anuncio = separador.join(["**Distancia ",metrica,"**"])

        st.header(anuncio)

        if(metrica == "Euclideana"):
            metrica = "euclidean"
        elif(metrica == "Chebyshev"):
            metrica = "chebyshev"
        elif(metrica == "Manhattan"):
            metrica = "cityblock"
        elif(metrica == "Minkowski"):
            metrica = "minkowski"
            valor_lambda = st.text_input("Inserte el valor de lambda: ", value="1.5")

        if(metrica != "minkowski"):
            Distancias = pd.DataFrame(cdist(session_state_1.datosPCA, session_state_1.datosPCA, metric=metrica))
            Distancias

        else:
            Distancias = pd.DataFrame(cdist(session_state_1.datosPCA, session_state_1.datosPCA, metric=metrica, p=float(valor_lambda)))
            Distancias

    except:
        st.write("**Datos no cargados o incompatibles, por favor dirígete a la pestaña de Análisis de Componentes principales para cargar los datos.**")

def pagina_clustering():
    st.title('Clustering particional')
    try:
        st.header("**Acceso a los datos**")
        session_state_1.datosPCA
        st.header("**Selección de características**")
        columna = st.radio("Seleccione la columna principal a comparar para generar la matriz de correlaciones: ", session_state_1.datosPCA.columns)
        with st.spinner('Procesando matriz...'):
            sns.pairplot(session_state_1.datosPCA, hue=columna)
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        st.success('¡Hecho!')
        
        st.write("**Comparación de relación entre variables**")
        clustering_x = st.radio("Seleccione la primer variable", session_state_1.datosPCA.columns)
        clustering_y = st.radio("Seleccione la segunda variable", session_state_1.datosPCA.columns)
        sns.scatterplot(x=clustering_x,y=clustering_y,data=session_state_1.datosPCA,hue=columna)
        plt.title("Gráfico de dispersión")
        plt.xlabel(clustering_x)
        plt.ylabel(clustering_y)
        plt.show()
        st.pyplot()

        st.write("**Matriz de correlaciones**")
        datosPCA_corr = session_state_1.datosPCA.corr(method="pearson")
        datosPCA_corr
        st.write("**Mapa de calor**")
        plt.figure(figsize=(14,7))
        MatrizInf = np.triu(datosPCA_corr)
        sns.heatmap(datosPCA_corr, cmap="RdBu_r", annot=True, mask=MatrizInf)
        plt.show()
        st.pyplot()

        st.header("**Elección de variables**")
        columnasPCA = st.multiselect(
            "Escoja las columnas de su elección: ", list(session_state_1.datosPCA.columns), list(session_state_1.datosPCA.columns)
        )
        if not columnasPCA:
            st.error("Por favor escoja al menos una columna a analizar")
        else:
            datosPCA = session_state_1.datosPCA[columnasPCA]
        
        datosPCA

        st.header("**Algoritmo K-means**")
        SSE=[]
        for i in range(2,12):
            km=KMeans(n_clusters=i,random_state=0)
            km.fit(datosPCA)
            SSE.append(km.inertia_)

        #Se grafica SSE en función de k
        plt.figure(figsize=(10,7))
        plt.plot(range(2,12),SSE,marker="o")
        plt.xlabel("Cantidad de clústers *k*")
        plt.ylabel("SSE")
        plt.title("Método del codo")
        plt.show()
        st.pyplot()

        kl = KneeLocator(range(2,12),SSE,curve="convex",direction="decreasing")
        st.write("**Número de clústers con mayor convergencia**")
        st.text(kl.elbow)

        st.write("**Clúster al que pertenece cada registro**")
        MParticional = KMeans(n_clusters=4, random_state=0).fit(session_state_1.datosPCA)
        MParticional.predict(session_state_1.datosPCA)
        MParticional.labels_

        st.write("**Se añade la columna de número de clúster de cada registro**")
        session_state_1.datosPCA["clusterP"] = MParticional.labels_
        session_state_1.datosPCA

        st.write("**Número de registros en cada centroide**")
        st.write(session_state_1.datosPCA.groupby(["clusterP"])["clusterP"].count())

        st.write("**Gráfica para mostrar clústers por color**")
        plt.figure(figsize=(10,7))
        plt.scatter(datosPCA.iloc[:,0], datosPCA.iloc[:,1],c=MParticional.labels_,cmap="rainbow")
        plt.show()
        st.pyplot()

        st.write("**Media de los registros en cada centroide**")
        CentroidesP = MParticional.cluster_centers_
        st.write(pd.DataFrame(CentroidesP.round(4),columns=datosPCA.columns))
        #from plotnine import *

        plt.rcParams["figure.figsize"] = (10,7)
        plt.style.use("ggplot")
        colores=["red","blue","green","yellow"]
        asignar=[]
        for row in MParticional.labels_:
            asignar.append(colores[row])

        st.write("**Gráfica 3D de la clasificación de los registros**")

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(datosPCA.iloc[:,0],datosPCA.iloc[:,1],datosPCA.iloc[:,2],marker="o",c=asignar,s=60)
        ax.scatter(CentroidesP[:,0],CentroidesP[:,1],CentroidesP[:,2],marker="*",c=colores,s=1000)
        plt.show()
        st.pyplot()
        #st.pyplot(ggplot.draw(fig))

        st.write("**Registros más cercanos al centroide**")
        Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, datosPCA)
        Cercanos

        #st.write("**1. Dimensiones de la data**")
    except:
        st.write("**Datos no cargados o incompatibles, por favor dirígete a la pestaña de Análisis de Componentes principales para cargar los datos.**")

def pagina_apriori():
    st.title('Algoritmo Apriori')
    #st.write("En esta página se visualizan las componentes principales de un set de datos.")

    st.header("**Importación de datos**")
    st.write("**1. Lectura de datos**")

    if(st.radio("¿Contiene Header?",("Si","No"))=="No"):
        datosApriori_header = None
    else:
        datosApriori_header = 0

    datosApriori_sep = st.text_input("Separación de los datos: ", value=",")

    datosApriori_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['csv', 'xlsx', 'xls'])

    if(datosApriori_subido is not None):
        with st.spinner('Procesando datos...'):
            datosApriori = pd.DataFrame(pd.read_csv(datosApriori_subido, header=datosApriori_header, sep=datosApriori_sep))
            st.write("**Datos leídos**")
            st.write(datosApriori)
        st.success('¡Hecho!')

        columnasPCA = st.multiselect(
            "Escoja las columnas de su elección: ", list(datosApriori.columns), list(datosApriori.columns)
        )

        soporte = st.text_input("Soporte mínimo para el análisis",value="0.028")
        confianza = st.text_input("Confianza mínima para el análisis",value="0.3")
        elevacion = st.text_input("Elevación mínima para el análisis",value="1.01")

        if not columnasPCA:
            st.error("Por favor escoja al menos una columna a analizar")
        else:
            if st.button("Iniciar ejecución"):
                st.header("**Procesamiento de datos**")
                listaObjetos = []
                for i in range(0, 7460):
                    listaObjetos.append([str(datosApriori.values[i,j]) 
                    for j in range(0, 20)])
                st.write(listaObjetos)

                ListaReglas = apriori(listaObjetos, min_support=float(soporte), min_confidence=float(confianza), min_lift=float(elevacion))
                ReglasAsociacion = list(ListaReglas)
                st.write("**Número de reglas de asociación**")
                st.text(len(ReglasAsociacion))
                ReglasAsociacionVerdaderas = []
                for i in range (len(ReglasAsociacion)):
                    if("nan" not in ReglasAsociacion[i].items):
                        ReglasAsociacionVerdaderas.append(ReglasAsociacion[i])
                st.json(ReglasAsociacionVerdaderas)

                st.write("ok")

def pagina_pruebas():
    st.title('Prueba')
    st.header("**Importación de datos**")
    st.write("**1. Lectura de datos**")

    if(st.radio("¿Contiene Header?",("Si","No"))=="No"):
        datos_header = None
    else:
        datos_header = 0

    datos_sep = st.text_input("Separación de los datos: ", value=",")

    datos_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['csv', 'xlsx', 'xls'])

    soporte = st.text_input("Soporte mínimo para el análisis",value="0.028")
    confianza = st.text_input("Confianza mínima para el análisis",value="0.3")
    elevacion = st.text_input("Elevación mínima para el análisis",value="1.01")

    if(datos_subido is not None):
        with st.spinner('Procesando datos...'):
            datos = pd.DataFrame(pd.read_csv(datos_subido, header=datos_header, sep=datos_sep))
            st.write("**Datos leídos**")
            st.write(datos)
            #my_bar = st.progress(0)
            #for percent_complete in range(100):
            #    time.sleep(0.01)
            #    my_bar.progress(percent_complete + 1)
        st.success('¡Hecho!')
    
        listaObjetos = []
        for i in range(0, 7460):
            listaObjetos.append([str(datos.values[i,j]) 
            for j in range(0, 20)])
        st.write(listaObjetos)

        ListaReglas = apriori(listaObjetos, min_support=float(soporte), min_confidence=float(confianza), min_lift=float(elevacion))
        ReglasAsociacion = list(ListaReglas)
        st.write("**Número de reglas de asociación**")
        st.text(len(ReglasAsociacion))
        ReglasAsociacionVerdaderas = []
        for i in range (len(ReglasAsociacion)):
            if("nan" not in ReglasAsociacion[i].items):
                ReglasAsociacionVerdaderas.append(ReglasAsociacion[i])
        st.json(ReglasAsociacionVerdaderas)


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

    elif(selection == "Correlaciones"):
        pagina_correlaciones()

    elif(selection == "Métricas de similitud"):
        pagina_similitudes()

    elif(selection == "Clustering particional"):
        pagina_clustering()

    elif(selection == "Algoritmo apriori"):
        pagina_apriori()

    elif(selection == "Pruebas"):
        pagina_pruebas()
        
if __name__ == "__main__":
    main()

# st.dataframe() st.table()