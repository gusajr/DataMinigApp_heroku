import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.proto.DataFrame_pb2 import DataFrame
import seaborn as sns

from numpy.random import random_integers
from scipy.sparse.construct import rand

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import io 

import SessionState
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
from app_aux import *
import time
from sklearn import linear_model 
from sklearn import model_selection
from sklearn.metrics import classification_report 
import math

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score
#from PIL import Image
#from random import randint
#from session_state import get_session_state
#from apyori import apriori
#from scipy.spatial.distance import cdist

#sudo apt-get install python3-tk
global datos, datos_subido
datos = None
datos_subido = None
st.caching.clear_cache()

# class PaginaDatos:
#     datos = 0
#     datos_sep = 0
#     datos_subido = 0
#     datos_header = 0
#     datos_csv = 0

# global paginaDatos
# paginaDatos = PaginaDatos()

PAGES = (
    "🤖 Inicio",
    "💾 Carga de datos",
    "📖 Análisis exploratorio de datos",
    "📊 Análisis de componentes principales",
    "👪 Clustering particional",
    "🔎 Clasificación",
    "🔎 Prueba de modelo RL"
)

global session_state
global session_state_1

session_state = SessionState.get(datosEDA = 0, datos = None, valor_reemplazo = 0, widgetKey = 0)
session_state_1 = SessionState.get(datosPCA = 0, clustering_sns_pairplot = None)
session_state_1.clustering_sns_pairplot = None
def pagina_inicio():

    st.title('Bienvenid@ a mi Pequeña Herramienta de Minería de Datos (PHMD)')

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0005)
        my_bar.progress(percent_complete + 1)

    chart_data = pd.DataFrame(
        np.random.randn(50, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)
    

    st.subheader("Esta herramienta fue creada con ❤️ por Gustavo Jiménez, alumno de la Facultad de Ingeniería de la Universidad Nacional Autónoma de México.")
    st.write("_Para comenzar el análisis de datos puedes dar clic en alguna de las opciones de la barra de navegación._")

def pagina_carga_datos():
    
    global datos
    global datos_subido
    global session_state

    st.title('Carga y modificación de datos')
    st.header("Importación de datos")
    st.write("1. Lectura de datos")

    if(st.radio("¿Contiene Header?",("Si","No"),key="1")=="No"):
        datos_header = None
    else:
        datos_header = 0

    datos_sep = st.text_input("Separación de los datos: ", value=",")
    datos_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['csv', 'xlsx', 'xls'], key=session_state.widgetKey)

    if(datos_subido is not None):
        with st.spinner('Procesando datos...'):
            datos_csv = pd.read_csv(datos_subido, header=datos_header, sep=datos_sep)
            session_state.datos_iniciales = pd.DataFrame(datos_csv)
            #st.write(session_state.datos_iniciales)
            session_state.datos_iniciales_0 = session_state.datos_iniciales
            session_state.widgetKey = session_state.widgetKey + 1
        st.success('¡Hecho!')
        session_state.datos = session_state.datos_iniciales
    
    if(session_state.datos is None):
        st.error("Por favor, sube un archivo")

    try:

        datos_sep = 0
        datos_subido = None
        #datos = 0
        datos_csv = 0

        columnas = st.multiselect(
            "Escoge las columnas que utilizarás para los algoritmos: ", list(session_state.datos_iniciales_0.columns), list(session_state.datos.columns), key="ms_datos_1"
        )
        if not columnas:
            st.error("Por favor escoge al menos una columna a analizar")
        
        st.write("Datos iniciales")
        st.write(session_state.datos_iniciales_0[columnas])

        st.success("¡Ya puedes continuar con otra opción!")

        st.header("Reemplazar valores en alguna columna")
        columnaClasificacion = st.selectbox(
            "Escoge la columa a reemplazar: ", columnas, index=0
        )

        with st.spinner("Recargando datos"):
            session_state.datos = session_state.datos_iniciales[columnas]

        if not columnaClasificacion:
            st.error("No se ha seleccionado ninguna columna para sustituir valores")
        else:
            valorSustitucionClasificacionReemplazar = st.text_input("Inserta [valor_a_reemplazar] y seguido de una coma [valor_nuevo]: ")

            if(valorSustitucionClasificacionReemplazar==""):
                st.warning("Ningún valor insertado. Al dar enter se reemplazará el valor")
            else:
                valores = valorSustitucionClasificacionReemplazar.split(",")
                if "," not in valorSustitucionClasificacionReemplazar:
                    st.error("Por favor, inserta una coma entre valor a reemplazar y valor nuevo")
                else:
                        #session_state.datos_iniciales = session_state.datos_iniciales[columnas]
                    session_state.datos[columnaClasificacion] = session_state.datos[columnaClasificacion].replace(valores[0], valores[1])
                    session_state.datos_iniciales[columnaClasificacion] = session_state.datos[columnaClasificacion].replace(valores[0], valores[1])
                    st.warning("Al dar enter se reemplazará el valor")
                    st.success("¡Datos modificados!")
        st.write("Datos leídos con datos modificados, **estos datos serán los que se usen en las demás pestañas**")
        st.write(session_state.datos)
                    
    except Exception as e:
        st.write("Sin datos.")
        #st.exception(e)


def pagina_analisisExploratorioDeDatos():

    st.title('Análisis Exploratorio de Datos (EDA)')

    datosEDA = session_state.datos

    if(datosEDA is not None):
        with st.spinner('Recargando datos...'):
            st.header("Datos")
            st.write("1. Visualización de datos")
            st.write("Datos leídos")
            st.write(datosEDA)
        st.success('¡Hecho!')
        st.warning("Si necesitas editar alguna columna de tu dataset, dirígete a la opción de carga de datos")

    try:
        VariableValoresAtipicos = st.multiselect(
                "Escoge las columnas de tu elección para visualizar valores atípicos: ", 
                pd.DataFrame(datosEDA).select_dtypes(include=np.number).columns.tolist(), 
                pd.DataFrame(datosEDA).select_dtypes(include=np.number).columns.tolist()
            )
        if not VariableValoresAtipicos:
            st.error("Por favor escoge al menos una columna a analizar para valores atípicos. Algunas columnas no se muestran por no ser numéricas")
        else:
            if st.button("Iniciar ejecución",key="b_1"):
                st.header("Descripción de la estructura de los datos")
                st.write("1. Dimensiones de la data")
                st.write("Renglones:")
                datosEDA.shape[0]
                st.write("Columnas:")
                datosEDA.shape[1]
                st.write("2. Tipos de dato de las variables")
                datosEDA.dtypes

                st.header("Identificación de datos faltantes")
                buffer = io.StringIO() 
                datosEDA.info(buf=buffer)
                info = buffer.getvalue()
                st.text(info)

                st.header("Detección de valores atípicos")
                st.write("1. Distribución de variables numéricas")
                datosEDA.hist(figsize=(14,14), xrot=45)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.write("2. Resumen estadístico de variables numéricas")
                st.write(datosEDA.describe())

                st.write("3. Diagramas para detectar posibles valores atípicos")
                with st.spinner('Procesando diagrama de caja...'):
                    for col in VariableValoresAtipicos:
                        sns.boxplot(col, data=datosEDA)
                        plt.show()
                        st.pyplot()
                st.success('¡Hecho!')
                
                try:
                    st.write("4. Distribución de variables categóricas")
                    st.write(datosEDA.describe(include='object'))
                    
                    with st.spinner('Procesando histograma...'):
                        for col in datosEDA.select_dtypes(include='object'):
                            if datosEDA[col].nunique()<10:
                                st.write((datosEDA.groupby(col).agg(['mean'])))
                                sns.countplot(y=col, data=datosEDA)
                                plt.show()
                                st.pyplot()
                    st.success('¡Hecho!')

                    # for col in datosEDA.select_dtypes(include='object'):
                    #     if datosEDA[col].nunique() < 10:
                    #         st.write((datosEDA.groupby(col).agg(['mean'])))
                
                except Exception as e:
                    st.warning("Algunos datos no se muestran por el tipo de datos en el dataset")
                    ##st.exception(e)

                st.header("Identificación de relaciones entre pares de variables")
                st.write("1. Matriz de correlaciones")
                st.write(datosEDA.corr())
                st.write("2. Mapa de calor de la matriz de correlaciones")
                plt.figure(figsize=(14,14))
                sns.heatmap(datosEDA.corr(), cmap='RdBu_r', annot=True)
                plt.show()
                st.pyplot()
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos")
        #st.exception(e)

def pagina_analisisComponentesPrincipales():
 
    #"""
    # Componentes principales
    #En esta página se visualizan las componentes principales de un set de datos.
    #"""

    st.title('Análisis de Componentes Principales (PCA)')

    datosPCA = session_state.datos

    if(datosPCA is not None):
        with st.spinner('Recargando datos...'):
            st.header("Datos")
            st.write("1. Visualización de datos")
            st.write("Datos leídos")
            st.write(datosPCA)
        st.success('¡Hecho!')
        st.warning("Si necesitas editar alguna columna de tu dataset, dirígete a la opción de carga de datos")
        st.warning("Este algoritmo funciona con datos numéricos")

        if st.button("Iniciar ejecución"):

            datosPCA = pd.DataFrame(datosPCA).select_dtypes(include=np.number).replace(np.NaN,0)
            st.header("Estandarización de los datos")
            normalizar = StandardScaler()
            normalizar.fit(datosPCA)
            datosPCA_normalizada = normalizar.transform(datosPCA)
            st.write("1. Dimensiones de la tabla")
            st.write("Renglones:")
            datosPCA_normalizada.shape[0]
            st.write("Columnas:")
            datosPCA_normalizada.shape[1]
            #st.write(datosPCA_normalizada.shape)
            st.write("2. Datos normalizados")
            st.write(pd.DataFrame(datosPCA_normalizada, columns=datosPCA.columns))
            
            st.header("Matriz de covarianzas y correlaciones, varianza y componentes")
            Componentes = PCA(n_components=len(datosPCA.columns))
            Componentes.fit(datosPCA_normalizada)
            X_Comp = Componentes.transform(datosPCA_normalizada)
            st.write(pd.DataFrame(X_Comp))

            st.header("Elección del número de componentes principales (eigen-vectores)")
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
    else:
        st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos")
        ##st.exception(e)

def pagina_clustering():
    st.title('Clustering particional')
    try:
        if(session_state.datos is not None):
            st.header("Datos")
            st.write("1. Visualización de datos")
            with st.spinner("Recargando datos"):
                session_state.datos
            st.success("¡Hecho!")
            st.warning("Si necesitas editar alguna columna de tu dataset, dirígete a la opción de carga de datos")
            st.warning("Este algoritmo funciona con datos numéricos")
            st.header("Selección de características")
            
            st.write("Comparación de relación entre variables")

        #while True:

        session_state.numericos = pd.DataFrame(session_state.datos).select_dtypes(include=np.number).replace(np.NaN,0)


        columna0 = st.selectbox("Selecciona la columna principal a comparar para los ejes: ", session_state.numericos.columns, key="15")
        clustering_x = st.selectbox("Selecciona la primer variable:", session_state.numericos.columns, key="4")
        clustering_y = st.selectbox("Selecciona la segunda variable:", session_state.numericos.columns, key="5")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.scatterplot(x=clustering_x,y=clustering_y,data=session_state.numericos,hue=columna0)
        plt.title("Gráfico de dispersión")
        plt.xlabel(clustering_x)
        plt.ylabel(clustering_y)
        plt.show()
        st.pyplot()

        columna = 0
        columna_tmp = columna
        columna = st.selectbox("Selecciona la columna principal a comparar para generar la matriz de correlaciones: ", session_state.numericos.columns, key="6")
        
        if(st.button("Iniciar ejecución", key="bt_clustering")):
            with st.spinner('Procesando matriz...'):
                if(session_state_1.clustering_sns_pairplot == None  or columna_tmp != columna):
                    clustering_sns_pairplot = sns.pairplot(session_state.numericos, hue=columna)
                    session_state_1.clustering_sns_pairplot = clustering_sns_pairplot
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            st.success('¡Hecho!')

            st.write("Matriz de correlaciones")
            datosPCA_corr = session_state.numericos.corr(method="pearson")
            datosPCA_corr
            st.write("Mapa de calor")
            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(datosPCA_corr)
            sns.heatmap(datosPCA_corr, cmap="RdBu_r", annot=True, mask=MatrizInf)
            plt.show()
            st.pyplot()

            st.header("Variables utilizadas para clustering")
            datosPCA = session_state.numericos
            datosPCA

            st.warning("Recuerda que si necesitas modificar tus datos debes dirigirte a la pestaña de carga de datos")

            st.header("Algoritmo K-means")
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
            st.write("Número de clústers con mayor convergencia")
            st.text(kl.elbow)

            st.write("Clúster al que pertenece cada registro")
            MParticional = KMeans(n_clusters=4, random_state=0).fit(session_state.numericos)
            MParticional.predict(session_state.numericos)
            MParticional.labels_

            st.write("Se añade la columna de número de clúster de cada registro")
            datosPCA["clusterP"] = MParticional.labels_
            datosPCA

            st.write("Número de registros en cada clúster")
            st.write(datosPCA.groupby(["clusterP"])["clusterP"].count())

            st.write("Gráfica para mostrar clústeres por color")
            plt.figure(figsize=(10,7))
            plt.scatter(datosPCA.iloc[:,0], datosPCA.iloc[:,1],c=MParticional.labels_,cmap="rainbow")
            plt.show()
            st.pyplot()

            st.write("Media de los registros en cada clúster")
            CentroidesP = MParticional.cluster_centers_
            st.write(pd.DataFrame(CentroidesP.round(4),columns=datosPCA.columns[0:len(datosPCA.columns)-1]))
            #from plotnine import *

            plt.rcParams["figure.figsize"] = (10,7)
            plt.style.use("ggplot")
            colores=["red","blue","green","yellow"]
            asignar=[]
            for row in MParticional.labels_:
                asignar.append(colores[row])

            st.write("Gráfica 3D de la clasificación de los registros")

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(datosPCA.iloc[:,0],datosPCA.iloc[:,1],datosPCA.iloc[:,2],marker="o",c=asignar,s=60)
            ax.scatter(CentroidesP[:,0],CentroidesP[:,1],CentroidesP[:,2],marker="*",c=colores,s=1000)
            plt.show()
            st.pyplot()
            #st.pyplot(ggplot.draw(fig))

            st.write("Registros más cercanos al centroide")
            Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, datosPCA[0:len(datosPCA.columns)-2])
            Cercanos

        #st.write("1. Dimensiones de la data")
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos.")
        st.exception(e)

#@st.cache(suppress_st_warning=True)
#def clustering_cache_sns(sns_data, columna, columna_tmp):

def pagina_clasificacion():

    datosClasificacion = session_state.datos
    st.title('Clasificación: Regresión Logística')

    try:
        if(datosClasificacion is not None):
            with st.spinner('Recargando datos...'):
                st.header("Datos")
                st.write("1. Visualización de datos")
                st.write("Datos leídos")
                st.write(datosClasificacion)
            st.success('¡Hecho!')

            st.warning("Si necesitas editar alguna columna de tu dataset, dirígete a la opción de carga de datos")

            st.header("Variable predictora")
            columnaClasificacion = st.selectbox(
                "Escoge la variable predictora: ", list(session_state.datos.columns), index=0
            )
            st.write(datosClasificacion.groupby(columnaClasificacion).size())

            st.warning("Recuerda que el algoritmo funciona con valores 1-0, es necesario que hagas estos ajustes desde la pestaña de carga de datos")
            
            st.header("Selección de características")
            
            st.write("Comparación de relación entre variables")

            #while True:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            clustering_x = st.selectbox("Selecciona la primer variable:", session_state.datos.columns, key="12")
            clustering_y = st.selectbox("Selecciona la segunda variable:", session_state.datos.columns, key="13")
            sns.scatterplot(x=clustering_x,y=clustering_y,data=session_state.datos,hue=columnaClasificacion)
            plt.title("Gráfico de dispersión")
            plt.xlabel(clustering_x)
            plt.ylabel(clustering_y)
            plt.show()
            st.pyplot()

            if(st.button("Iniciar ejecución",key="b_2")):

                columna = 0
                columna_tmp = columna

                st.write("Matriz de correlaciones")
                datos_corr = session_state.datos.corr(method="pearson")
                datos_corr
                st.write("Mapa de calor")
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(datos_corr)
                sns.heatmap(datos_corr, cmap="RdBu_r", annot=True, mask=MatrizInf)
                plt.show()
                st.pyplot()
                
                st.warning("Si tienes que realizar eliminación de columnas, recuerda hacerlo en la pestaña de carga de datos")
                X=session_state.datos.drop(columnaClasificacion, axis=1)
                Y=session_state.datos[columnaClasificacion]

                st.header("Aplicación del algoritmo")
                Clasificacion = linear_model.LogisticRegression()
                seed=1234
                X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.2,random_state=seed,shuffle=True)

                pd.DataFrame(X_train)
                pd.DataFrame(Y_train)
                
                st.write("Entrenamiento del modelo a partir de los datos de entrada")
                st.write(Clasificacion.fit(X_train,Y_train))

                st.write("Predicciones probabilísticas")
                Probabilidad = Clasificacion.predict_proba(X_train)
                st.write(pd.DataFrame(Probabilidad))

                st.write("Predicciones con clasificación final")
                Predicciones = Clasificacion.predict(X_train)
                st.write(pd.DataFrame(Predicciones))

                # st.write("Exactitud")
                exactitud = Clasificacion.score(X_train,Y_train)
                # st.info(exactitud)

                st.header("Validación del modelo")

                st.write("Matriz de clasificación")
                PrediccionesNuevas = Clasificacion.predict(X_validation)
                confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, rownames=["Real"], colnames=["Clasificación"])
                confusion_matrix

                st.write("Reporte de la clasificación")
                with st_stdout("info"):
                    print(classification_report(Y_validation,PrediccionesNuevas))

                st.write("Intercepto")
                st.write(Clasificacion.intercept_)
                st.write("Coeficientes")
                st.write(Clasificacion.coef_)

                session_state.modeloRL = Clasificacion.coef_
                session_state.interceptoRL = Clasificacion.intercept_
                session_state.datosRL = X
                session_state.exactitudRL = exactitud

                with st_stdout("info"):
                    for i in range(len(session_state.datosRL.columns)):
                        print("{0:s} tiene una carga de: **{1:.3f}**\n".format(session_state.datosRL.columns[i],float(session_state.modeloRL[0,i])))
                with st_stdout("info"):
                    print("La exactitud del modelo es: **{0:.2f}%**".format(float(session_state.exactitudRL)*100))

                st.success("¡Se ha copiado el modelo predictivo! Dirígete a la pestaña de prueba de modelo RL para insertar datos")
        else:
            st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos.")
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos.")
        #st.exception(e)

def pagina_prueba_modelo():
    st.title('Prueba del modelo de Clasificación RL')
    st.header("Modelo actual")
    st.write("Coeficientes:")

    try:
        with st_stdout("info"):
            for i in range(len(session_state.datosRL.columns)):
                print("{0:s} tiene una carga de: **{1:.3f}**\n".format(session_state.datosRL.columns[i],float(session_state.modeloRL[0,i])))
        with st_stdout("info"):
            print("La exactitud del modelo es: **{0:.2f}%**".format(float(session_state.exactitudRL)*100))

        botonesRL = []

        st.header("Prueba del modelo")
        st.write("Inserta los valores del paciente: ")

        id = st.text_input("ID del paciente",value="0", key="tx")

        for i in range(len(session_state.datosRL.columns)):
            botonesRL.append(st.text_input(session_state.datosRL.columns[i],value="",key="tx_"+str(i)))

        if(st.button("Predecir cáncer en paciente")):
            suma = 0
            for i in range(len(session_state.datosRL.columns)):
                suma = suma + (float(botonesRL[i])*float(session_state.modeloRL[0,i]))
            diagnosticoRL = 1/(1+math.e**(-(session_state.interceptoRL+suma)))

            if(diagnosticoRL<=0.5):
                with st_stdout("error"):
                    print("El paciente con ID **P-{0:s}** es diagnosticado con cáncer maligno".format(id))
            else:
                with st_stdout("success"):
                    print("El paciente con ID **P-{0:s}** es diagnosticado con cáncer benigno".format(id))
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dirígete a la pestaña de carga de datos. Si ya lo has hecho, dirígete a clasificación para generar el modelo de predicción")
        st.warning("Verifica que los datos insertados sean numéricos")
        #st.exception(e)

def main():

    st.set_page_config(
        page_title="PHMD",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )

    #"""Main function of the App"""
    st.sidebar.title("🟡 Barra de navegación")
    selection = st.sidebar.radio("🔷 Da clic en la función que te gustaría utilizar: ", PAGES)

    if(selection == "📖 Análisis exploratorio de datos"):
        pagina_analisisExploratorioDeDatos()

    elif(selection == "🤖 Inicio"):
        pagina_inicio()

    elif(selection == "💾 Carga de datos"):
        pagina_carga_datos()

    elif(selection == "📊 Análisis de componentes principales"):
        pagina_analisisComponentesPrincipales()

    elif(selection == "👪 Clustering particional"):
        pagina_clustering()

    elif(selection == "🔎 Clasificación"):
        pagina_clasificacion()

    elif(selection == "🔎 Prueba de modelo RL"):
        pagina_prueba_modelo()
        
if __name__ == "__main__":
    main()