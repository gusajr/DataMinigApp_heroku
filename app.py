# '''
#     Aplicaci贸n de Miner铆a de datos
#     Gustavo Alfredo Jim茅nez Ruiz
#     09 de agosto del 2021
# '''

# '''
#     Importaci贸n de bibliotecas y funciones
# '''

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

st.set_page_config(
        page_title="PHMD",
        page_icon="馃",
        layout="centered",
        initial_sidebar_state="auto"
    )

PAGES = (
    "馃 Inicio",
    "馃捑 Carga de datos",
    "馃摉 An谩lisis exploratorio de datos",
    "馃搳 An谩lisis de componentes principales",
    "馃應 Clustering particional",
    "馃攷 Clasificaci贸n RL",
    "馃攷 Prueba de modelo RL"
)


# '''
#     Variables de la sesi贸n
# '''
global session_state
session_state = SessionState.get(datosEDA = 0, datos = None, valor_reemplazo = 0, widgetKey = "new.1")
session_state_1 = SessionState.get(datosPCA = 0, clustering_sns_pairplot = None)
session_state_1.clustering_sns_pairplot = None

# '''
#     Funci贸n principal de la aplicaci贸n
# '''
def main():

    st.sidebar.title("馃煛 Barra de navegaci贸n")
    selection = st.sidebar.radio("馃敺 Da clic en la funci贸n que te gustar铆a utilizar: ", PAGES)

    if(selection == "馃摉 An谩lisis exploratorio de datos"):
        pagina_analisisExploratorioDeDatos()

    elif(selection == "馃 Inicio"):
        pagina_inicio()

    elif(selection == "馃捑 Carga de datos"):
        pagina_carga_datos()

    elif(selection == "馃搳 An谩lisis de componentes principales"):
        pagina_analisisComponentesPrincipales()

    elif(selection == "馃應 Clustering particional"):
        pagina_clustering()

    elif(selection == "馃攷 Clasificaci贸n RL"):
        pagina_clasificacion()

    elif(selection == "馃攷 Prueba de modelo RL"):
        pagina_prueba_modelo()


# '''
#     P谩gina que muestra el inicio de la aplicaci贸n
# '''
def pagina_inicio():

    st.title('Bienvenid@ a mi Peque帽a Herramienta de Miner铆a de Datos (PHMD)')

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.0005)
        my_bar.progress(percent_complete + 1)

    chart_data = pd.DataFrame(
        np.random.randn(10, 3),
        columns=['a', 'b', 'c'])
    st.line_chart(chart_data)
    

    st.subheader("Esta herramienta fue creada con 鉂わ笍 por Gustavo Jim茅nez, alumno de la Facultad de Ingenier铆a de la Universidad Nacional Aut贸noma de M茅xico.")
    st.write("_Para comenzar el an谩lisis de datos puedes dar clic en alguna de las opciones de la barra de navegaci贸n._")


# '''
#     P谩gina que muestra la carga de datos
# '''
def pagina_carga_datos():

    st.title('Carga y modificaci贸n de datos')
    st.header("Importaci贸n de datos")
    st.write("1. Lectura de datos")

    if(st.radio("驴Contiene Header?",("Si","No"),key="1")=="No"):
        datos_header = None
    else:
        datos_header = 0

    datos_sep = st.text_input("Separaci贸n de los datos: ", value=",")
    datos_subido = st.file_uploader("Escoge el archivo que quieres analizar: ", type = ['txt', 'csv', 'xls'],key=session_state.widgetKey)

    if(datos_subido is not None):
        with st.spinner('Procesando datos...'):
            if(datos_subido.name.split(".")[1] == "txt" or datos_subido.name.split(".")[1] == "csv"):
                datos_csv = pd.read_csv(datos_subido, header=datos_header, sep=datos_sep)
            if(datos_subido.name.split(".")[1] == "xls"):
                datos_csv = pd.read_excel(datos_subido)
            session_state.datos_iniciales = pd.DataFrame(datos_csv)
            session_state.datos_iniciales_0 = session_state.datos_iniciales
        st.success('隆Hecho!')
        session_state.widgetKey=session_state.widgetKey.split(".")[0]+"."+str(int(session_state.widgetKey.split(".")[1])+1)
        session_state.datos = session_state.datos_iniciales
    
    if(session_state.datos is None):
        st.error("Por favor, sube un archivo")

    try:

        datos_sep = 0
        datos_subido = None
        #datos = 0
        datos_csv = 0

        columnas = st.multiselect(
            "Escoge las columnas que utilizar谩s para los algoritmos: ", list(session_state.datos_iniciales_0.columns), list(session_state.datos.columns), key="ms_datos_1"
        )
        if not columnas:
            st.error("Por favor escoge al menos una columna a analizar")
        
        st.write("Datos iniciales")
        st.write(session_state.datos_iniciales_0[columnas])

        st.success("隆Ya puedes continuar con otra opci贸n!")

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
                st.warning("Ning煤n valor insertado. Al dar enter se reemplazar谩 el valor")
            else:
                valores = valorSustitucionClasificacionReemplazar.split(",")
                if "," not in valorSustitucionClasificacionReemplazar:
                    st.error("Por favor, inserta una coma entre valor a reemplazar y valor nuevo")
                else:
                        #session_state.datos_iniciales = session_state.datos_iniciales[columnas]
                    session_state.datos[columnaClasificacion] = session_state.datos[columnaClasificacion].replace(valores[0], valores[1])
                    session_state.datos_iniciales[columnaClasificacion] = session_state.datos[columnaClasificacion].replace(valores[0], valores[1])
                    st.warning("Al dar enter se reemplazar谩 el valor")
                    st.success("隆Datos modificados!")
        st.write("Datos le铆dos con datos modificados, **estos datos ser谩n los que se usen en las dem谩s pesta帽as**")
        st.write(session_state.datos)
                    
    except Exception as e:
        st.write("Sin datos.")
        #st.exception(e)

# '''
#     P谩gina que muestra los pasos para el An谩lisis Exploratorio de Datos
# '''
def pagina_analisisExploratorioDeDatos():

    st.title('An谩lisis Exploratorio de Datos (EDA)')

    datosEDA = session_state.datos

    if(datosEDA is not None):
        with st.spinner('Recargando datos...'):
            st.header("Datos")
            st.write("1. Visualizaci贸n de datos")
            st.write("Datos le铆dos")
            st.write(datosEDA)
        st.success('隆Hecho!')
        st.warning("Si necesitas editar alguna columna de tu dataset, dir铆gete a la opci贸n de carga de datos")

    try:
        VariableValoresAtipicos = st.multiselect(
                "Escoge las columnas de tu elecci贸n para visualizar valores at铆picos: ", 
                pd.DataFrame(datosEDA).select_dtypes(include=np.number).columns.tolist(), 
                pd.DataFrame(datosEDA).select_dtypes(include=np.number).columns.tolist()
            )
        if not VariableValoresAtipicos:
            st.error("Por favor escoge al menos una columna a analizar para valores at铆picos. Algunas columnas no se muestran por no ser num茅ricas")
        else:
            if st.button("Iniciar ejecuci贸n",key="b_1"):
                st.header("Descripci贸n de la estructura de los datos")
                st.write("1. Dimensiones de la data")
                st.write("Renglones:")
                datosEDA.shape[0]
                st.write("Columnas:")
                datosEDA.shape[1]
                st.write("2. Tipos de dato de las variables")
                datosEDA.dtypes

                st.header("Identificaci贸n de datos faltantes")
                buffer = io.StringIO() 
                datosEDA.info(buf=buffer)
                info = buffer.getvalue()
                st.text(info)

                st.header("Detecci贸n de valores at铆picos")
                st.write("1. Distribuci贸n de variables num茅ricas")
                datosEDA.hist(figsize=(14,14), xrot=45)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

                st.write("2. Resumen estad铆stico de variables num茅ricas")
                st.write(datosEDA.describe())

                st.write("3. Diagramas para detectar posibles valores at铆picos")
                with st.spinner('Procesando diagrama de caja...'):
                    for col in VariableValoresAtipicos:
                        with st_stdout("info"):
                            print("Diagrama de cajas y bigote para "+str(col))
                        sns.boxplot(col, data=datosEDA)
                        plt.show()
                        st.pyplot()
                st.success('隆Hecho!')
                
                try:
                    st.write("4. Distribuci贸n de variables categ贸ricas")
                    st.write(datosEDA.describe(include='object'))
                    
                    with st.spinner('Procesando histograma...'):
                        for col in datosEDA.select_dtypes(include='object'):
                            if datosEDA[col].nunique()<10:
                                st.write((datosEDA.groupby(col).agg(['mean'])))
                                sns.countplot(y=col, data=datosEDA)
                                plt.show()
                                st.pyplot()
                    st.success('隆Hecho!')

                    # for col in datosEDA.select_dtypes(include='object'):
                    #     if datosEDA[col].nunique() < 10:
                    #         st.write((datosEDA.groupby(col).agg(['mean'])))
                
                except Exception as e:
                    st.warning("Algunos datos no se muestran por el tipo de datos en el dataset")
                    ##st.exception(e)

                st.header("Identificaci贸n de relaciones entre pares de variables")
                st.write("1. Matriz de correlaciones")
                st.write(datosEDA.corr())
                st.write("2. Mapa de calor de la matriz de correlaciones")
                plt.figure(figsize=(14,14))
                sns.heatmap(datosEDA.corr(), cmap='RdBu_r', annot=True)
                plt.show()
                st.pyplot()
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos")
        #st.exception(e)

# '''
#     P谩gina que muestra los pasos para el An谩lisis de las Componentes Principales
# '''
def pagina_analisisComponentesPrincipales():
    st.title('An谩lisis de Componentes Principales (PCA)')

    datosPCA = session_state.datos

    if(datosPCA is not None):
        with st.spinner('Recargando datos...'):
            st.header("Datos")
            st.write("1. Visualizaci贸n de datos")
            st.write("Datos le铆dos")
            st.write(datosPCA)
        st.success('隆Hecho!')
        st.warning("Si necesitas editar alguna columna de tu dataset, dir铆gete a la opci贸n de carga de datos")
        st.warning("Este algoritmo funciona con datos num茅ricos")

        if st.button("Iniciar ejecuci贸n"):

            datosPCA = pd.DataFrame(datosPCA).select_dtypes(include=np.number).replace(np.NaN,0)
            st.header("Estandarizaci贸n de los datos")
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

            st.header("Elecci贸n del n煤mero de componentes principales (eigen-vectores)")
            Varianza = Componentes.explained_variance_ratio_
            st.write("Eigenvalues:",Varianza)
            st.write("Varianza acumulada: ",sum(Varianza[0:5]))

            plt.plot(np.cumsum(Componentes.explained_variance_ratio_))
            plt.xlabel("N煤mero de componentes")
            plt.ylabel("Varianza acumulada")
            plt.grid()
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.header("An谩lisis de proporci贸n de relevancias (cargas)")
            st.write("Se revisan los valores absolutos de los componentes principales seleccionados. Cuanto mayor sea el valor absoluto, m谩s importante es esa variable en el componente principal.")
            st.write(pd.DataFrame(abs(Componentes.components_),columns=datosPCA.columns))

    else:
        st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos")
        #st.exception(e)

# '''
#     P谩gina que muestra el algoritmo de Clustering
# '''
def pagina_clustering():
    st.title('Clustering particional')
    try:
        if(session_state.datos is not None):
            st.header("Datos")
            st.write("1. Visualizaci贸n de datos")
            with st.spinner("Recargando datos"):
                session_state.datos
            st.success("隆Hecho!")
            st.warning("Si necesitas editar alguna columna de tu dataset, dir铆gete a la opci贸n de carga de datos")
            st.warning("Este algoritmo funciona con datos num茅ricos")
            st.header("Selecci贸n de caracter铆sticas")
            
            st.write("Comparaci贸n de relaci贸n entre variables")
            
        session_state.numericos = pd.DataFrame(session_state.datos).select_dtypes(include=np.number).replace(np.NaN,0)
        columna0 = st.selectbox("Selecciona la columna principal a comparar para los ejes: ", session_state.numericos.columns, key="15")
        clustering_x = st.selectbox("Selecciona la primer variable:", session_state.numericos.columns, key="4")
        clustering_y = st.selectbox("Selecciona la segunda variable:", session_state.numericos.columns, key="5")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.scatterplot(x=clustering_x,y=clustering_y,data=session_state.numericos,hue=columna0)
        plt.title("Gr谩fico de dispersi贸n")
        plt.xlabel(clustering_x)
        plt.ylabel(clustering_y)
        plt.show()
        st.pyplot()

        columna = 0
        columna_tmp = columna
        columna = st.selectbox("Selecciona la columna principal a comparar para generar la matriz de correlaciones: ", session_state.numericos.columns, key="6")
        
        if(st.button("Iniciar ejecuci贸n", key="bt_clustering")):
            with st.spinner('Procesando matriz...'):
                if(session_state_1.clustering_sns_pairplot == None  or columna_tmp != columna):
                    clustering_sns_pairplot = sns.pairplot(session_state.numericos, hue=columna)
                    session_state_1.clustering_sns_pairplot = clustering_sns_pairplot
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            st.success('隆Hecho!')

            st.write("Matriz de correlaciones")
            datosPCA_corr = session_state.numericos.corr(method="pearson")
            datosPCA_corr
            st.write("Mapa de calor")
            plt.figure(figsize=(14,7))
            MatrizInf = np.triu(datosPCA_corr)
            sns.heatmap(datosPCA_corr, cmap="RdBu_r", annot=True, mask=MatrizInf)
            plt.show()
            st.pyplot()

            st.header("Variables utilizadas en el an谩lisis")
            columnasPCA = st.multiselect(
                "Se enlistan las columnas que se utilizan:", list(session_state.numericos.columns), list(session_state.numericos.columns)
            )
            if not columnasPCA:
                st.error("Por favor escoge al menos una columna a analizar")
            else:
                datosPCA = session_state.numericos[columnasPCA]

            datosPCA
            st.warning("Recuerda que si necesitas modificar tus datos debes dirigirte a la pesta帽a de carga de datos")

            st.header("Algoritmo K-means")
            SSE=[]
            for i in range(2,12):
                km=KMeans(n_clusters=i,random_state=0)
                km.fit(datosPCA)
                SSE.append(km.inertia_)

            plt.figure(figsize=(10,7))
            plt.plot(range(2,12),SSE,marker="o")
            plt.xlabel("Cantidad de cl煤sters *k*")
            plt.ylabel("SSE")
            plt.title("M茅todo del codo")
            plt.show()
            st.pyplot()

            kl = KneeLocator(range(2,12),SSE,curve="convex",direction="decreasing")
            st.write("N煤mero de cl煤sters con mayor convergencia")
            st.text(kl.elbow)

            st.write("Cl煤ster al que pertenece cada registro")
            MParticional = KMeans(n_clusters=4, random_state=0).fit(session_state.numericos)
            MParticional.predict(session_state.numericos)
            MParticional.labels_

            st.write("Se a帽ade la columna de n煤mero de cl煤ster de cada registro")
            datosPCA["clusterP"] = MParticional.labels_
            datosPCA

            st.write("N煤mero de registros en cada cl煤ster")
            st.write(datosPCA.groupby(["clusterP"])["clusterP"].count())

            st.write("Gr谩fica para mostrar cl煤steres por color")
            plt.figure(figsize=(10,7))
            plt.scatter(datosPCA.iloc[:,0], datosPCA.iloc[:,1],c=MParticional.labels_,cmap="rainbow")
            plt.show()
            st.pyplot()

            st.write("Media de los registros en cada cl煤ster")
            CentroidesP = MParticional.cluster_centers_
            st.write(pd.DataFrame(CentroidesP.round(4),columns=datosPCA.columns[0:len(datosPCA.columns)-1]))

            plt.rcParams["figure.figsize"] = (10,7)
            plt.style.use("ggplot")
            colores=["red","blue","green","yellow"]
            asignar=[]
            for row in MParticional.labels_:
                asignar.append(colores[row])

            st.write("Gr谩fica 3D de la clasificaci贸n de los registros")

            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(datosPCA.iloc[:,0],datosPCA.iloc[:,1],datosPCA.iloc[:,2],marker="o",c=asignar,s=60)
            ax.scatter(CentroidesP[:,0],CentroidesP[:,1],CentroidesP[:,2],marker="*",c=colores,s=1000)
            plt.show()
            st.pyplot()

            st.write("Registros m谩s cercanos al centroide")
            Cercanos,_ = pairwise_distances_argmin_min(MParticional.cluster_centers_, session_state.numericos)
            Cercanos

    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos.")
        #st.exception(e)

# '''
#     P谩gina que muestra el algoritmo de Clasificaci贸n
# '''
def pagina_clasificacion():

    datosClasificacion = session_state.datos
    st.title('Clasificaci贸n: Regresi贸n Log铆stica')

    try:
        if(datosClasificacion is not None):
            with st.spinner('Recargando datos...'):
                st.header("Datos")
                st.write("1. Visualizaci贸n de datos")
                st.write("Datos le铆dos")
                st.write(datosClasificacion)
            st.success('隆Hecho!')

            st.warning("Si necesitas editar alguna columna de tu dataset, dir铆gete a la opci贸n de carga de datos")

            st.header("Variable predictora")
            columnaClasificacion = st.selectbox(
                "Escoge la variable predictora: ", list(session_state.datos.columns), index=0
            )
            st.write(datosClasificacion.groupby(columnaClasificacion).size())

            st.warning("Recuerda que el algoritmo funciona con valores 1-0, es necesario que hagas estos ajustes desde la pesta帽a de carga de datos")
            
            st.header("Selecci贸n de caracter铆sticas")
            
            st.write("Comparaci贸n de relaci贸n entre variables")

            st.set_option('deprecation.showPyplotGlobalUse', False)
            clustering_x = st.selectbox("Selecciona la primer variable:", session_state.datos.columns, key="12")
            clustering_y = st.selectbox("Selecciona la segunda variable:", session_state.datos.columns, key="13")
            sns.scatterplot(x=clustering_x,y=clustering_y,data=session_state.datos,hue=columnaClasificacion)
            plt.title("Gr谩fico de dispersi贸n")
            plt.xlabel(clustering_x)
            plt.ylabel(clustering_y)
            plt.show()
            st.pyplot()

            if(st.button("Iniciar ejecuci贸n",key="b_2")):

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
                
                st.warning("Si tienes que realizar eliminaci贸n de columnas, recuerda hacerlo en la pesta帽a de carga de datos")
                X=session_state.datos.drop(columnaClasificacion, axis=1)
                Y=session_state.datos[columnaClasificacion]

                st.header("Aplicaci贸n del algoritmo")
                Clasificacion = linear_model.LogisticRegression()
                seed=1234
                X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=0.2,random_state=seed,shuffle=True)

                pd.DataFrame(X_train)
                pd.DataFrame(Y_train)
                
                st.write("Entrenamiento del modelo a partir de los datos de entrada")
                st.write(Clasificacion.fit(X_train,Y_train))

                st.write("Predicciones probabil铆sticas")
                Probabilidad = Clasificacion.predict_proba(X_train)
                st.write(pd.DataFrame(Probabilidad))

                st.write("Predicciones con clasificaci贸n final")
                Predicciones = Clasificacion.predict(X_train)
                st.write(pd.DataFrame(Predicciones))

                exactitud = Clasificacion.score(X_train,Y_train)

                st.header("Validaci贸n del modelo")

                st.write("Matriz de clasificaci贸n")
                PrediccionesNuevas = Clasificacion.predict(X_validation)
                confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, rownames=["Real"], colnames=["Clasificaci贸n"])
                confusion_matrix

                st.write("Reporte de la clasificaci贸n")
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

                VP=confusion_matrix.loc[0,0]
                FN=confusion_matrix.loc[0,1]
                FP=confusion_matrix.loc[1,0]
                VN=confusion_matrix.loc[1,1]

                precision= VP/(VP+FP)
                tasaError=(FP+FN)/(VP+VN+FP+FN)
                sensibilidad=VP/(VP+FN)
                especificidad= VN/(VN+FP)

                with st_stdout("info"):
                    for i in range(len(session_state.datosRL.columns)):
                        print("{0:s} tiene una carga de: **{1:.3f}**\n".format(session_state.datosRL.columns[i],float(session_state.modeloRL[0,i])))
                with st_stdout("info"):
                    print("La exactitud del modelo es: **{0:.2f}%**\n".format(float(session_state.exactitudRL)*100))
                    print("La precisi贸n del modelo es: **{0:.2f}%**\n".format(precision*100))
                    print("La tasa de error del modelo es: **{0:.2f}%**\n".format(tasaError*100))
                    print("La sensibilidad del modelo es: **{0:.2f}%**\n".format(sensibilidad*100))
                    print("La especificidad del modelo es: **{0:.2f}%**\n".format(especificidad*100))

                st.success("隆Se ha copiado el modelo predictivo! Dir铆gete a la pesta帽a de prueba de modelo RL para insertar datos")
        else:
            st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos.")
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos.")
        #st.exception(e)

# '''
#     P谩gina que muestra el modelo de Clasificaci贸n RL
# '''
def pagina_prueba_modelo():
    st.title('Prueba del modelo de Clasificaci贸n RL')
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

        if(st.button("Predecir c谩ncer en paciente")):
            suma = 0
            for i in range(len(session_state.datosRL.columns)):
                suma = suma + (float(botonesRL[i])*float(session_state.modeloRL[0,i]))
            diagnosticoRL = 1/(1+math.e**(-(session_state.interceptoRL+suma)))

            if(diagnosticoRL<=0.5):
                with st_stdout("error"):
                    print("El paciente con ID **P-{0:s}** es diagnosticado con c谩ncer maligno".format(id))
            else:
                with st_stdout("success"):
                    print("El paciente con ID **P-{0:s}** es diagnosticado con c谩ncer benigno".format(id))
    except Exception as e:
        st.error("Datos no cargados o incompatibles, por favor dir铆gete a la pesta帽a de carga de datos. Si ya lo has hecho, dir铆gete a clasificaci贸n para generar el modelo de predicci贸n")
        st.warning("Verifica que los datos insertados sean num茅ricos")
        #st.exception(e)
        
if __name__ == "__main__":
    main()