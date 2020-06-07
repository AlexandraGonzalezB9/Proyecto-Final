# -*- coding: utf-8 -*-


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os;
import glob
import pywt
from linearFIR import filter_design, mfreqz
import scipy.signal as signal
import pandas as pd
import seaborn as sns #Graficacion

ruta_archivos = '/Users/yenifher/Desktop/semestre/bioseñales/8 mayo/example_data/'   #Ruta de la carpeta donde se encuentran los archivos a cargar

archivos_texto = glob.glob(ruta_archivos +'/*.txt',recursive = True) #El comando glob.glob permite clasificar en una lista la ruta de los archivos que terminen en txt
archivos_audio = glob.glob(ruta_archivos +'/*.wav',recursive = True) #El comando glob.glob permite clasificar en una lista la ruta de los archivos que terminen en wav

archivos=os.listdir(ruta_archivos) # permite enlistar  todos los archivos que se encuentran en esa ruta de archivos en una lista

def open_archivo(ruta_paciente_txt,ruta_paciente_audio):
    " Esta funcion recibe la ruta del archivo txt y del wav y retorna data que corresponde al archivo de texto, la señal de audio sin filtrar y su frecuencia de muestreo "
    data = np.loadtxt(ruta_paciente_txt) #carga la ruta del archivo txt
    senal, fm = librosa.load(ruta_paciente_audio) # por medio de librosa se carga el archivo de audio
    return data,senal,fm


def filtrar_senal(senal_f,fs):
    "Esta funcion recibe la señal sin filtar y la frecuencia de muestreo y retorna la señal filtrada"
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0); # implementación del filtro pasa bajas con filter_design
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);  # implementación del filtro pasa altas con filter_design
    senal_hp = signal.filtfilt(highpass,1,senal_f); # implementacion del diseño del filtro
    senal_lp = signal.filtfilt(lowpass,1,senal_hp);
    return senal_lp

def wthresh(coeff,thr):
    "funcion que correspondiente a el filtrado de wavelet"
    y   = list();
    s = wnoisest(coeff);
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
    return y;
    
def thselect(signal):
    "funcion que correspondiente a el filtrado de wavelet"
    Num_samples = 0;
    for i in range(0,len(signal)):
        Num_samples = Num_samples + signal[i].shape[0];
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wnoisest(coeff):
    "funcion que correspondiente a el filtrado de wavelet"
    stdc = np.zeros((len(coeff),1));
    for i in range(1,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
    return stdc;


def Preprocesamiento_senal(senal_P, fmo):
    "Esta funcion recibe la señal sin filtrar y la frecuencia de muestreo y hace tanto el filtrado lineal con el filtrado de wavelet retornando la señal completamente filtrada"
    
    #Aplicación de los filtros lineales a la señal sin ruido cardíaco
    senal_filt = filtrar_senal(senal_P,fmo) 

    # aplicación del Wavelet    
    LL = int(np.floor(np.log2(senal_filt.shape[0])))
    #
    coeff = pywt.wavedec(senal_filt, 'db36', level=LL );
    #
    thr = thselect(coeff);
    #    
    coeff_t = wthresh(coeff,thr);
    
    #Señal a con filtro wavelet
    senalf = pywt.waverec(coeff_t, 'db36');
    senalf= senalf[0:senal_filt.shape[0]];
      
    return senalf


def rango(ciclo):
    "Funcion que recibe un ciclo de la señal de un paciente y retorna el rango de este ciclo"
    maximo_ciclo = np.max(ciclo) # valor maximo
    minimo_ciclo = np.min(ciclo) # valor minimo
    rango_ciclo = maximo_ciclo-minimo_ciclo
    return rango_ciclo


def varianza(ciclo):
    "Funcion que recibe un ciclo de la señal de un paciente y retorna la varianza de este ciclo"
    varianza_ciclo = np.var(ciclo) #saca la varianza
    return varianza_ciclo

def Promedio_espectro(ciclo,fm):
    "Funcion que recibe un ciclo de la señal de un paciente y retorna el Promedio de espectro de este ciclo"
    f, Pxx = signal.welch(ciclo,fm) # aplicacion de welch
    Promedio_M = np.mean(Pxx)  # se saca el promedio de Pxx
    return Promedio_M

def Promedio_movil(ciclo):
    "Funcion que recibe un ciclo de la señal de un paciente y retorna el Promedio movil de este ciclo"
    cantidadM = 800 # cantidad de muestras
    corrimiento = 100 # el corrimiento que se va a realizar en el ciclo
    recorrido = np.arange(0,len(ciclo)-cantidadM,corrimiento) # creacion de vector recorrido
    promedioD = []
    for i in recorrido:
        promedioD.append(np.mean([ciclo[i:i+cantidadM]])) # se realiza el corrimiento entre las muestras de ese ciclo
    promedioD.append(np.mean([ciclo[recorrido[len(recorrido)-1]:]]))
    suma_promedios = np.mean(promedioD) # se saca el promedio
    return  suma_promedios

def Indices_estadistica(ciclo,fm):
    "Funcion que recibe un ciclo de la señal  y la frecuencia de muestreo, llamando algunas funciones antes creadas para sacar los valores de retorno varianza, rango, promedio de espectro y promedio movil "
    varianza_ciclo = varianza(ciclo) # se llama a la funcion varianza
    rango_ciclo = rango(ciclo) # se llama a la funcion rango
    promedio_e_ciclo = Promedio_espectro(ciclo,fm) # se llama a la funcion Promedio_espectro
    promedio_suma_movil = Promedio_movil(ciclo) # se llama a la funcion  Promedio_movil
    return varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil

# se crean diferentes listas para guardar los correspindientes datos

lista_ciclos = [] 

estado_GLOBAL = []
lista_sonido_crepitancia = [] 
lista_sonido_sibilancia = []
lista_sonido_sano = []
lista_sonido_CS = []


varianzaGLOBAL = []
varianza_de_ciclos_C =[] 
varianza_de_ciclos_S = []
varianza_de_ciclos_sano = []
varianza_de_ciclos_CS = []

rangoGLOBAL = []
rango_de_ciclos_C=[] 
rango_de_ciclos_S = []
rango_de_ciclos_sano =[]
rango_de_ciclos_CS = []


promedio_e_GLOBAL = []
promedio_espet_C= [] 
promedio_espect_S = []
promedio_espect_sano = []
promedio_espect_CS = []

Promedio_m_GLOBAL=[]
Promedio_SM_C = []
Promedio_SM_S = []
Promedio_SM_sano = []
Promedio_SM_CS = []


for i in range(0,len(archivos_audio)): # ciclo for que permite recorrer los 920 archivos de cada paciente
    dato,senal1,fm2= open_archivo(archivos_texto[i],archivos_audio[i]) # se llama a la funcion open_archivo
    senal_filtrada = Preprocesamiento_senal(senal1,fm2) # se llamam a la funcion Preprocesamiento_senal
    ciclo = dato.shape[0] # nos permite conocer la longitud de la columna de un archivo de texto
    j=0; # se inicializa ell contador 
    k=0;
    
    for j in range(0,ciclo): 
        muestra_inicial = np.int((dato[j,0])*fm2) ## cantidad de muestras en el tiempo inicial
        muestra_final = np.int((dato[j,1])*fm2) ## cantidad de muestras en el tiempo final 
        cicloi = senal_filtrada[muestra_inicial:muestra_final] # se divide la señal entre esos tiempos 
        lista_ciclos.append(cicloi) # se agraga las divisiones a una lista 
        ciclo_p = lista_ciclos[i] # se accede a los ciclos
        varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil =Indices_estadistica(ciclo_p,fm2) # se llama a la funcion Indices_estadistica
        varianzaGLOBAL.append(varianza_ciclo) # se agrega a la lista 
        rangoGLOBAL.append(rango_ciclo)
        promedio_e_GLOBAL.append(promedio_e_ciclo)
        Promedio_m_GLOBAL.append(promedio_suma_movil)
        
        
        if ((dato[j,2] == 1) and (dato[j,3] ==0)): # ingresa la columna 2 y 3 del archivo de texto que tenga 1 y 0 respectivamente
            lista_sonido_crepitancia.append('Crepitancia') # agregar el estado del ciclo a una lista 
            varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil =Indices_estadistica(ciclo_p,fm2) # se llama a la funcion Indices_estadistica
            varianza_de_ciclos_C.append(varianza_ciclo) # se agrega esa variable a una lista
            rango_de_ciclos_C.append(rango_ciclo)
            promedio_espet_C.append(promedio_e_ciclo)
            Promedio_SM_C.append(promedio_suma_movil)
            estado_GLOBAL.append('Crepitancia')  # agregar el estado del ciclo a una lista 
        elif ((dato[j,2] == 0) and (dato[j,3] ==1)):  # ingresa la columna 2 y 3 del archivo de texto que tenga 0 y 1 respectivamente
            lista_sonido_sibilancia.append('Sibilancia') # agregar el estado del ciclo a una lista 
            estado_GLOBAL.append('Sibilancia') # agregar el estado del ciclo a una lista 
            varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil =Indices_estadistica(ciclo_p,fm2) # se llama a la funcion Indices_estadistica
            varianza_de_ciclos_S.append(varianza_ciclo) # se agrega esa variable a una lista
            rango_de_ciclos_S.append(rango_ciclo)
            promedio_espect_S.append(promedio_e_ciclo)
            Promedio_SM_S.append(promedio_suma_movil)
        elif ((dato[j,2] == 0) and (dato[j,3] ==0)):
            lista_sonido_sano.append('Sano') # agregar el estado del ciclo a una lista 
            estado_GLOBAL.append('Sano') # agregar el estado del ciclo a una lista 
            varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil =Indices_estadistica(ciclo_p,fm2) # se llama a la funcion Indices_estadistica
            varianza_de_ciclos_sano.append(varianza_ciclo) # se agrega esa variable a una lista
            rango_de_ciclos_sano.append(rango_ciclo)
            promedio_espect_sano.append(promedio_e_ciclo)
            Promedio_SM_sano.append(promedio_suma_movil)
        else: 
            lista_sonido_CS.append('Crepitancia-Sibilancia')# agregar el estado del ciclo a una lista 
            estado_GLOBAL.append('Crepitancia-Sibilancia') # agregar el estado del ciclo a una lista 
            varianza_ciclo,rango_ciclo,promedio_e_ciclo,promedio_suma_movil =Indices_estadistica(ciclo_p,fm2) # se llama a la funcion Indices_estadistica
            varianza_de_ciclos_CS.append(varianza_ciclo) # se agrega esa variable a una lista
            rango_de_ciclos_CS.append(rango_ciclo)
            promedio_espect_CS.append(promedio_e_ciclo)
            Promedio_SM_CS.append(promedio_suma_movil)
                             
    i=i+1
    
# DATAFRAME PARA SANOS        
dfsano = pd.DataFrame({'Estado':lista_sonido_sano,'Rango':rango_de_ciclos_sano,'Varianza':varianza_de_ciclos_sano,'Promedio de espectro':promedio_espect_sano,'Promedio sumatoria movil':Promedio_SM_sano})    

# DATAFRAME PARA CREPITANCIAS
dfcrepitancia = pd.DataFrame({'Estado':lista_sonido_crepitancia,'Rango':rango_de_ciclos_C,'Varianza':varianza_de_ciclos_C,'Promedio de espectro':promedio_espet_C,'Promedio sumatoria movil':Promedio_SM_C}) 

#DATAFRAME PARA SIBILANCIA
dfsibilancia = pd.DataFrame({'Estado':lista_sonido_sibilancia,'Rango':rango_de_ciclos_S ,'Varianza':varianza_de_ciclos_S,'Promedio de espectro':promedio_espect_S,'Promedio sumatoria movil':Promedio_SM_S})    

#DATAFRAME PARA CREPITANCIA Y SIBILANCIA
dfCS = pd.DataFrame({'Estado':lista_sonido_CS,'Rango':rango_de_ciclos_CS ,'Varianza':varianza_de_ciclos_CS,'Promedio de espectro':promedio_espect_CS,'Promedio sumatoria movil':Promedio_SM_CS })

#DATAFRAME GLOBAL DE TODOS LOS DATOS
dfGLOBAL = pd.DataFrame({'Estado':estado_GLOBAL,'Rango':rangoGLOBAL ,'Varianza':varianzaGLOBAL,'Promedio de espectro':promedio_e_GLOBAL,'Promedio sumatoria movil':Promedio_m_GLOBAL })


#Estadística descriptiva para:
#Sanos
estadisticaDES_sano = dfsano.describe() # esa funcion retorna una tabla con las medidas estadisticas de esa matriz 
#crepitancias
estadisticaDES_crepitancia = dfcrepitancia.describe() # esa funcion retorna una tabla con las medidas estadisticas de esa matriz 
#sibilancias
estadisticaDES_sibilancia = dfsibilancia.describe() # esa funcion retorna una tabla con las medidas estadisticas de esa matriz 
#crepitancias y sibilancias
estadisticaDES_CS = dfCS.describe() # esa funcion retorna una tabla con las medidas estadisticas de esa matriz 


# Box plots de estado y rango a nivel de todos los pacientes

plt.figure()
sns.boxplot(x='Estado',y='Rango',data=dfGLOBAL)
plt.title('Box plots')
plt.plot()
plt.show()

# Box plots de estado y varianza a nivel de todos los pacientes
plt.figure()
sns.boxplot(x='Estado',y='Varianza',data=dfGLOBAL)
plt.title('Box plots')
plt.plot()
plt.show()

# Box plots de estado y  promedio de espectros a nivel de todos los pacientes
plt.figure()
sns.boxplot(x='Estado',y='Promedio de espectro',data=dfGLOBAL)
plt.title('Box plots')
plt.plot()
plt.show()

# Box plots de estado y  promedio de espectros a nivel de todos los pacientes
plt.figure()
sns.boxplot(x='Estado',y='Promedio sumatoria movil',data=dfGLOBAL)
plt.title('Box plots')
plt.plot()
plt.show()


#GRÁFICOS DE DISPERSIÓN

#Grafico de dispersión para el rango en los diferentes estados
plt.scatter(dfGLOBAL['Estado'],dfGLOBAL['Rango'])
plt.title('Gráficos de dispersión')
plt.ylabel('Rango')
plt.xlabel('Estado')
plt.show()


#Grafico de dispersión para la varianza en los diferentes estados
plt.figure()
plt.scatter(dfGLOBAL['Estado'],dfGLOBAL['Varianza'])
plt.title('Gráficos de dispersión')
plt.ylabel('Varianza')
plt.xlabel('Estado')
plt.show()


#Grafico de dispersión para el promedio de espectros en los diferentes estados
plt.figure()
plt.scatter(dfGLOBAL['Estado'],dfGLOBAL['Promedio de espectro'])
plt.title('Gráficos de dispersión')
plt.ylabel('Promedio de espectro')
plt.xlabel('Estado')
plt.show()


#Grafico de dispersión para el promedio movil en los diferentes estados
plt.figure()
plt.scatter(dfGLOBAL['Estado'],dfGLOBAL['Promedio sumatoria movil'])
plt.title('Gráficos de dispersión')
plt.ylabel('Promedio sumatoria móvil')
plt.xlabel('Estado')
plt.show()

#Histograma del rango para ciclos sanas
count,bin_edges = np.histogram(dfsano['Rango'])
dfsano['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del rango para los ciclos sanos')
plt.xlabel('Rango')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma de la varianza para ciclos sanas
count,bin_edges = np.histogram(dfsano['Varianza'])
dfsano['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma de la varianza para los ciclos sanos')
plt.xlabel('Varianza')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de espectro para ciclos sanas
count,bin_edges = np.histogram(dfsano['Promedio de espectro'])
dfsano['Promedio de espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de espectro para los ciclos sanos')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de sumatoria móvil para ciclos sanas
count,bin_edges = np.histogram(dfsano['Promedio sumatoria movil'])
dfsano['Promedio sumatoria movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de sumatoria móvil para los ciclos sanos')
plt.xlabel('Promedio sumatoria móvil')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del rango para ciclos con crepitaciones
count,bin_edges = np.histogram(dfcrepitancia['Rango'])
dfcrepitancia ['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del rango para ciclos con crepitaciones')
plt.xlabel('Rango')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma de varianza para ciclos con crepitaciones
count,bin_edges = np.histogram(dfcrepitancia['Varianza'])
dfcrepitancia['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma de la varianza para ciclos con crepitaciones')
plt.xlabel('Varianza')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de espectro para ciclos con crepitaciones
count,bin_edges = np.histogram(dfcrepitancia ['Promedio de espectro'])
dfcrepitancia ['Promedio de espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de espectro para ciclos con crepitaciones')
plt.xlabel('Promedio de espectro')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de sumatoria móvil para ciclos con crepitaciones
count,bin_edges = np.histogram(dfcrepitancia['Promedio sumatoria movil'])
dfcrepitancia['Promedio sumatoria movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de sumatoria móvil para ciclos con crepitaciones')
plt.xlabel('Promedio sumatoria móvil')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del rango para ciclos con sibilancia
count,bin_edges = np.histogram(dfsibilancia ['Rango'])
dfsibilancia ['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del rango para ciclos con sibilancias')
plt.xlabel('Rango')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma de varianza para ciclos con sibilancias
count,bin_edges = np.histogram(dfsibilancia ['Varianza'])
dfsibilancia ['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma de la varianza para ciclos con sibilancias')
plt.xlabel('Varianza')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de espectro para ciclos con sibilancias
count,bin_edges = np.histogram(dfsibilancia  ['Promedio de espectro'])
dfsibilancia  ['Promedio de espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de espectro para ciclos con sibilancias')
plt.xlabel('Promedio de espectro')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de sumatoria móvil para ciclos con sibilancias
count,bin_edges = np.histogram(dfsibilancia ['Promedio sumatoria movil'])
dfsibilancia ['Promedio sumatoria movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de sumatoria móvil para ciclos con sibilancias')
plt.xlabel('Promedio sumatoria móvil')
plt.ylabel('Cantidad')
plt.grid()
plt.show()


#Histograma del rango para ciclos con crepitaciones-sibilancias
count,bin_edges = np.histogram(dfCS['Rango'])
dfCS ['Rango'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del rango para ciclos con crepitaciones-sibilancias')
plt.xlabel('Rango')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma de varianza para ciclos con crepitaciones-sibilancias
count,bin_edges = np.histogram(dfCS['Varianza'])
dfCS['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma de la varianza para crepitaciones-sibilancias')
plt.xlabel('Varianza')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de espectro para ciclos con crepitaciones-sibilancias
count,bin_edges = np.histogram(dfCS ['Promedio de espectro'])
dfCS ['Promedio de espectro'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de espectro para crepitaciones-sibilancias')
plt.xlabel('Promedio de espectro')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

#Histograma del promedio de sumatoria móvil para ciclos con crepitaciones-sibilancias
count,bin_edges = np.histogram(dfCS['Promedio sumatoria movil'])
dfCS['Promedio sumatoria movil'].plot(kind='hist',xticks=bin_edges)
plt.title('Histograma del promedio de sumatoria móvil para crepitaciones-sibilancias')
plt.xlabel('Promedio sumatoria móvil')
plt.ylabel('Cantidad')
plt.grid()
plt.show()

# CORRELACIÓN PARA LOS CICLOS SANOS
correlation_matrix_sanos = dfsano.corr()
sns.heatmap(correlation_matrix_sanos, annot=True)
plt.title('Correlación para los ciclos sanos')
plt.show()

# CORRELACIÓN PARA LOS CICLOS CON CREPITACIONES
correlation_matrix_crepitaciones = dfcrepitancia.corr()
sns.heatmap(correlation_matrix_crepitaciones, annot=True)
plt.title('Correlación para los ciclos con crepitaciones')
plt.show()

# CORRELACIÓN PARA LOS CICLOS CON SIBILANCIAS
correlation_matrix_sibilancias = dfsibilancia.corr()
sns.heatmap(correlation_matrix_sibilancias , annot=True)
plt.title('Correlación para los ciclos con sibilancias')
plt.show()

# CORRELACIÓN PARA LOS CICLOS CON CREPITANCIAS- SIBILANCIAS
correlation_matrix_CS = dfCS.corr()
sns.heatmap(correlation_matrix_CS  , annot=True)
plt.title('Correlación para los ciclos con crepitaciones y sibilancias')
plt.show()

# CORRELACIÓN PARA LOS CICLOS CON SIBILANCIAS
correlation_matrix_global= dfGLOBAL.corr()
sns.heatmap(correlation_matrix_sibilancias , annot=True)
plt.title('Correlación para los ciclos con todos los ciclos')
plt.show()

# PRUEBA  DE HIPÓTESIS

import scipy.stats as stats

print('Prueba U de Mann-Whitney para los rangos entre ciclos sanos y con crepitancias')
print(stats.mannwhitneyu(dfsano['Rango'] ,dfsano['Rango']))


print('Prueba U de Mann-Whitney para los rangos entre ciclos sanos y con sibilancias')
print(stats.mannwhitneyu(dfsano['Rango'] ,dfsibilancia['Rango']))

print('Prueba U de Mann-Whitney para los rangos entre ciclos con crepitaciones y sibilancias')
print(stats.mannwhitneyu(dfcrepitancia['Rango'] ,dfsibilancia['Rango']))


# Varianza 
print('Prueba U de Mann-Whitney para la varianza entre ciclos sanos y con crepitancias')
print(stats.mannwhitneyu(['Varianza'] ,dfcrepitancia['Varianza']))

print('Prueba U de Mann-Whitney para la varianza entre ciclos sanos y con sibilancias')
print(stats.mannwhitneyu(['Varianza'] ,dfsibilancia['Varianza']))

print('Prueba U de Mann-Whitney para la varianza  entre ciclos con crepitaciones y sibilancias')
print(stats.mannwhitneyu(dfcrepitancia['Varianza'] ,dfsibilancia['Varianza']))

#Promedio de espectro'

print('Prueba U de Mann-Whitney para el promedio de espectro entre ciclos sanos y con crepitancias')
print(stats.mannwhitneyu(dfsano['Promedio de espectro'] ,dfcrepitancia['Promedio de espectro']))

print('Prueba U de Mann-Whitney para el promedio de espectro entre ciclos sanos y con sibilancias')
print(stats.mannwhitneyu(dfsano['Promedio de espectro'] ,dfsibilancia['Promedio de espectro']))

print('Prueba U de Mann-Whitney para el promedio de espectro entre ciclos con crepitaciones y sibilancias')
print(stats.mannwhitneyu(dfcrepitancia['Promedio de espectro'] ,dfsibilancia['Promedio de espectro']))

print('Prueba U de Mann-Whitney para la sumatoria movíl entre ciclos sanos y con crepitancias')
print(stats.mannwhitneyu(dfsano['Promedio sumatoria movil'] ,dfcrepitancia['Promedio sumatoria movil']))

print('Prueba U de Mann-Whitney para la sumatoria movíl entre ciclos sanos y con sibilancias')
print(stats.mannwhitneyu(dfsano['Promedio sumatoria movil'] ,dfsibilancia['Promedio sumatoria movil']))


print('Prueba U de Mann-Whitney para la sumatoria movíl entre ciclos con crepitaciones y sibilancias')
print(stats.mannwhitneyu(dfcrepitancia['Promedio sumatoria movil'] ,dfsibilancia['Promedio sumatoria movil']))



