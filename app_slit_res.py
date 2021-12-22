# -*- coding: utf-8 -*-
"""
José Carlos Machicao
GestioDinámica
Fecha de producción: 2021_08_28
Fuentes: 
    http://localhost:8503/
    https://builtin.com/machine-learning/streamlit-tutorial

"""

import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import base64

c1, c2 = st.columns([5,5])
with c1:
    st.image('gdmk.png', width=200, caption='www.gestiodinamica.com')

st.title('Identificación de Personas')

st.write('**Procedimiento: Generación de Identidad Digital Base**')

up_files = st.file_uploader('Elija archivos base: ', accept_multiple_files=True)

l_pics = len(up_files)

def genera_imagenes_base(up_files):
    ims = []
    lista_fotos = []
    for imk in up_files:
        lista_fotos.append(imk.name)
        imsh = Image.open(imk)
        ims.append(imsh)
    return ims, lista_fotos

ims, lista_fotos = genera_imagenes_base(up_files)
    
for i, col in enumerate(st.columns(l_pics)):
    col.image(ims[i], width=150)

def genera_tensores_identidad(up_files):
    codes = []
    for foto in up_files:
        imagex = face_recognition.load_image_file(foto)
        facecodex = face_recognition.face_encodings(imagex)[0]
        codes.append(facecodex)
    return codes

codes = genera_tensores_identidad(up_files)
np.save('codes.npy', np.array(codes))
st.write('Reportando: Archivo Base Generado')
codes_df = pd.DataFrame(codes)

codes_dwl = codes_df.to_csv(index=False)
b64 = base64.b64encode(codes_dwl.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}" download="tensores_base.csv">Descargar Archivo</a>'

st.write('Archivo con Tensores Base: ')
st.markdown(href, unsafe_allow_html=True)

st.markdown('**Procedimiento: Carga de Pantallazos de Evento**')

up_evento = st.file_uploader('Elija pantallazos: ', accept_multiple_files=True)

ims_ev, lista_pics = genera_imagenes_base(up_evento)

st.write(lista_pics)

caras_embed = []
codesx = codes

for j, pic in enumerate(up_evento):
    ima = face_recognition.load_image_file(pic)
    face_locs = face_recognition.face_locations(ima)
    face_enco = face_recognition.face_encodings(ima)
    
    for k, facex in enumerate(face_enco):
        top, right, bottom, left = face_locs[k]
        face_frame = ima[top:bottom, left:right]
        pil_image = Image.fromarray(face_frame)
        pil_image_100 = pil_image.resize((100,100))
        rotulox = 'nom_' + str(j) + '_' + str(k) + '.jpg'
        #pil_image_100.save(rotulox)
        caras_embed.append([rotulox, facex, pil_image_100])
        
caras_evento_code_df = pd.DataFrame(caras_embed)
caras_evento_code_df.columns = ['rotulo', 'face_embed', 'image']

st.markdown('**Procedimiento: Comparación**')

resultados = []
for face in caras_evento_code_df.face_embed:
    matches = face_recognition.compare_faces(list(codesx), face)
    timestamp = datetime.datetime.now()
    try:
        indice = int(np.where(matches)[0])
        halla = lista_fotos[indice]
        resultados.append([timestamp, halla])
    except:
        resultados.append([timestamp, 'Desconocido'])

timestamp = str(datetime.datetime.now()).replace(':','-')

resultados_df = pd.DataFrame(resultados)
resultados_df.columns = ['timestamp', 'nombre']
resultados_df['arch_evento'] = caras_evento_code_df.rotulo
resultados_df['imagenes'] = caras_evento_code_df.image
resultados_df.to_csv('resultados'+timestamp+'.csv')

n_fig = 10
fig, ax = plt.subplots(1, n_fig, figsize=(21, 2))
for i in range(n_fig):
    if i > len(resultados_df)-1:
        img = Image.open('void.jpg')
        ax[i].imshow(img)
        ax[i].axis('off')
    else:
        img = resultados_df.imagenes.iloc[i]
        ax[i].imshow(img)
        ax[i].set_title(str(resultados_df.nombre.iloc[i]))
        ax[i].axis('off')
        
timestamp = str(datetime.datetime.now()).replace(':','-')
#plt.savefig('resultados_'+timestamp+'.jpg')

#st.image(Image.open('resultados_'+timestamp+'.jpg'), width=800)
st.pyplot(fig)

csv = resultados_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}" download="resultados_identidad.csv">Descargar Archivo</a>'

st.write('Archivo con Resultados: ')
st.markdown(href, unsafe_allow_html=True)