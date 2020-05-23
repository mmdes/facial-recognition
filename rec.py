import os
import cv2
import numpy as np
from PIL import Image
import pickle

from keras_vggface.vggface import VGGFace
from keras.layers import Flatten
from keras.models import Model, Sequential

from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy.spatial import distance

from PIL import Image
import numpy as np
import cv2

from keras_vggface.vggface import VGGFace
from keras.applications.vgg16 import VGG16

from keras.layers import Flatten
from keras.models import Model, Sequential
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from scipy.spatial import distance
import matplotlib.pyplot as plt

#PRA RODAR NA GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from scipy import spatial



def redim(alt, larg, imagem):
    #imagem = Image.open('p (' + str(i)+ ')' + '.tif')
    porc = larg/float(imagem.size[0])
    if(porc * imagem.size[1] <= alt):
        novaAlt = int(porc * imagem.size[1])
        novaLarg = larg
    else:
        porc = alt/float(imagem.size[1])
        if(porc * imagem.size[0] <= larg):
            novaLarg = int(porc * imagem.size[0])
            novaAlt = alt
    imagem = imagem.resize((novaLarg, novaAlt), Image.ANTIALIAS)
    #Fundo
    new_img = Image.new('RGB', (larg, alt), (0, 0, 0))
    areaCol = (int(round(((larg - imagem.size[0])/2), 0)), int(round(((alt - imagem.size[1])/2),0)))
    new_img.paste(imagem, areaCol)
    #imagem = np.array(fundo)
    #imCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite('p (' + str(i)+ ')' + '.tif', imCinza)
    return new_img



def ler():
    images_array = []
    label_array = []
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = '/home/matheus/Documentos/facial-recognition'
    image_dir = os.path.join(BASE_DIR, "people")
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                redim_image = image.load_img(path, target_size=(224, 224))
                x = image.img_to_array(redim_image)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=1) 
                images_array.append(x)
                label_array.append(label)
    return images_array, label_array

def euclideanDistance(base, teste):
    var = base-teste
    var = np.sum(np.multiply(var, var))
    var = np.sqrt(var)
    return var

def cosineDistance(vet1, vet2):
    dist = 1 - spatial.distance.cosine(vet1, vet2)
    return dist

def vgg_face():
    vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
    #vgg_model.summary()
    model = Model(inputs=vgg_model.layers[0].input, outputs=vgg_model.layers[-2].output)
    return model


def separa(images_array, label_array):
    i = 1
    aux = []
    aux2 = []
    arrayLabSep = []
    arrayImaSep = []

    while i != len(label_array): 
        if label_array[i] == label_array[i-1]:
            aux.append(label_array[i-1])
            aux2.append(images_array[i-1])
        else:
            aux.append(label_array[i-1])
            aux2.append(images_array[i-1])
            arrayLabSep.append(aux)
            arrayImaSep.append(aux2)
            aux = []
            aux2 = []
        i = i +1
    aux.append(label_array[i-1])
    aux2.append(images_array[i-1])
    arrayLabSep.append(aux)
    arrayImaSep.append(aux2)

    return arrayImaSep, arrayLabSep


def pred(modelo, vImg):
    vPred = []
    for i in range(len(vImg)):
        predict = modelo.predict(vImg[i])
        vPred.append(predict)
    return vPred


def calculaDistanciaIguais(array1, array2, dist):
    array_distancia = []
    j = 0
    for i in range(len(array1)):
        while j != len(array2):
            val = dist(array1[i], array2[j])
            array_distancia.append(val)
            j = j + 1
        j = i + 1
    return array_distancia

def calculaDistanciaDif(array1, array2, dist):
    array_distancia = []
    for i in range(len(array1)):
        for j in range(len(array2)):
            val = dist(array1[i], array2[j])
            array_distancia.append(val)
    return array_distancia

def soma(vet):
    vet = np.array(vet)
    soma = vet[1]
    i = 1
    while i < len(vet):
        soma = soma + vet[i]
        i = i + 1
    return soma

def media_all(vet):
    media = []
    for i in range(len(vet)):
        media.append(soma(vet[i])/len(vet[i]))
    return media

def maximo(vet):
    maximo = vet[0]
    posicao = 0
    for i in range(len(vet)):
        if vet[i] > maximo:
            maximo = vet[i]
            posicao = i
    return maximo, posicao


arrayImagens, arrayEtiquetas = ler()
vgg_face = vgg_face()



array_pred = pred(vgg_face, arrayImagens)
del arrayImagens

img_pred_sep, etiqueta = separa(array_pred, arrayEtiquetas)
del array_pred, arrayEtiquetas

array_media = media_all(img_pred_sep)
del img_pred_sep

etiq = []
for i in range(len(etiqueta)):
    aux = etiqueta[i][0]
    etiq.append(aux)
del i, aux, etiqueta


predicts = []
for i in range(13):
    path = '/home/matheus/Documentos/facial-recognition/test/'+str(i)+'.jpg'
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    predict = vgg_face.predict(x)
    predicts.append(predict)
del path, predict, x, img, i
    

distancias = []
vet = []
for i in range(len(predicts)):
    for j in range(len(array_media)):
        aux = cosineDistance(predicts[i], array_media[j])
        vet.append(aux)
    distancias.append(vet)
    vet = []
del i, vet, aux, j

posicoes = []
for i in range(len(distancias)):
    maxi, posicao = maximo(distancias[i])
    posicoes.append(posicao)
del i, maxi, posicao

resultados = []
for i in range(len(posicoes)):
    resultados.append(etiq[posicoes[i]])
del i

print(resultados)
