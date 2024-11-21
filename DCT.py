import numpy as np
# import matplotlib.pyplot as plotagem
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.io.wavfile
import math
import time

class DCT:
    
    def __init__(self,image):
        # self.imagem_original = Image.open(image)
        self.imagem_frequencia = self.dct_2d(image.copy())
        # self.idct_2d()
        self.R = self.imagem_frequencia.shape[0]
        self.C = self.imagem_frequencia.shape[1]

    def dct_1d(self, vetor):
        N = len(vetor)
        # LINHA RESULTANTE PARA O DOMINIO DA FREQUENCIA
        X = np.zeros(N)
        aK = math.sqrt(2.0/N)
        N = len(vetor)

        for k in range(N):
            ck = math.sqrt(1.0/2.0) if k == 0 else 1
            somatorio = 0

            for n in range(N):
                a1 = 2.0 * math.pi * k * n
                a2 = k * math.pi
                nn = 2.0 * N

                somatorio += vetor[n] * math.cos((a1/nn) + (a2/nn))

            X[k] = aK * ck * somatorio

        return X

    def idct_1d(self, vetor):
        N = len(vetor)
        #LINHA NO DOMINIO DO ESPAÇO RESULTANTE
        x = np.zeros(N)
        aK = math.sqrt(2.0/N)
        N = len(vetor)

        for n in range(N):
            somatorio = 0
            for k in range(N):
                a1 = 2.0 * math.pi * k * n
                a2 = k * math.pi
                nn = 2.0 * N
                cK = math.sqrt(1.0/2.0) if k == 0 else 1

                somatorio += cK * vetor[k] * math.cos((a1/nn) + (a2/nn))

            x[n] = aK * somatorio

        return x
    # TRANSFORMADA 2D 
    def dct_2d(self,imagem):
        #EXIBINDO ORIGINAL
        imagem_dct = np.zeros(imagem.shape)
        # inicio = time.time()

        # APLICA DCT NAS LINHAS
        for indice, linha in enumerate(imagem):
            imagem_dct[indice] = self.dct_1d(linha)
 
        # TRASPÕE MATRIZ
        imagem_dct = imagem_dct.T
        
        # APLICA DCT NAS COLUNAS
        for indice, coluna in enumerate(imagem_dct):
            imagem_dct[indice] = self.dct_1d(coluna)


        # RETORNANDO A MATRIZ SEM TRANSPOSIÇÃO
        imagem_dct = imagem_dct.T
        imagem_retorno = imagem_dct

        imagem_norm = imagem_dct.copy()
        # COLOCANDO OS COSSENOS COM O MODULO
        imagem_dct = self.normaliza_imagem(imagem_norm)
        # CALCULANDO COEFICIENTE DC


        # APAGA NÍVEL DC
        imagem_dct[0][0] = 0
        # EXIBE DOMINIO DA FREQUENCIA
        return imagem_retorno

    # TRANSFORMADA INVERSA
    def idct_2d(self):
        imagem_idct = np.zeros(self.imagem_frequencia.shape)
        inicio = time.time()

        # IDCT NAS LINHAS
        for indice, linha in enumerate(self.imagem_frequencia):
            imagem_idct[indice] = self.idct_1d(linha)

        imagem_idct = imagem_idct.T
        # IDCT NAS COLUNAS
        for indice, coluna in enumerate(imagem_idct):
            imagem_idct[indice] = self.idct_1d(coluna)

        fim = time.time()
        print("IDCT Levou: {0:.2f} segundos".format((fim - inicio)))

        # RETORNANDO A MATRIZ SEM TRANSPOSIÇÃO
        imagem_idct = imagem_idct.T

        plt.imshow(imagem_idct, cmap="gray")
        plt.title("Retorno")
        plt.show()

    def idct_2d_coeficientes(self,imagem_coeficientes):
        imagem_idct = np.zeros(imagem_coeficientes.shape)
        # inicio = time.time()

        # IDCT NAS LINHAS
        for indice, linha in enumerate(imagem_coeficientes):
            imagem_idct[indice] = self.idct_1d(linha)

        imagem_idct = imagem_idct.T
        # IDCT NAS COLUNAS
        for indice, coluna in enumerate(imagem_idct):
            imagem_idct[indice] = self.idct_1d(coluna)

        # RETORNANDO A MATRIZ SEM TRANSPOSIÇÃO
        imagem_idct = imagem_idct.T
        return imagem_idct
#   ahsdkhaksd
    def normaliza_imagem(self,imagem):
        for i in range(imagem.shape[0]):
            for j in range(imagem.shape[1]):
                imagem[i][j] = abs(imagem[i][j])
        return imagem

    # FILTRA OS COEFICIENTES MAIS IMPORTANTES
    def coeficientes_importantes(self):
        # PRECISA DO VETOR NORMALIZADO PARA SELECIONAR IMPORTANTES
        imagem_frequencia = self.normaliza_imagem(self.imagem_frequencia.copy())
        # PRECISA DO ORIGINAL PARA ARMAZENAR O RESULTANTE
        imagem_frequencia_original = self.imagem_frequencia.copy()
        # RECEBENDO QUANTIDADE DE FREQUENCIAS IMPORTANTES DOS USUÁRIOS

        imagem_cos_importantes = np.zeros(imagem_frequencia.shape)
        # TRANFORMA EM LISTA TRADICIONAL PARA ACESSO AO INDICE
        imagem_frequencia = imagem_frequencia.tolist()

        maximo = 0 
        index_maximo = 0
        iteracao = 0
        inicio = time.time()
        # REPETINDO ITERACAO ATÉ O NUMERO DE COSSENOS
        while(iteracao < 40):
            # COLETANDO A MAXIMA FREQUENCIA
            for i in range(len(imagem_frequencia)):
                maximo_aux = max(imagem_frequencia[i])
                #ARMAZENA INDICE DO COSSENO PARA ARMAZENAR NO RESULTANTE
                index_aux_max = imagem_frequencia[i].index((max(imagem_frequencia[i])))
                if maximo_aux > maximo:
                    maximo = maximo_aux
                    index_maximoi,index_maximoj = i,index_aux_max
            # IMAGEM RESULTANTE RECEBE FREQUENCIA
            imagem_cos_importantes[index_maximoi][index_maximoj] = imagem_frequencia_original[index_maximoi,index_maximoj]
            # IMAGEM LIDA APAGA MAIOR FREQUENCIA
            imagem_frequencia[index_maximoi][index_maximoj] = 0 
            iteracao += 1
            maximo = 0
            index_maximoi = 0
            index_maximoj = 0

        # RECONSTRUINDO NO ESPACO PARA VER O RESULTANDO
        return self.idct_2d_coeficientes(imagem_cos_importantes)

    #FILTRO DE SUAVIZAÇÃO APLICANDO FILTRO NXN NOS PIXEIS DA IMAGEM NO DOM FREQUENCIA
    def passa_baixa(self):

        imagem_frequencia = self.imagem_frequencia.copy()
        # RECEBE TAMANHO DO FILTRO

        # IMAGEM RESULTANTE
        imagem_filtrada = np.zeros(imagem_frequencia.shape)
        # SE COEFICIENTE LOCALIZADO COUBER NO FILTRO ADICIONA A MATRIZ RESULTANTE
        # 11,14,17
        for i in range(len(imagem_frequencia)):
            if(i > 11):
                break
            for j in range(len(imagem_frequencia[i])):
                if(j > 11):
                    break
                else:
                    imagem_filtrada[i][j] = imagem_frequencia[i][j]
        # NORMALIZA APENAS PARA EXIBIÇÃO

        # PASSA PARA O DOMINIO DO ESPACO PARA VER O RESULTADO
        return self.idct_2d_coeficientes(imagem_filtrada)

