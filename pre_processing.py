import DCT
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_image(i, predictions_array, true_label, img):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] 
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def test_filter(image):
    # print(type(image))
    frequency = DCT.DCT(image)
    retorno = frequency.coeficientes_importantes()
    
    plt.imshow(retorno, cmap="gray")
    plt.title("Coeficientes Importantes")
    plt.show()
    retorno = frequency.passa_baixa()
    
    plt.imshow(retorno, cmap="gray")
    plt.title("Filtro Passa-Baixas")
    plt.show()

def filter_cosseno(images):
    imagens_filtradas = []
    print("iniciando Filtragem .....")
    inicio = time.time()
    for i in images:
        frequency = DCT.DCT(i)
        imagens_filtradas.append(frequency.coeficientes_importantes())
    
    fim = time.time()
    print("IDCT Levou: {0:.2f} segundos".format((fim - inicio)))
    print(len(imagens_filtradas))
    print(len(imagens_filtradas[0]))

    imagens_filtradas  = np.array(imagens_filtradas)
    return imagens_filtradas

def filter_passa_baixa(images):
    imagens_filtradas = []
    print("iniciando Filtragem .....")
    inicio = time.time()
    for i in images:
        frequency = DCT.DCT(i)
        imagens_filtradas.append(frequency.passa_baixa())
    
    fim = time.time()
    print("IDCT Levou: {0:.2f} segundos".format((fim - inicio)))
    print(len(imagens_filtradas))
    print(len(imagens_filtradas[0]))

    imagens_filtradas  = np.array(imagens_filtradas)
    return imagens_filtradas