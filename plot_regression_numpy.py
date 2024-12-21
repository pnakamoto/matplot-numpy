# PARA PLOTAR UM GRAFICO DE REGRESSAO


import matplotlib

from sklearn.datasets import make_regression
x, y = make_regression(n_samples=1000, n_features=1, noise=100)
import matplotlib.pyplot as plt

plt.scatter(x,y)
plt.show()


==========================================================================================================
# USANDO BIBLIOTECA NUMPY 


# Importar bibliotecas necessárias
import numpy as np
from PIL import Image
import IPython.display as display
import urllib.request

# URL de uma imagem válida
url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"

# Caminho para salvar a imagem baixada
image_path = "test.jpg"

try:
    # Baixar a imagem e salvar no caminho especificado
    urllib.request.urlretrieve(url, image_path)

    # Abrir a imagem usando PIL
    image = Image.open(image_path)

    # Exibir a imagem no Colab
    display.display(image)

    # Obter as dimensões da imagem
    width, height = image.size
    print(f"Dimensões da imagem: Largura = {width}, Altura = {height}")

    # Converter a imagem para um array NumPy
    np_image = np.array(image)

    # Exibir as informações do array da imagem
    print(f"Formato do array NumPy: {np_image.shape}")

except Exception as e:
    print(f"Erro ao processar a imagem: {e}")



# SAIDA 

Versão do TensorFlow: 2.17.1
Treinando o modelo...
Treinamento concluído!
Previsão para x = 10.0:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
Resultado: [[18.987545]]
Pesos: [[1.9981949]], Biases: [-0.9944035]
