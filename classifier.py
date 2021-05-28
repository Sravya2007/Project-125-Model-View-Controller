import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X = np.load('Project 125- Alphabet Recognition.npz')['arr_0']

y = pd.read_csv("Project 125- Alphabet Labels Data.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 3500, test_size = 500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(X_train_scaled, y_train)

def get_prediction(image):
    pillow_image = Image.open(image)

    grayscale_image = pillow_image.convert("L")

    #22 x 30 = 660 = number of pixels in 1 alphabet
    resized_image = grayscale_image.resize((22, 30), Image.ANTIALIAS)

    pixel_filter = 20

    min_pixel = np.percentile(resized_image, pixel_filter)

    scaled_image = np.clip(resized_image - min_pixel, 0, 255)

    max_pixel = np.max(resized_image)

    scaled_image = np.asarray(scaled_image)/max_pixel

    sample = np.array(scaled_image).reshape(1, 660)

    prediction = classifier.predict(sample)

    return prediction[0]