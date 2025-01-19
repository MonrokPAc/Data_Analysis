import re
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import string, os, random
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()
import cv2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn. metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Embedding, LSTM
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



file_path = 'cancer issue.csv'
data = pd.read_csv(file_path)

df = data.copy()

categorical_columns = ['Gender', 'Race/Ethnicity', 'SmokingStatus', 'FamilyHistory', 
                       'CancerType', 'Stage', 'TreatmentType', 'TreatmentResponse', 
                       'Recurrence', 'GeneticMarker', 'HospitalRegion']
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

numerical_columns = ['Age', 'BMI', 'TumorSize', 'SurvivalMonths']
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

#print(df.head())


X = df.drop('SurvivalMonths', axis=1)
y = df['SurvivalMonths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4224)
y_train = keras.utils.to_categorical(y_train, len(y.unique()))
y_test = keras.utils.to_categorical(y_test, len(y.unique()))

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)), 
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(y_train.shape[1], activation='softmax')
])

optimizer = Adam(learning_rate=0.000006)
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=optimizer)

history1 = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)

model.summary()
results = model.evaluate(X_test, y_test, verbose=0)
print('Losses:', results[0])
print('Accuracy:', results[1])

plt.figure(figsize=(6, 3))
plt.plot(history1.history['loss'], label='Losses')
plt.plot(history1.history['accuracy'], label='Accuracy')
plt.legend(loc='upper right')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.show()

path = 'PetImages/'
categories = ['Cat', 'Dog']

x, y = [], []
for n, category in enumerate(categories):
    category_path = os.path.join(path, category)
    count = 0  # ������� ����������
    for f in os.listdir(category_path):
        if count >= 1000:
            break
        image_path = os.path.join(category_path, f)
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (30, 50))
            image = image / 255.0
            x.append(image)
            y.append(n)
            count += 1
        except Exception as e:
            pass


x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2)
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

print('x_train shape:', x_train.shape)
print('train number:', x_train.shape[0])
print('test number:', x_test.shape[0])


model2 = Sequential([
    Input(shape=(50, 30, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

optimizer = Adam(learning_rate=0.009)
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

history2 = model2.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=100)

results2 = model2.evaluate(x_test, y_test, verbose=0)
print('Losses:', results2[0])
print('Accuracy:', results2[1])


plt.figure(figsize=(6, 3))
plt.plot(history2.history['loss'], label='Losses')
plt.plot(history2.history['accuracy'], label='Accuracy')
plt.legend(bbox_to_anchor=(1, 1))
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.show()


path = 'PetImages/'
categories = ['Cat', 'Dog']


x, y = [], []

for n, category in enumerate(categories):
    category_path = os.path.join(path, category)
    count = 0  # ������� ����������
    for f in os.listdir(category_path):
        if count >= 200:  # ����������� �� 50 ����������
            break
        image_path = os.path.join(category_path, f)
        image = cv2.imread(image_path)
        try:
            image = cv2.resize(image, (224, 224))
            image = image / 255.0
            x.append(image)
            y.append(n)
            count += 1
        except Exception as e:
            pass


x = np.array(x)
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)


base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


base_model.trainable = False


trans_model = Sequential([
    base_model, 
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax') 
])


trans_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
trans_model.summary()

history = trans_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=10)


results_t = trans_model.evaluate(x_test, y_test, verbose=0)
print('Losses:', results_t[0])
print('Accuracy:', results_t[1])


plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'], label='Losses')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.legend(bbox_to_anchor=(1, 1))
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.show()


# Load and clean the dataset
file_path = 'spam.csv'
spam_data = pd.read_csv(file_path, encoding='latin-1')

# Keep only relevant columns
spam_data = spam_data[['v1', 'v2']]
spam_data.columns = ['Category', 'Message']

# Define a cleaning function
def cleaning(value):
    punctuation_cleaning = []
    stopwords_cleaning_string = ''
    
    for i in value:
        if i not in string.punctuation:
            punctuation_cleaning.append(i)
    punctuation_cleaning = "".join(punctuation_cleaning).split()
    
    for j in punctuation_cleaning:
        if j.lower() not in stopwords.words("english"):
            stopwords_cleaning_string += j.lower() + ' '
    
    return stopwords_cleaning_string

# Apply cleaning to messages
spam_data['Message'] = spam_data['Message'].apply(cleaning)

# Split data into features and labels
X = spam_data['Message']
y = spam_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train) 
y_test = encoder.transform(y_test)

# Tokenize and pad sequences
token = Tokenizer(lower=False)
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

# Determine the maximum sequence length
array = [len(i) for i in X_train]
maxlen = int(np.ceil(np.mean(array)))

# Pad sequences
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
total_words = len(token.word_index) + 1

# Build the LSTM model
model = Sequential([
    Embedding(total_words, 32, input_length=maxlen),
    LSTM(10),
    Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model
results = model.evaluate(X_test, y_test, verbose=0)
print('Losses:', results[0])
print('Accuracy:', results[1])

# Visualize training results
plt.figure(figsize=(6, 3))
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.legend(['Losses', 'Accuracy'], bbox_to_anchor=(1, 1))
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.show()


# Load and clean the dataset
file_path = 'spam.csv'
spam_data = pd.read_csv(file_path, encoding='latin-1')

# Keep only relevant columns
spam_data = spam_data[['v1', 'v2']]
spam_data.columns = ['Category', 'Message']

# Define a cleaning function
def cleaning(value):
    punctuation_cleaning = []
    stopwords_cleaning_string = ''
    
    for i in value:
        if i not in string.punctuation:
            punctuation_cleaning.append(i)
    punctuation_cleaning = "".join(punctuation_cleaning).split()
    
    for j in punctuation_cleaning:
        if j.lower() not in stopwords.words("english"):
            stopwords_cleaning_string += j.lower() + ' '
    
    return stopwords_cleaning_string

# Apply cleaning to messages
spam_data['Message'] = spam_data['Message'].apply(cleaning)

# Split data into features and labels
X = spam_data['Message']
y = spam_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)

# Encode labels
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train) 
y_test = encoder.transform(y_test)

# Tokenize and pad sequences
token = Tokenizer(lower=False)
token.fit_on_texts(X_train)
X_train = token.texts_to_sequences(X_train)
X_test = token.texts_to_sequences(X_test)

# Determine the maximum sequence length
array = [len(i) for i in X_train]
maxlen = int(np.ceil(np.mean(array)))

# Pad sequences
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
total_words = len(token.word_index) + 1

# Load GloVe embeddings
embedding_index = {}
embedding_dim = 50
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create the embedding matrix
embedding_matrix = np.zeros((total_words, embedding_dim))
for word, i in token.word_index.items():
    if i >= total_words:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Build the model
trans_model2 = Sequential([
    Embedding(total_words, embedding_dim, weights=[embedding_matrix], trainable=False),
    LSTM(10, dropout=0.2, recurrent_dropout=0.2, name="lstm_layer"),
    Dense(1, activation='sigmoid', name="output_layer")
])

# Compile the model
trans_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history_trans2 = trans_model2.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, shuffle=True)

# Evaluate the model
results3_t = trans_model2.evaluate(X_test, y_test, verbose=0)
print('Losses:', results3_t[0])
print('Accuracy:', results3_t[1])

# Visualize training results
plt.figure(figsize=(6, 3))
plt.plot(history_trans2.history['loss'], label='Losses')
plt.plot(history_trans2.history['accuracy'], label='Accuracy')
plt.legend(bbox_to_anchor=(1, 1))
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.show()