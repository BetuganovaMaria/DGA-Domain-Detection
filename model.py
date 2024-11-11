import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Activation, Embedding, LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

dataset = pd.read_csv('dataset.csv')
x, y = dataset['domain'], dataset['is_dga']

max_length = np.max([len(i) for i in x])
chars = {i: index for index, i in enumerate(set(''.join(x)), start=1)}
x_encoded = [[chars[ch] for ch in domain if ch in chars] for domain in x]
x_padded = keras.preprocessing.sequence.pad_sequences(x_encoded, maxlen=max_length)
max_features = len(chars) + 1

x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.2)

model = keras.Sequential()
model.add(Embedding(max_features, 128, input_length=max_length))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(x_train, y_train, batch_size=16, epochs=1)

y_prediction = model.predict(x_test)

cm = confusion_matrix(y_test, y_prediction > 0.5)
percent_cm = cm * 100 / len(x_test)

labels = ['legitimate', 'dga']

for i, label_i in enumerate(labels):
    for j, label_j in enumerate(labels):
        print("%s/%s: %.2f%% (%d/%d)" % (label_i, label_j, (percent_cm[i][j]), cm[i][j], cm[i].sum()))

print('AUC: ', roc_auc_score(y_test, y_prediction))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(visible=False)
cax = ax.matshow(percent_cm, cmap='coolwarm')
plt.title('Confusion matrix')
fig.colorbar(cax)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
