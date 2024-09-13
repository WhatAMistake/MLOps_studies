from keras.models import load_model
from keras.models import save

loaded_model = load_model('./my-trained-model.h5')
save_path = './saved-model.h5'
save(loaded_model, save_path)

loaded_model = load_model('./saved-model.h5')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils

earlystopper = EarlyStopping(monitor='val_loss', patience=5)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(loaded_model)))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', AUC()])

model.fit(X, Y, epochs=50, batch_size=10, validation_split=0.2, callbacks=[earlystopper])

accuracy = model.evaluate(X_test, Y_test)
print("Accuracy:", accuracy * 100, "%")
