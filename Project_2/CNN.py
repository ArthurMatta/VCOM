# Shuffle Data
randomize = np.arange(len(X))
np.random.shuffle(randomize)
X = X[randomize]
Y = Y[randomize]

# Split Data
percent = round(len(X) * 0.8)

X_train, X_test = X[:percent], X[percent:]
Y_train, Y_test = Y[:percent], Y[percent:]

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Create CNN model
nb_filters = 32
pool_size = (2, 2)
kernel_size = (5, 5)
input_shape = X[0].shape
num_classes = 3

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size, input_shape=input_shape, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters*2, kernel_size, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters*4, kernel_size, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters*8, kernel_size, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, Y_train, batch_size=8, epochs=20, verbose=1, validation_split=0.1)

loss, score = model.evaluate(X_train, Y_train)
print(f'Test accuracy (train): {score:0.05f} loss: {loss:0.05f}')

loss, score = model.evaluate(X_test, Y_test)
print(f'Test accuracy (test): {score:0.05f} loss: {loss:0.05f}')