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
kernel_size = (3, 3)
input_shape = X[0].shape

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters*2, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, batch_size=64, epochs=5, verbose=1, validation_split=0.1)

# Test the model
score = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {score[1]:0.05f}')