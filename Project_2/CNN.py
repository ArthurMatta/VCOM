# Create CNN model
nb_filters = 1
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = X[0].shape
model = Sequential()
model.add(Convolution2D(nb_filters, kernel_size, input_shape=input_shape, padding='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, Y_train, batch_size=64, epochs=3, verbose=1, validation_split=0.1)

# Test the model
score = model.evaluate(X_test, Y_test)
print(f'Test accuracy: {score[1]:0.05f}')