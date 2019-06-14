import tensorflow as tf

class Model:
    def __init__(self):
        INPUT_SHAPE = (90, 144, 1)
        OUTPUT_SHAPE = 6

        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu', input_shape=INPUT_SHAPE))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1000, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.25))

        self.model.add(tf.keras.layers.Dense(OUTPUT_SHAPE, activation='linear'))

        self.model.compile(loss='mse', optimizer='adam')

    def train(self, X, y, epochs):
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1, shuffle=True).history

    def save(self, path):
        self.model.save_weights(path+'weights.h5')

    def load(self, path):
        self.model.load_weights(path+'weights.h5')

    def info(self):
        return self.model.summary()

    def predict(self, image):
        return self.model.predict(image)
    

