import tensorflow as tf
def train_model(model, X_train, y_train, X_val, y_val, epochs=1):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
    return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stop])
