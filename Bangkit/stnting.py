import tensorflow as tf

# Import data
data = pd.read_csv("data_stunting.csv")

# Preprocessing data
data = data.dropna()
data = data.normalize()

# Buat model prediksi
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Latih model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(data[["usia", "berat_badan_lahir", "tinggi_badan_lahir", "jenis_kelamin"]], data["stunting"], epochs=100)

# Konversi model ke JSON
model_json = model.to_json()

# Buat model baru
new_model = tf.keras.models.model_from_json(model_json)

# Lakukan prediksi
prediksi = new_model.predict(data[["usia", "berat_badan_lahir", "tinggi_badan_lahir", "jenis_kelamin"]])

# Cek akurasi
akurasi = tf.metrics.accuracy(data["stunting"], prediksi)
print(akurasi)
