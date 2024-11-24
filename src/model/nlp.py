import tensorflow as tf
from data_processing.split import to_dataset


shakespeare = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", origin=shakespeare)
with open(filepath) as f:
    shakespeare_text = f.read()

#print(shakespeare_text[:148])

text_vector_layer = tf.keras.layers.TextVectorization(split="character",standardize="lower")
text_vector_layer.adapt([shakespeare_text])
encoded = text_vector_layer([shakespeare_text])[0]

#Reduce the token ID numbers corresponding to characters by 2 to remove padding tokens and OOV
#패딩 토큰과 OOV를 제거하기 위해 문자에 해당하는 토큰 ID 번호를 2만큼 줄임
encoded -= 2 
#고유한 문자의 수
# Number of distinct characters 
n_tokens = text_vector_layer.vocabulary_size() - 2 
#텍스트의 총 문자 수
# Total number of characters in the text 
ds_size = len(encoded)


len = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], len=len, shuffle=True,
seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], len=len)
test_set = to_dataset(encoded[1_060_000:], len=len)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16,),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
check_point = tf.keras.callbacks.ModelCheckpoint("shakes_model.keras",monitor="val_accuracy",save_best_only=True)
history = model.fit(train_set, epochs=10, validation_data=valid_set, callbacks=[check_point])