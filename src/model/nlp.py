from data_processing.split import to_dataset
import tensorflow as tf
import os




shakespeare = "https://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", origin=shakespeare)
with open(filepath) as f:
    shakespeare_text = f.read()

#print(shakespeare_text[:148])

text_vector_layer = tf.keras.layers.TextVectorization(split="character",standardize="lower")
text_vector_layer.adapt([shakespeare_text])
encoded = text_vector_layer([shakespeare_text])[0]

model = tf.keras.models.load_model("../data/models/best_shakes_model.keras")

wrapped_model = tf.keras.Sequential([
    text_vector_layer, 
    tf.keras.layers.Lambda(lambda X: X-2),
    model
])

log_probas = tf.math.log([0.5, 0.4, 0.1])
tf.random.set_seed(42)  
print(tf.random.categorical([log_probas], num_samples=10))

def next_char(text, temperature=1):
    y_proba = wrapped_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) [0,0]
    return text_vector_layer.get_vocabulary()[char_id + 2]

def text_extender(text,n_chars=200,temperature=1):
    for _ in range(n_chars):
        text += next_char(text,temperature)
    return text

tf.random.set_seed(42)
print(text_extender("I told about tha", temperature=0.02))
print(text_extender("I told about tha", temperature=1))
print(text_extender("I told about tha", temperature=100))
