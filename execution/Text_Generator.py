
import pandas as pd
import numpy as np
import tensorflow as tf
from prac import model

def generate_text(model2, temp):
  idx_to_morp_df = pd.read_csv(r'C:\Users\집\Flask\static\idx_to_morp.csv')
  idx_to_morp = {i: u for i, u in idx_to_morp_df.values}

  num_generate = 120
  input_eval = np.array([np.random.randint(0,17670)])
  print('input_eval : ',input_eval)
  input_eval=np.expand_dims(input_eval,axis=0)
  print('expended i e',input_eval)
  text_generated = []

  temperature = temp

  model2.reset_states()
  for i in range(num_generate):
      predictions = model2(input_eval)
      print('predictioons ',predictions)
      predictions = tf.squeeze(predictions, 0)

      predictions = predictions / temperature
      print(predictions.shape)
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx_to_morp[predicted_id])
      print(text_generated)

  return (' '.join(text_generated))

def main():

    vocab_size = 17673
    embedding_dim = 820
    rnn_units = 480
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[1, None]))
    model.add(tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Dense(vocab_size))
    model.load_weights('../static/model/model_weigths')
    poem = ''
    while 1:
        a = generate_text(model, 1.6)
        if '\n\n' in a:
            poem = '\t\t\t\t\tPoem Written by A.I\n\n'+a
            # print('\t\t\t\t\t무 제\n\n', a)
            break
    poem.replace('\n','<br/>')

    return poem


if __name__ == "__main__":
	main()

