import tensorflow as tf

def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])

batch_size = 10
lstm_size = 10
max_steps = 100
num_layers = 1
num_features = 10

# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])

# Construct stacked LSTM
cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size, state_is_tuple=True)
lstm = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

# Get output and last hidden state
outputs, state = tf.nn.rnn(cell, unpack_sequence(words), dtype=tf.float32)
output = pack_sequence(outputs)