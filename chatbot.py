# Simple Chatbot for Movie Dialogue

# tutorial from sirajj ravel @ https://github.com/llSourcell/tensorflow_chatbot
# and https://www.youtube.com/watch?v=SJDEOWLHYVo

# install dependencies
# pip install tensorflow
# pip install numpy
# pip install scipy
# pip install six

# load libraries/packages
import tensorflow as tenfl
import data_utils
import seq2seq_model

# create training funtion
def train():
    # prepare encoding data (what's heard) with decoding data (response)
    enc_train, dec_train = data_utils.prepare_custom_data(gConfig['working_directory'])
    train_set = read_data(enc_train, dec_train)

# create funtion for model
def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
    return tenfl.nn.seq2seq.embedding_attention_seq2seq(
        encoder_inputs, decoder_inputs, cell,
        num_encoder_symbols = source_vocab_size,
        num_decoder_symbols = target_vocab_size,
        embedding_size = size,
        output_projection= output_projection,
        feed_previous= do_decode)

# run tf model with a session
with tenfl.Session(config=config) as sess:
    model = create_model(sess, False)
    while True:
        sess.run(model)