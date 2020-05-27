# Simple Speech Recognizer

# tutorial from sirajj ravel @ https://www.youtube.com/watch?v=u9FPqkuoEJ8 
# and https://github.com/llSourcell/tensorflow_speech_recognition_demo/blob/master/demo.py

# install dependencies
# pip install tensorflow
# pip install tflearn
# pip install future

# load libraries/packages
import tflearn
import speech_data

# define learning rate and number of trainings = tradeoff between speed and accuracy of learning
learning_rate = 0.0001
training_iterations = 300000

# use help of waive data for speech sounds
batch = word_batch = speech_data.mfcc_batch_generator(64)

# break up into training and test data
X, Y = next(batch)
trainX, trainY = X, Y
testX, testY = X, Y

# create  multi-layer reccurent neural net since speech is a train of sounds
# first layer -> use tflearn and it takes two inputs width of data (number of utterances) and height
nnet = tflearn.input_data([None, 20, 80])
# second layer -> defining how many nets and dropout rate (prevent overfitting by dumping that which doesnt make cuttoff)
nnet = tflearn.lstm(nnet, 128, dropout = .80)
# third layer -> making all layers fully connecteed with each other and only recognize 10 digits and softmax to convert numerical data into probabilities
net = tflearn.fully_connected(nnet, 10, activation='softmax')
# fourth layer -> use regression to make single predition per utterance
net = tflearn.regression(nnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# init network through TF's DFF model
model = tflearn.DFF(net, tensorboard_verbose=0)
# init training loop
while 1:
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=64)
  y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)