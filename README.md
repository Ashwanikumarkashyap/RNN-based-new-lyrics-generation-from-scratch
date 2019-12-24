### 1 INTRODUCTION - RNN BASED LYRICS GENERATION

In the project, we aim to generate new song lyrics based on the artist’s previously released song’s context and style. We have chosen a Kaggle dataset of over 57,000 songs, having over 650 artists. The dataset contains artist name, song name, a link of the song for reference & lyrics of that song. We tend to create an RNN character-level language model on the mentioned dataset. Using model evaluation techniques, the model is checked for its accuracy and is parameter optimized. The trained model will predict the next character based on the context of the previous sequence and will generate new lyrics based on an artist’s style.

### 2 RELATED WORK

Before starting with this project, we were familiar with
two types of learning in the field of Machine Learning:
1. Supervised Learning
2. Unsupervised Learning

As our interest lies more towards the former, we have chosen Supervised Learning for the scope of our project. With Supervised learning, we can see the actual/predicted label of a data instance after complete training and can utilize the error function to calculate the difference between the actual class and the predicted class.

We came across applications of Deep Learning which is a subfield of machine learning, dealing with algorithms inspired by the structure and function of the brain called Artificial Neural Networks. Neural networks are made up of artificial neurons, similar in concept to neurons in the human brain. [10] These neurons are connected to each other, form huge connections, and the system works through the activation of these neurons. [10] We have studied the internal concepts of feed-forward neural networks. Feed-forward neural networks allow signals to travel one-way only - from input to output. There are no loops (feedback) i.e., the output of any layer doesn’t affect the same layer. [11] Feedforward neural networks tend to be straight-forward, from input to output.

In lyrics generation, we must keep the context of the previous words to generate future words/characters. Since feed-forward neural networks don’t allow the output of one layer to be used as the input for the same layer, we had to think of some algorithms which allow the feedback concepts. We came across the concepts of Recurrent Neural Networks. Feedback (Recurrent) networks can have signals traveling in both directions by

introducing loops in the network. [11] Computations derived from earlier inputs are fed back into the network, which gives them a kind of memory. [11] For our experiments, we have chosen over 57000 Song Lyrics Kaggle Dataset which contains an exhaustive list of artists and their songs. In order to be able to interpret the better results, we have built a python wrapper that allows the user to select the artist whom they want to generate the songs, or instead just simply to run the model on the entire global dataset.


### 3 IMPLEMENTATION

Keeping the concept of Recurrent Neural Networks in mind, we started the implementation of the model by defining certain initial configurations of the model to be followed throughout the training.
* Size of hidden layer nodes: In the model, the number of nodes in a hidden layer is taken as 100. Each node in a hidden layer takes a certain input and produces an output. After a certain activation, this output is served as an input for the next hidden state.
*  Input sequence length: It is the length of the sequence to be fed as the input from the entire input string of lyrics during each cycle of forward-backward propagation (epoch).
* Pass: Pass is the number of iterations to be performed during training until the entire data string is covered as the input (separated as small input chunks).
* Chars to predict: Chars to predict is the number of the next sequential characters to predict after the complete training of the model.

In addition to it, a utility model is implemented to be used throughout the model. The utility model holds information of – the data, dimensions of the data, data size, tokenized dictionaries and is responsible for the following

* Tokenization: The method tokenizes each unique character in the data to a specific unique number and reverse-map the same number back to the original character.
* Preprocess: The method is responsible for taking input from the user given a list of artists and based on the user’s selected artist, method preprocess the data from the entire dataset.
* Print Artist: The method prints the list of the available artist for which the songs are available.

The crux of the program resides in the module named ‘RNN’, which is required to be initialized using the configuration and utility modules. The RNN module is responsible for the following:

* Training: The method is responsible for the training on the dataset by executing the number of forward-backward propagation cycles defined in the configuration. With each iteration, the method update weights to minimize the loss computed during backpropagation.
* Forward Pass: The method runs a forward propagation given an input chunk of data and a certain hidden state at a time and computes a cross-entropy loss based on the predicted output.
* Backward Pass: Given a certain hidden state and predicted outputs, the backward pass is responsible for computing the change in weight matrices by back-propagating over the predicted output at every timestamp.
* Generate Sequence: Given a specific hidden state and a starting input character, the method generates the next sequence of a specific length using the trained weights updated after the training of the model.

### 4 RESULTS & ANALYSIS
We trained the RNN on the lyrics of the selected artist (here, an artist named Usher) and extended our model to predict sequences of predicted song lyrics based on previous songs of that artist. We trained our model with more than 5 7 ,000 songs, 100 hidden layers, 10 million iterations and predicting the next 200 characters of the sequence. After every 5 0th iteration, we check the cross-entropy loss. We investigated the cross-entropy
loss and based on our data and model parameter configurations observed how the cross-entropy loss decreases. The initial loss was 104.99467761 and it decreased to 40.42472864 after 10 million iterations. Some snapshots of the cross-entropy loss with iterations are captured below.

Iteration Cross  | Entropy Loss
--------------------- | --------------------
1 | 104 
161900 | 69
1000000 | 64
2836750 | 56
4064350 | 50
5332750 | 47
8213200 | 44
9999850 | 40

As we can see from the table the cross-entropy loss decreases with the number of iterations. As a result, the generated words make more sense
and relate to their past words. The following song lyrics were generated on training the RNN.

```
Iteration 50 -
“oeem rn(tg nMserenk o iiTs iyet,ehs itoemitusii s fo
yree esnh
eige hoy tyf a)esn uleetrnnoyhoe e yeeeem eiyseno
s
ne neseeit eeruey
yiemof e e e
el i eeeeat
oeenemneot s nt uM nglumirsoey eon(',sLh”
```

```
Iteration 2836750 -
“ten in your love alling as. I'm
Everyed oh


For to you're a-blamby and my my Myer to that a a dop
not I high do
What Athe brirdyy while you with you all my on hamb
thing, leacep on the rockay”
```

```
Iteration 9999850 -
“So goodbye
Don't die the even babaty whispel the parted snarsic
You said you

Sanessings that feel shown
Just pullilleh, this heat goad turna better nection the
wordlde lid.....
A
Don't all”
```

### 5 CONCLUSION & FUTURE WORK

In the project, we looked into the concepts of RNN and after thorough analysis and consideration, we observed certain trends and concluded that though RNN keeps the context of the previous input sequence, in order to generate the next output sequence and is able to generate a few sequences properly, RNN alone is not sufficient enough to automate lyrics generation. RNN suffers from a term called “exploding/vanishing gradient” in which with each iteration, the changes in the weight matrix are either significantly huge or just almost negligible. Such a problem results in inappropriate weight changes which lead to the poor prediction of the next sequence.

In order to overcome such a problem, a technique called Long-Short-Term-Memory (LSTM) is employed so that during backpropagation the change in weight remains in a considerable range. LSTM uses the concepts of an internal mechanism called gates, which helps in regulating the flow of information. Using these gates, LSTM knows which information to keep in a sequence, which to neglect. [7]

The key concept of LSTM is the cell state and its various gates. The cell state is responsible for transferring all the relevant information down to the sequence chain. [7] As the information is carried via a cell state, the new information gets added or removed using
gates. The three gates in an LSTM network includes:

1. Forget Gate: Responsible for throwing irrelevant in-
formation.
2. Input Gate: Responsible to update the cell state with
some input values.
3. Output Gate: Responsible for computing the next
hidden state.

Though RNN’s are good for processing sequential data, they are prone to ‘Short Term Memory’. LSTM is an alternate way to get rid of such a problem. In 2018, Bill Gates called LSTM a “Huge Milestone in advancing artificial intelligence” when OpenAI bots
were able to beat humans in DOTA 2 game. [ 8 ] For more accurate results and a significant decrease in the cross-entropy loss, we can include LSTM implementation in our model to make it better.

### 6 PROGRAMMING INSTRUCTIONS
* The code has the following dependencies, which need to be installed before running this code:
	* Python, Install version greater than 3.5
	* Pandas, More details at https://pandas.pydata.org/
	* numpy, More details at https://numpy.org/

* Execute driver.py file

Note - Since data is to be downloaded from the cloud, the pre-processing part of the program will take time.

### 7 REFERENCES
[1] Olah, C, (2015). “Understanding LSTM networks,” 2015, URL https://colah.github.io/posts/2015- 08 - Un-
derstanding-LSTMs/
[2] Danish, Irfan, “A brief summary of maths behind RNN”, URL https://medium.com/towards-artificial-in-telligence/a-brief-summary-of-maths-behind-rnn-recurrent-neural-networks-b71bbc183ff
[3] Banerjee Survo, (2018). “An Introduction to Recurrent Neural Networks” 2018, URL https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf
[4] Cross Entropy. “ML Glossary”, URL https://mlcheatsheet.readthedocs.io/en/latest/loss_func-
tions.html
[ 5 ] Britz, Denny (2015). Recurrent Neural NetworkTutorial, Part 1- Introduction to RNNs, 2015. URL http://www.wildml.com/2015/09/recurrent neural-networks-tutorial-part- 1 - introduction-to-rnns/
[6] Raval, Siraj, (2017). “Artificial Intelligence Educa-tion”, URL, 2017 https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A
[7] Raval, Siraj, (2017). “Recurrent Neural Network, The Math of Intelligence”, 2017 URL https://www.youtube.com/watch?v=BwmddtPFWtA
[8] Nguyen Michael, (2018). “Illustrated Guide to LSTM’s and GRU’s: A step by step explanation”. URL https://towardsdatascience.com/illustrated-guide-tolstms-and-gru-s-a-step-by-step-explanation44e9eb85bf
[9] Rodriguez, Jesus (July 2, 2018). "The Science Behind OpenAI Five that just Produced One of the Greatest Breakthrough in the History of AI". Towards DataScience. Retrieved 2019- 01 - 15.
[10] Olafenwa John, (2017). INTRODUCTION TO NEURAL NETWORKS, https://me-
dium.com/@johnolafenwa/introduction-to-neural-networks-ca7eab1d27d
[11] Dontas George, (2010). difference-between-feedforward-and-recurrent-neural-networks. URL https://towardsdatascience.com/illustrated-guide-tolstms-and-gru-s-a-step-by-step-explanation44e9eb85bf