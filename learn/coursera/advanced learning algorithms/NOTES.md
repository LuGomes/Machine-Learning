## Neural Networks

Applications:
- Speech recognition
- Computer vision
- Text (NLP)

Basic internals:

- Input layer is a vector of features
- Hidden layers output activation values
- Output layers outputs the prediction

Activations are higher level features that the NN learns by itself. 

Every layer has a vector as input, applies logistic regression or some other model and feeds that to the next layer.

For a layer l and neuron j, the activation function could be:

$$\vec a_j^{[l]} = g(\vec w_j^{[l]} \cdot \vec a ^{[l-1]}+b_j^{[l]})$$

