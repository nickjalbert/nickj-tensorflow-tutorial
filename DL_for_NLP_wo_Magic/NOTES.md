### Deep Learning for NLP without Magic (Part 1)

* [Video](https://www.youtube.com/watch?v=eixGKz0Asr8)

* Classic machine learning: 
  * human designed representations and features
  * Optimize weights to make best final prediction

* **Representation Learning** - learn good features or representation automatically

* **Deep learning** to automatically learn multiple levels of abstraction

* Learning representations automatically is great bc handcrafting is time
  consuming and often results in overspecified and incomplete

* Atomic representations make systems fragile.  Deep learning results in 
  robust distributed representations

* Learned representations are useful for the embedded similarity model

* Distributed rep. helps deal with curse of dimensionality by learning a
  similarity kernel

* NN allow for unsupervised feature and weight learning

* Human language is recursive, so a recurrent model seems plausible

* 2006 - we started figuring out how to train deep networks efficiently

* Why do we need non-linearities (e.g. logistic function)? 
  * In regression world, it's to keep things as probabilities
  * In the NN world, it to allow for arbitrary function approximation

* Summary of key components
  * Neuron - a logistic regression or similar
  * Input layer - input vector
  * Bias - always on feature/neuron "threshold"
  * Activation - neuron reponse, function is smooth (logistic/sigmoid)
  * Backpropagation - running stoachastic gradient descent layer by layer
  * Weight decay - regularization/ Bayesian prior

* Standard NLP word reps: one-hot
  * this loses similarity of words


* Lots of statistical NLP is building up models of word similarity
  * Represent a word by means of its neighbors in text
  * Traditionally done by clustering
  * Latent semantic analysis (LSA)


* **Neural word embedding** (NWE) - represent words as a dense vector
  * Gives you a powerful similarity space
  * For specific embeddings, you can actually perform vector ops and find interesting things
  * e.g. x_apple - x_apples = x_car - x_cars = x_family - x_families

* How do you learn NWE word representations?
    * Start with a random vector for each word and refine the vector to become a good representation
    * Find example sentence. 
    * Look up word vector for each word in sentence and concatenate for a context vector
    * Feed context vector through one layer NN (weighting + bias + nonlinearity) to get intermediate vector
    * Multiply by scoring vector to get a real number
    * this is the score of the context vector == example sentence
    * Then you want scoring to have nice properties like "Grammatically correct sentences score higher than broken sentences"
    * One specific example is to have a counterexample (a broken sentence), then:
    * Use gradient descent to minimize cost function C = max(0, 1 - score(good) + score(bad))
    * You make progress by scoring good higher or scoring bad lower

* 52:00

* Train with backpropagation
  * Taking derivatives and using the chain rule
  * Re-use derivatives from higher layers in computing derivatives for lower layers
  * It's analytically approximating how small changes in weights/wordvecs/scoring will
    change the outcome and then tuning that outcome as desired


* Question: How do you handle multiple contexts with a fixed input layer?  Do you fix context size beforehand?

* You can backpropagate labels into word vec.  E.g. you have sentiment label
  data, you can adjust the word representation to be useful for classifying
  sentiments.

* 1:10:21

* Example: tagging parts of speech.
  * Train word representations on unlabeled data (unsupervised pre-training, initialize word representations)
  * Take small amount of labeled data
  * Switch out last layer of network for softmax
  * Backprop labeled data errors to optimize the softmax (supervised fine tuning, refine word representations for specific task)

### Deep Learning for NLP without Magic (Part 2)

* [Video](https://www.youtube.com/watch?v=zHXOHqIyeD4)
