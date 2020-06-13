# Programming Exercise 3: feed forward single hidden layer
Python implementation that explores how different parameters impact a feed-forward neural network with a single hidden layer. 

A brief analysis of the results is [provided in Portuguese](https://github.com/fredericoschardong/programming-exercise-3-single-hidden-layer/blob/master/report%20in%20portuguese.pdf). It was submited as an assignment of a graduate course named [Connectionist Artificial Intelligence](https://moodle.ufsc.br/mod/assign/view.php?id=2122514) at UFSC, Brazil.

In short, sine and cosine are fed to the FF network which tries to learn and predict their output. Different amounts of neurons, test subjects, learning rate, epochs and activation functions are testes separatedly. They all use gradient descent.

Base case uses 10 neurons in the hidden layer, 200 instances for training, 20000 epochs, 0.005 learning rate and some noise:
![alt text](https://github.com/fredericoschardong/programming-exercise-3-single-hidden-layer/blob/master/results/base%2C%20error:%201%2C8580186288063332.png "Logo Title Text 1")

The [result folder](https://github.com/fredericoschardong/programming-exercise-3-single-hidden-layer/tree/master/results) holds the results of other scenarios where different amount of neurons, training instances, epochs, learning rates and noise are tested.
