# Day 4: LSTM with IMDB - Does your neural network overfit and why?   

Even a shallow neural network (e.g an LSTM with one hidden layer) will have tens of thousands of parameters, whereas a 
deep neural network (e.g. GPT3) could have billions of parameters. In the meantime, training dataset size are typically much smaller
than the number of parameters. So why don't neural networksterribly overfit?  

### The concept of overfitting and underfitting in classical machine learning 
This is how the concept of overfitting/underfitting is often explained according to classical machine learning: 

![plot](./figures/bias_variance.png) (source: [Belkin et al, 2020](https://arxiv.org/abs/2003.00307))   

In this context, "Underfitting" happens when we have when the model (or function class) H doesn't have enough parameters and 
is therefore not complex enough to capture the pattern of the training data. Within the underfitting regime, both training and testing error are relative high. In contrast, 
"overfitting" happens when the model has many parameters and is too complex. It captures too much noise of the training data and 
does not generalize well to the testing data. Within the overfitting regime, training error is relatively low but testing error 
is relatively high. This curve implies that: 

i) there is a sweet spot in the middle where both training error and testing error are low.  
ii) When we increase the model complexity (e.g. through increasing the number of model parameters), the testing error is doomed to increase. 

### How many parameters does our current network have?   
   The general formula for calculating number of parameters in an LSTM network is given as follows:

   num_params = 4 * [(num_units + input_dim + 1) * num_units]
   
   Where 4 comes from the 4 neural network layers (W_forget, W_input, W_output, W_cell} and 1 comes from bias. 

   So the number of parameters in our LSTM model: 

   4 * (num_units^2 + (input_dim+1) * num_units) = 4 * (64 ^2 + 301 * 64) = 93,440 parameters 

   We are training the model on 50,000 * 0.9 * 0.9 = 40,500 samples 

### Why does our model not overfit?

   Given that we are training on 93,440 parameters but only uses 40,500 samples for training purposes, why doesn't it terribly overfit? 

#### Where does neuralwork fit on the bias/variance trade-off curve? 
 
   This contradiction is explained by [Belkin et al](https://arxiv.org/abs/2003.00307), who showed that the classical understanding is just 
   part of a bigger picture: 
   
   ![plot](./figures/bias_variance_2.png) 
   (source: [Belkin et al, 2020](https://arxiv.org/abs/2003.00307))   
   
   This plot effectively explains why neural networks work: With a large model capacity, we've moved into the overparameterized regime,  
   where training error is almost zero, and testing error stays low. (Note that the author describes this as "double descent curve", 
   but not a "double U" curve.) 
   The author reasons that this is because model complexity may not tell you how well the underlying inductive bias matches the current problem. 
   Give that a reasonable inductive bias is "the regularity or smoothness of a function as measured by a certain function space norm". A larger model 
   such as a neural network makes it easier to find interpolating functions with smaller norm, thus matching the inductive bias.

#### Our own experiment in the over-parameterized region 
In this experiment, the training, validation and test datasets come from a 8.1:0.9:1 split. Each experiment was run for a specific hidden dimension size for 
200 epochs. With number of hidden dimensions between 16 and 2048 and a fairly small dataset (training sample size = 40,500), 
we are already in the over-parameterized regime. Let's take a look at how the model performance vary based on hidden dimension. 

![plot](figures/hidden_size_effect.png) 

When the number hidden dimensions from 16 to 2,048, our training accuracy is between 0.972 and 0.999, indicating a near-zero 
training error. Based on the classical machine learning terminology, we are most certainly "overfitting". In the meantime, 
the testing and validation accuracy is reaching a plateau but shows no significant signs of deterioration. Our observation agrees with 
the description of Belkin et al in the "modern, interpolation regime". 

Our experiments tells us that we should not be terrified of "overfitting" by using a larger neural network, but we should keep 
in mind the trade-off between performance gain from increasing parameters and computational cost. 

### Neural network and its intrinsic dimensions  
   Another way to look at this is, for simpler models such as linear regression, the number of parameters is equal to the degrees of freedom. However, [the number 
   is much lower than the number of parameters for deep neural networks](https://arxiv.org/abs/1603.09260). 
   One way of getting a quantitative understanding of degrees of freedom for DNN is by calculating the intrinsic dimension. This [paper](https://arxiv.org/abs/1804.08838) 
   from the Uber AI lab comes with a [fun and educative video](https://www.youtube.com/watch?v=uSZWeRADTFI&feature=emb_logo).  
 
   Briefly, they proposed to train the neural network in a smaller, randomly oriented subspace rather than the large, native space. 
    ![plot](figures/intrinsic_dim_paper1.png)    
   This way, the gradient of loss in a space of D dimensions only moves in the d-dimensional subspace: 
   
   ![plot](figures/intrinsic_dim_paper2.png) 

   And finally, the intrinsic dimension is defined as d<sub>int90</sub>, or the d dimensions at which the model achieves 90% performance. 

#### Our experiment with intrinsic dimensions 
To find out the intrinsic dimension of our LSTM model wrt IMDB dataset, I used [an existing github 
implementation](https://github.com/jgamper/intrinsic-dimensionality) with some minor modifications (see code [here]()). 

This requires a two-step modification from a regular NN module: 
First, in the initialization stage: Declare the random P matrix, extract all weights from pytorch, store them in a class param 
and delete the originals:  

           for name, param in module.named_parameters():
               ... 

                # Generates random projection matrices P, sets them to no grad
                self.random_matrix[name] = (
                    torch.randn(matrix_size, requires_grad=False).to(device)
                    / intrinsic_dimension ** 0.5
                )

                base, localname = module, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)
                self.name_base_localname.append((name, base, localname))

           for name, base, localname in self.name_base_localname:
               delattr(base, localname)

Second, in the forward pass: Apply the projection matrix P, and then copy the parameters back to their original location:   
      
      def forward(self, x, x_lens):
        # Iterate over the layers
        for name, base, localname in self.name_base_localname:

            # Product between matrix P and \theta^{d}
            ray = torch.matmul(self.random_matrix[name], self.V)

            # Add the \theta_{0}^{D} to P \dot \theta^{d}
            param = self.initial_value[name] + torch.squeeze(ray, -1)

            setattr(base, localname, param)

            ...        
   
And now we can run a range of experiments by varying the dimensions of the subspace from 10 to 500: 
      
      for dim in [10, 50, 100, 200, 300, 400, 500]:
         run_experiment(dim)

Our experiment shows that the intrinsic dimension of our LSTM model is ~400, or **0.4%** of the parameter size. We only run it 
for hidden dimension=64, but you are likely to see similar answers with a different choice, as the magnitude of intrinsic dimensions 
is fairly robust against changes in layer width and depth. 
 

## Main takeaway 
1. The classical concept of "overfitting" and "underfitting" and its implications do not work well with neural networks in
   the overparameterized region. Overparameterized neural networks are likely to achieve zero training error without compromising 
   testing performance. In other words, "overfitting" on training data is not the enemy of neural networks (and other complex ML models).  
2. The intrinsic dimension idea can further help us understand why "overfitting" does not happen with neural networks: 
   a) The intrinsic dimension of a neural network is orders of magnitude smaller than the number of parameters. 
   b) The size of intrinsic dimension (as a measure of model complexity) is quite robust against changes in the number of parameters (layer width and depth).  
3. Given 1 and 2, when choosing number of hidden dimensions of our LSTM model, we are less concerned that having too many parameters 
   would "overfit" and lead to large testing error. Instead, we are more concerned with the trade-off between computational cost 
   and performance. In other words, you are likely to choose a middle ground, where your computation is not yet too expensive 
   and your model performance is reasonably good.


## Further reading 
1. [Random projection using fastfood](http://proceedings.mlr.press/v28/le13.pdf) 
2. [Lilian Weng's blog](https://lilianweng.github.io/lil-log/2019/03/14/are-deep-neural-networks-dramatically-overfitted.html#intrinsic-dimension) (Note: this post is **a rare gem**. 
   Highly recommended for a thorough review and insightful discussions on model complexity.)
3. [The need for biases in learning generalization](http://www-cgi.cs.cmu.edu/~tom/pubs/NeedForBias_1980.pdf)

