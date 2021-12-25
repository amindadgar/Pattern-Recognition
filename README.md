# pattern-recognition-homework
This codes belong to my pattern recognition Homeworks.

## HW1 Codes description
- **Question no2:**<br>
  Generate 100000, n dimension uniform i.i.d. random variables and calculate the volume of the n dimension shape of it.
  *Answer:* 50, 100, 200 and 1000 dimension iid uniform random variables was created. The graphs were plotted. 
- **Question no3:** <br>
  calculate the probability functions, $P(Y)$, $P(Y|X)$ using the joint probability function $P(X,Y)$. Also plot the density functions of thoes probability. <br>
  
 
   $$
   \begin{equation}
   p(x,y) = \frac{1}{2\pi ab} \exp({-(\frac{y-\mu}{2a^2} + \frac{(x-y)^2}{2b^2})})
   \end{equation}
   $$
 
- **Question no4:** <br>
   (a) Calculate the eignvalue and the eignvectors of The given covariance matrix. <br>
   (b) Generate 200 points using the distribution ```Normal(0,Cov_matrix)```. Plot the data into 2D graph, Also project the Covariance vectors on it.
  $$
  \begin{equation}
   Covariance\_matrix =
    \begin{bmatrix}
      64 & -25 \\
      -25 & 64
    \end{bmatrix}
   \end{equation}
$$
- **Question no5:** <br>
   (a) Using the two i.i.d ```X1```,```X2``` random variables with distribution of uniform(0,1) with feature sizes of 2, Given ```Y=X1+X2``` Calculate ```P(Y)``` and ```P(X1|Y)``` and also plot the density functions of thoes. <br>
   (b) Re-calculate part a, using ```X1```,```X2``` with the distribution Normal(0,1).
- **Question no8:**
    Plot the distributions of two function ```1 / sqrt(20 * np.pi) * exp( -1 * x**2 / 20 )``` and ```1 / sqrt(12 * np.pi) * exp(-1 * (x-6)**2 / 12)```, And also find the regions for w1 and w2 class in the plot.
    
 ## HW2 Codes description
 - **decision boundary example:** Create a program that gets the covariance and mean of multiple distributions, Plot the contours of data and its decision boundary.
 - **Question no1:** 
   Plot the probability of x given θ using the formula below. <br>

   $$
   \begin{equation}
    p(x|\theta) = 
    \begin{cases}
     \theta e^{{-\theta  x}},& \text{if } x\geq 0\\
     0,      & \text{otherwise}
    \end{cases}
   \end{equation}
   $$
   Then Plot the $ p(x|\theta) $ versus $\theta (0 \leq \theta \leq 5)$ for $x = 2$. 
   <br>
   At last mark the maximum likelihood value in the plot.
## HW3 description

 # Notes:
 - i.i.d stands for independece and identical distribution
