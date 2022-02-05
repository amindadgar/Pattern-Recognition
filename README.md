# pattern-recognition-homework
This codes belong to my pattern recognition Homeworks.

## HW1 Codes description
- **Question no2:**<br>
  Generate 100000, n dimension uniform i.i.d. random variables and calculate the volume of the n dimension shape of it.
  *Answer:* 50, 100, 200 and 1000 dimension iid uniform random variables was created. The graphs were plotted. 
- **Question no3:** <br>
  calculate the probability functions, <img src="https://latex.codecogs.com/svg.image?P(Y)"/>, <img src="https://latex.codecogs.com/svg.image?P(Y|X)" /> using the joint probability function <img src="https://latex.codecogs.com/svg.image?P(X,Y)" />. Also plot the density functions of thoes probability. <br>
  

 
   <img src="https://render.githubusercontent.com/render/math?math=p(x,y) = \frac{1}{2\pi ab} \exp({-(\frac{y-\mu}{2a^2} + \frac{(x-y)^2}{2b^2})}) ">
 
- **Question no4:** <br>
   (a) Calculate the eignvalue and the eignvectors of The given covariance matrix. <br>
   (b) Generate 200 points using the distribution ```Normal(0,Cov_matrix)```. Plot the data into 2D graph, Also project the Covariance vectors on it.

<!--   <img src="https://latex.codecogs.com/svg.image?\begin{equation}Covariance=\begin{bmatrix} 64 & -25\\ -25 & 64 \end{bmatrix}\end{equation}" /> -->
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\begin{equation}Covariance=\begin{bmatrix}&space;64&space;&&space;-25\\&space;-25&space;&&space;64&space;\end{bmatrix}\end{equation}" title="\begin{equation}Covariance=\begin{bmatrix} 64 & -25\\ -25 & 64 \end{bmatrix}\end{equation}" />

- **Question no5:** <br>
   (a) Using the two i.i.d X1,X2 random variables with distribution of uniform(0,1) with feature sizes of 2, Given Y=X1+X2 Calculate <img src="https://latex.codecogs.com/svg.image?P(Y)"/> and <img src="https://latex.codecogs.com/svg.image?P(X1|Y)"/> and also plot the density functions of thoes. <br>
   (b) Re-calculate part a, using X1,X2 with the distribution Normal(0,1).
- **Question no8:**
    Plot the distributions of two function <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{\sqrt{20\pi}} \e^{- x^2 / 20}"> and <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{sqrt(12 \pi)} e^{- (x-6)^2 / 12}">, And also find the regions for w1 and w2 class in the plot.
    
 ## HW2 Codes description
 - **decision boundary example:** Create a program that gets the covariance and mean of multiple distributions, Plot the contours of data and its decision boundary.
 - **Question no1:** 
   Plot the probability of x given θ using the formula below. <br>
  P(X|\theta) = if (x >= 0 ) then \theta e ^(-thetax) else 0 <br>
   Then Plot the <img src="https://latex.codecogs.com/png.image?p(x|\theta)" /> versus <img src="https://render.githubusercontent.com/render/math?math=\theta (0 \leq \theta \leq 5)"> for <img src="https://render.githubusercontent.com/render/math?math=x=2">. 
   <br>
   At last mark the maximum likelihood value in the plot.
## HW3 description
- **Question no1:** <br>
   *(a)* Implement KNN algorithm for [toy dataset](https://github.com/amindadgar/pattern-recognition-homework/tree/main/HW3/toy%20dataset), and report the results of cpu and memory usage. Also create the confusion matrix of the results. <br>
   *(b)* We want to find the suitable K for I dataset, So from each class seperate 100 data for validation and 900 for training. At last compare the results of each K.
- **Question no2:** <br>
   *(a)* Using the bayes classifier find the optimal class for each one of the data. The parameters of each class is given in [Q2](https://github.com/amindadgar/pattern-recognition-homework/blob/main/HW3/Q2/Q2_main.ipynb). <br>
   *(b)* check if the covariance was isotropic, how the result would change. <br>
   *(c)* compare the results with knn results in question 1.
- **Question no3:** <br>
   *(a)* Using the numbers dataset, classify classes with KNN algorithm. <br>
   *(b)* Again Apply KNN algorithm and use 10 precent of each class for validation set and other for training.
- **Question no4:** <br>
   Apply the bayes classifier for each number image like question 2. The main difference here is the parameters aren't given, So compute them using a method such as Maximum Likelihood.

## HW4
In this homework use SVM and PCA models on MNIST dataset. More detailed problem is in the jupyter files.

 # Notes:
 - i.i.d stands for independece and identical distribution
