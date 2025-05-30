# Supervised Machine Learning: Regression and Classification

## Supervised Learning

- x → y mappings
- you give examples to learn from - the correct labels - **labeled** **data**
- **regression** & **classification** problems
    - Regression: infinite number of possible outputs
    - Classification: finite number of possible outputs

## Unsupervised Learning

- x only
- find interesting structures / patterns in the input **unlabeled** **data**
- **Clustering** is one example, e.g. google news that groups related news together
- **Anomaly** **detection** is another example, e.g. detect fraud in financial system
- **Dimensionality reduction** is a third example, i.e. compressing data using fewer numbers

## Terminology

- Training set
- x → input variable / **feature**
- y → output variable / **target**
- $\hat{y}$ (“y-hat”) → **prediction**
- m → number of training examples
- $(\hat{x}^{(i)}, \hat{y}^{(i)})$ → i-th training example
- Linear regression with one variable or univariate linear regression

$$
f_{w,b}(x^{(i)})=\hat{y}^{(i)}=wx^{(i)}+b
$$

## Cost function

Squared error cost function

$J(w,b)=\frac{1}{2m} \sum(\hat{y}^{(i)} - y^{(i)})^2$

We want to minimize J at the cost of w,b

## Gradient Descent
Pseudo-code:

- Start with some w,b

- Keep changing w and b to reduce J(w,b), by taking small steps in the direction that yields the steepest descent

- Until we settle at or near a minimum

The learning rate alpha determines the size of the step. We update both parameters simultaneously.

$$
w = w - \alpha \dfrac{\partial{J(w,b)}}{\partial{w}}=w-\alpha \dfrac{1}{m}\sum(f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}
$$

$$
b = b - \alpha \dfrac{\partial{J(w,b)}}{\partial{b}}=b-\alpha \dfrac{1}{m}\sum(f_{w,b}(x^{(i)})-y^{(i)}
$$

If learning rate is too small, you decrease the cost super slowly, and you need a lot of steps. 

If learning rate it too large, gradient descent may overshoot or fail to converge. 

If you’re at a local minimum, gradient descent does nothing else.

The distance between steps shrinks as the gradient descent approaches zero, since the derivatives decrease fast.

For linear regression, the squared error cost is always bowl shaped surface and has only one minimum. In more complex problems, you can find local minimum instead of global minimum. 

“Batch” gradient descent - each update step uses all the training examples.

## Multiple linear regression
Not the same as multivariate linear regression

$$
f_{\vec{w},b}(\vec{x})=\vec{w}\cdot\vec{x}+b
$$

Vectorization with numpy: `f = np.dot(w,x)+b`. Runs faster than summing over all components, since it parallelizes computations. GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel.


**Gradient descent with multiple features**

$$
w_j = w_j - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{w_j}}=w_j-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}
$$

$$
b = b - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{b}}=b-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
$$

Normal equation
- Only works for linear regression
- Solve for w,b without iterations
- Slow if number of features is large (>10,000)
- Some machine learning libraries might use this to implement linear regression but gradient descent is the recommended method for solving it.

## Feature Scaling

If a feature has a large range of values, the final fitted parameter ends up being small compared to other features. Small changes of the parameter mean a large change on the estimated variable, if the range of values for the feature is large. The contour plots for the gradient descent end up being too tall and skinny (kind of an ellipsis), narrow on one feature and wide on the other. The gradient descent may end bouncing back and forth before finding the minimum. We need to scale the features, so all features have comparable ranges of values and gradient descent converges quickly. 

Aim for about -1 to 1.

Mean normalization

$$x = \frac{x-\mu}{x_{max}-x_{min}}$$

Z-score normalization

$$x = \frac{x-\mu}{\sigma}$$

Learning curve: Graph of cost function vs # of iterations. It's helpful to see if gradient descent is working perfectly, and check for convergence as well. We can also declare converge if cost function decreases by less than some threshold at one iteration.

### Choosing an appropriate learning rate

If cost function is oscillating with # of iterations, the learning rate is too large.
With small enough learning rate, the cost function should decrease at every iteration.

### Feature engineering & Polynomial regression

Feature scaling is paramount here

## Classification

### Logistical Regression


$$f_{\vec w, b} = g(\vec w \cdot \vec x + b)$$

where the sigmoid function

$$g(z)=\frac{1}{1+e^{-z}}, 0<g(z)<1$$

Idea is that it means the probability that the output is 1. 

Decision boundary - would yield probability 0.5
$$z = \vec w \cdot \vec x + b = 0$$

Squared error cost function applied to logistic regression would be non convex, which means there'd be a lot of local minima to apply gradient descent. ]

Loss function

$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)})=-log(f_{\vec w, b}(\vec x^{(i)}),y^{(i)}=1$$

$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)})=-log(1-f_{\vec w, b}(\vec x^{(i)}),y^{(i)}=0$$

Or equivalently

$$L(f_{\vec w, b}(\vec x^{(i)}), y^{(i)})=-y^{(i)}log(f_{\vec w, b}(\vec x^{(i)})-(1-y^{(i)})log(1-f_{\vec w, b}(\vec x^{(i)})$$

Cost function

$$-\frac{1}{m}\sum [y^{(i)}log(f_{\vec w, b}(\vec x^{(i)})+(1-y^{(i)})log(1-f_{\vec w, b}(\vec x^{(i)})]$$

Gradient descent

$$
w_j = w_j - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{w_j}}=w_j-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}
$$

$$
b = b - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{b}}=b-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
$$

Looks the same as the linear regression equations, but the definition of f(x) is different

The problem with overfitting

Undefit - high bias

Overfit - high variance

### Addressing Overfitting

- Collect more training examples
- Feature selection - select features to include/exclude - some info gets lost
- Reduce size of parameters $w_j$ - Regularization - we keep all features but they never dominate the prediction, it's like having a simpler smoother model at the end

### Cost function with regularization

$\lambda$ is the regularization parameter, >0. We don't penalize the parameter b, since it makes little difference.

$$J(w,b)=\frac{1}{2m} \sum(f_{\vec w, b}(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m}\sum w_j^2$$

### Regularized linear regression
$$
w_j = w_j - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{w_j}}=w_j-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}w_j
$$

$$
b = b - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{b}}=b-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
$$

### Regularized logistic regression

$$J(\vec w,b)=-\frac{1}{m}\sum [y^{(i)}log(f_{\vec w, b}(\vec x^{(i)})+(1-y^{(i)})log(1-f_{\vec w, b}(\vec x^{(i)})]+\frac{\lambda}{2m}\sum w_j^2$$

$$
w_j = w_j - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{w_j}}=w_j-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})x_j^{(i)}+\frac{\lambda}{m}w_j
$$

$$
b = b - \alpha \dfrac{\partial{J(\vec{w},b)}}{\partial{b}}=b-\alpha \dfrac{1}{m}\sum(f_{\vec{w},b}(\vec{x}^{(i)})-y^{(i)})
$$
