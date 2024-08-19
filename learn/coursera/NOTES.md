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
