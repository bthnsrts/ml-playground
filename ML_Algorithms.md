# ML Algorithms

> Sampling Techniques
Decision Tress
Ensemble Methods; Bagging, Random Forest
k-Nearest Neighbor
Logistic Regression
> 

# Supervised Methods

- Classification problems

Linear regression model examples

$y = B_0 + B_1*x_1 + B_2* x_2 + e$

$y = B_0 + B_1*x_1^2 + B_2* ln(x_2) + e$
where $e$ ~ N(0,1) 

coefficients $B_i$ are computed by the Least Squares Method.

Minimize the summation below for $B_i$ values.

$\sum_{i=1}^n (Y_i -Y'_i)^2$

$H_0 : B_i$  for $X_i$  = 0 {no linear relation between $X_i$  and  $Y$}.

$H_0 : B_i$  for $X_i$  ≠ 0 {linear relation between $X_i$  and  $Y$}.

p-values are independently computed for each x value after constructing an ANOVA table for the linear regression model.  If p-value is higher than the confidence level for a given $X_i$, we reject the null hypothesis and determine a significant relationship between $X_i$  and $Y$.

---

**Sampling Techniques ————-**

- Validation Set Approach
    
    to partition the dataset into two tables; train and test sets.
    

It is not possible to deteriorate the $R^2$ value for the model as one keeps adding new data. However, addition of much data results in *OVERFITTING*

![Screen-Shot-2017-07-25-at-3.55.30-PM.png](ML%20Algorithms/Screen-Shot-2017-07-25-at-3.55.30-PM.png)

- LOOCV (Leave-one-out-Cross-validation)
    
     to train the model on the dataset excluding one $(x_i,y_i)$ pair and test the model on the excluded data. Repeat this for n many times where n is the number of samples. Then take the average of the error rates.
    
- K-fold CV
    
    to split the data into k partitions and train the model on the data excluding the kth partition. Test the model on this excluded data and repeat this for k times with different test data. Then take the average of the error rates.
    

**Bias-Variance Trade Off**

 → Bias :  LOOCV < k-fold CV < Validation Set Approach

 → Variance : Validation Set Approach < k-fold CV < LOOCV

High bias : Underfitting the data so that model curve is too simple.

High variance : Overfitting the data so that model curve is too complex.

Classification model performance measures:

![1_NhPwqJdAyHWllpeHAqrL_g.png](ML%20Algorithms/1_NhPwqJdAyHWllpeHAqrL_g.png)

**Sensitivity or Recall** : How accurate at classifying actual positives? TP/(TP+FN) 

**Specificity** : How accurate at classifying actual negatives? TN/(TN+FP) 

**Precision** :  How precise are positive predictions? TP/(TP+FP)

---

    **Categorical values,**

**Nominal categorical** → [blue, red, brunette], [yes, no] etc.

**Ordinal categorical** → [low, medium, high], [poor, average, rich] etc.

When you use dummy variables for nominal categorical values (turning one column of n categorical variables into n number of columns with 1/0 as entries), it is sensible to use n-1 columns assuming one of the variables is the default status.

---

---

                                                         ~ **Decision Trees ~**

**Recursive binary splitting** : It starts at the top of the tree (at which point all observations belong to a single region) and then successively splits the predictor space; each split is indicated via two new branches further down on the tree. It is a greedy approach because at each step of the tree-building process, the best split is made at that particular step, rather than looking ahead and picking a split that will lead to a better tree in some future step.

**Classification Trees**

**classification error rate** is used as a criterion for making binary splits, however; it is not sufficiently sensitive for tree-growing, and in practice two other measures are preferable. Classification error rate is simply the fraction of the training observations in that region that do not belong to the most common class.

Each node in a classification tree (or any other tree), has both **gini index** and **entropy index** which shows the impurity created by each split. Gini index is simply the probability of misclassification for a selected class in a node with the range [0, 0.5]. Entropy is the expected value of surprise in a node with the range [0,1].

![0_GLIGkgB1AG4BAXJa.png](ML%20Algorithms/0_GLIGkgB1AG4BAXJa.png)

![IkBzK.png](ML%20Algorithms/IkBzK.png)

Now, we’d like to calculate the expected Gini value based on the outlook split for each node we construct and the compare the gini index with other alternatives.

$E(GI_{outlook}) = GI_{overcast-node} * (12/22) + GI_{sunny-node} * (5/22) + GI_{rain-node} * (5/22)$

where $GI_{overcast-node} = 1 - ((4/5)^2 + (1/5)^2 )$ for instance.

After choosing the best split, continue with that split towards the second layer.

Decision Tree might eliminate / ignore some of the features you give to the model.

      **Regression Trees**

Regression trees predicts using the average value of all the samples in a node and the error rate is calculated simply by MSE (mean squared errors).

In order to choose the best splitting input attribute for a regression tree, two variables are adjusted. The first, which attribute to use (by comparing total MSE created by that split) the second, which numerical value to use for splitting (by evaluating each distinct numerical value in that column).

A tree has several parameters:

```
max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it has at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.
```

**Cost complexity pruning:**

min (# of missclassifications) + $a$ * (# of leaf nodes)  for classification

or

min (mean squared prediction error) + $a$ * (# of leaf nodes) for regression

$a =$  cost complexity parameter

selection of $a$  value shows the trade-off between the depth and missclassifications.

---

---

**Ensemble Methods: Bagging, Random Forest**

---

**Bagging (Bootstrapping + Aggregating)**

1. Split the data set using one of the resampling approaches
2. Bootstrap training data for many times to construct different trees. Bootstrapping involves sampling with replacement, so in overall, %33 of the rows are not included in the training data for each tree.
3. Test all of the trees on the test set and take the average for regression, take the majority vote for classification.

Instead of using the test set, we might use **Out-Of-Bag** samples. For each tree, there are rows not included in the training set. We can take the average prediction/majority vote for each out of bag row using the trees that did not use that row in their training set. 

overall OOB MSE (for a regression problem) or classification error (for a classification problem) can be computed.

---

**Random Forest**

As in bagging, we build a number forest of decision trees on bootstrapped training samples. But when building these decision trees, each time a split in a tree is considered, a random sample of m predictors is chosen as split candidates from the full set of p predictors.

typically we choose m ≈ √p

**Random forest decreases the variability compared to bagging, but increase the bias.**

- **Feature Importance in trees**

one can obtain an overall summary of the importance of
each predictor using the SSE (for bagging regression trees) or the Gini index (for bagging classification trees). In the context of bagging classification trees, we can add up the total amount that the Gini index is decreased by splits over a given predictor, averaged over all B trees.

---

---

**K - Nearest Neighbor**

KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closest to the test data. The KNN algorithm uses the majority vote to classify using the classes of ‘K’ training data. In the case of regression, the value is the mean of the ‘K’ selected training points.

Distance is calculated by the Euclidean Distance. if K is high, the model is too strict, underfitting might occur. If K is too low, the model is highly flexible, overfitting might occur.

- Scale the numerical attributes before computing distances.

$$
X_i' = (X_i -min(X))/(max(X)-min(X))
$$

- Convert ordinal categorical variables into numerical values by [1,2,…n] according to their natural order. Then scale them as above.
- For any nominal categorical variable. Distance between different classes is 1, distance between same is 0.

![1_OVf0u6wAV6RJg23N2We9TQ.jpg](ML%20Algorithms/1_OVf0u6wAV6RJg23N2We9TQ.jpg)

---

---

---

**Logistic Regression**

models the probability for Y = 1, only used if target variable is categorical.

$$
P(Y=1) = f(z) = 1/(1+e^{-z})   
$$

$$
z = B_0 + B_1*X_1 +B_2*X_2
$$

f(z) is called the sigmoid function. 

- Binary logistic regression requires the dependent variable to be binary.
- For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.

$$
odd (Y=1) = P(Y=1)/P(Y = 0) = e^{B_0 + B_1*X_1 + B_2*X_2}
$$

$$
log(odd(Y=1)) = B_0 + B_1*X_1 + B_2*X_2 
$$

Logistic regression finds coefficients such that multiplication of P(Y=1 | Y_actual = 1) * P(Y=0 | Y_actual = 0) over all target variables is maximized.

P(Y=1) = 0.6 is found after logistic regression, now, how to classify?

We should choose a probability threshold to classify as 1.

*if* we choose 0.6 as the threshold, and P(Y=1) < 0.6, $Y_{predicted} = 0$ 

Receiving Operator Characteristics Curve is constructed by plotting True Positive Rate (sensitivity) against False Positive Rate (1-Specificity) for each threshold.

TPR =  TP/(TP+FN),  FPR  = FP/(TN+FP)

![ROCOutside3.png](ML%20Algorithms/ROCOutside3.png)

**Multinomial logistic regression**

If target classes **> 2, then we construct K many models (K = the number of levels)**

class A, B, C 

the First model: class A vs B+C as y=1 for A

the Second model : class B vs A+C as y=1 for B

the Third model: class C vs A+B as y=1 for C

classify as A.

# Unsupervised Methods

**Clustering**

1. K-Means Clustering
- Choose a value for k, the number of clusters.
- Randomly choose K center points in the space consisting of observation points.
- for each observation, compute the distance to each center (scale the dataset first).
- Assign observation to the closest center.
- After assignments, take the mean of each cluster as new cluster center.
- Repeat the steps 3-6 until the centers do not change.
1. K-Medoids Clustering
    - Same procedure except that medians of clusters are set as centers instead of means.

**?** How to choose K

silhoutte index for observation i : $[b(i) -a(i)] /max(a(i),b(i))$

where b(i) = min(avg. distance to other clusters)

     a(i) = avg. distance between the obs. i and other observations within the cluster.

avg. silhouette width = weighted average of cluster silhouettes

cluster silhouette = avg. of observation silhouettes in a cluster ****

![unnamed-chunk-98-1.png](ML%20Algorithms/unnamed-chunk-98-1.png)

1. Hierchical Clustering
    - Calculate the distances between each pair of observations
    - Cluster the two with the least distance
    - Calculate distances again, this time regard the cluster as a single observation.
    - Construct the dendogram as you move along the procedure
        
        ![1_00DlhAct31DXkMh_iAQIoQ.png](ML%20Algorithms/1_00DlhAct31DXkMh_iAQIoQ.png)
        
        Distance between cluster measurements:
        
        Avg. linkage = avearge of all pairwise distances between clusters
        
        Centroid linkage = distance between means
        
        Complete linkage = Distance between the farthest points of two clusters
        
        Single linkage = Distance between the closest points of two clusters