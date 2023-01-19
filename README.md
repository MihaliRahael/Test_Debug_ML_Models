**Testing & Debugging of ML Systems**

**Model testing**

![](media/e03c515cc71e91f86328c746f69f2fa5.png)

![](media/6dacba5c67b8dc945273f5d802c376bd.png)

Suppose in the dataset feature f1 is an ordinal feature like day of week. So we will be giving 1-7 for each day. So we can give a simple sanity check by using group by operation and check whether labels are only ranges from 1-7. We can do this test for every ordinal features. Skew check can be done using ‘group by count’. If we have more data on sun and less data on Friday then model will perform better in Sundays. We should be aware of this fact.

Another feature which is categorical not ordered, like state. Some label would be AP or Andrapradesh etc. So we need to arrange everything in proper format. We need to understand the distribution of categories to understand skewness. Suppose more data are there for Maharastra, bihar and UP and less data on Sikkim, kerala, then this distributional skew shows model performs better on former states than later.

![](media/66291ba3436e519ca94ce014d7b25bb8.png)

![](media/fb6343e758d403c996a91863e0b7bc88.png)

In English we know that power law distribution of words, we can check the data whether its true. Another issue is misspellings, we can use the dictionary and test the words which are not part of the dict.

![](media/226ea75b99f9409eedfbe90439104848.png)

This is one of the major issue. Splitting the data for train, CV and test. We should ensure that train set is so different from test. Otherwise the model will show high performance . We should do a statistical difference test. How?

![](media/441f61a9fa8f9687e164e5df9d61fee5.png)

Take an example, we need to know check whether training data is similar how different from cv data. Procedure is to combine the original xi and yi together to create xi’ and yi’ be 1 for training data. Similarly combine xi and yi of cv together to create new xi’ and yi’ be 0. Now we need to build a classification model like RF which can classify a pair of xi and yi belong to training data or cv data. If it classify very efficiently then both datas are well separated.

![](media/cfcdf4127e41650edb903405e7059ed7.png)

Next aspect is while doing feature transforms. Say while we do data normalization by substracting mean and divide by variance we should make sure that variance is not zero. Then division by zero can mess up the whole model itself. Zero variance is possible if all xi’s of particular feature are same. As a remedy either we add a small value to variance or avoid the feature where variance is zero cos that feature is irrelevant. Similar to div-by zero there are others problems like log(0), log(-ve) etc while we encounter numerical features, which we need to take care.

Another important thing is while a data is missing , doing imputation. we need to make sure that imputation is happening in equal amount on class zero and class 1

![](media/c13be40a000250d416a0f910c421f120.png)

While we plot the histogram and detect some outliers we can clip them since the presence of outliers may worsen the performance. We learned this in gradient clipping in DL. Next is when we do one hot encoding just do a sanity check whether only one category has assigned ‘1’ (say for C1C2C3C4C5 00100). Next thing is normalization and standardization. After doing these preprocessing just check whether mean =0 and var =1 if normalization.

![](media/0093def3da9ce7fa4fec383001e94619.png)

We need to make sure our word vectorizer is working well by giving some words and check the similar and dissimilar words . In BOW , length of vector should be equal to \# of words.

**Model Debugging**

![](media/ab303e644dbcc0d40d275e6fb72cb494.png)

Always do univariate analysis which helps in identifying best and worst features and discuss about these features with domain expert. Using these best features design a baseline model like linear or logistic regression. Using a metric we will have a performance figure of this model. So after this any complex model’s metric should be greater than this baseline model. Otherwise our new model is useless. Second thing is suppose we are creating a regression baseline model and metric is MAPE (Mean absolute percentage error) and say model gives a 10% of MAPE. Means our new complex model should give a value \<10% . Also In case of classification problem we usually creates random –model with metric of AUC and in case of regression we use mean-model (which returns just mean of all numericals; which will be the worst possible result.)

![](media/bdb6fd0a048871c001dbcde29b279c78.png)

Say we have an imbalanced dataset of 90% of zeros and remaining 1s. Then one of the very basic model is to return a dominant class where it always return class 0 and then we can measure the metric. Similarly we can use random model where we have roughly 50-50 split of class label. By adopting dominant class or random model strategy we can know how worse a model can be. After then we build a simple model like LR using the best univariate feature. Then go for all complex models. Performance metric value should be improved each stages.

![](media/81e832caf869448c77d21e84f284c783.png)

Take some ideal datapoints; say for classification. Implement it without using any regularizer ie lambda=0.Then check whether its overfitting. Similarly in case of a simple MLP, give constant weight of 0 or 1 and check the output. We know what the output already if we give this input. We can easily find the output using a pen and paper. These are sanity check to check whether the code implementation is correct. Similarly in gradient based problems always check the gradients whether the value is decaying.

![](media/22f73db18eb5c862e23e6fb83118fb87.png)

Once we checked without regularization slowly increase regularization by increasing data and lambda value. In case of DL increase dropouts. Plot loss vs epochs. And confirm that we are underfitting the model with large regularization and overfitting with low reg. In the plots of loss vs epochs or iterations in SGD based models (LR,Log reg, SVM, DL etc) ; we can see a sudden drop at particular epochs. This drop confirms that the learning rate changes as expected. After each SGD step learning rate will reduce which will make model to converge more to the dip.

Another observation in optimization based systems is as shown below. We can see an oscillation of CV loss after particular epoch. This is because in optimization the value is not reaching the actual minima and its oscillating back and forth around the minima. In that case we can understand we need to reduce learning rate.

![](media/276d9ffed9c9be042041daa095b9fc6c.png)

![](media/411a516d39b9621e5c99e45d71963d41.png)

This type of oscillation is also common which occurs if we doesn’t handle NaN properly or div-by zero or exploding gradient etc.

![](media/468a9161c5e58d38297a4b786214694a.png)

This is important. Usually what happens is in each epochs, we take random subset of data, update the weights and go on. i.e we are sending mini batches. Issue is whatever the batch which increases the loss, batch which decreases the loss, the same batches are repeating. The solution to this is simple. After every epoch, randomly shuffle the whole data once.

![](media/c0b29658f8382ae683d00d278a3a8cb1.png)

Unit test means , take a unit of code basically a fn or a class , test it. For a sigmoid fn, give an input of zero, a large value and a small value. We know the output . Check whether the fn is performing as expected. Same incase of derivative sigmoid, Relu, maxpool etc. If we are implementing a max pool layer give a simple 3X3 input of maximum value 8 , the test should give back 8.

![](media/7719650ac2e3bb73f5ea01432b8ad7df.png)

This we have learned . In deep NN, check the distribution of input to the 10th layer or so. If there is consistant change in it, then add batch normalization layer infront.

Solution of div-by zero issue where adding a small value to variance , is also done in batch normalization layer.

![](media/c9b3b5c95880a524fd0e48df7c67efce.png)

![](media/f45d21403df1cd2eb4fb17ffa3a136a5.png)

Sometimes for say in case of CNNs, we might wont get a predicted output label as expected. We might need to find the reason why the model didn’t classify it correctly and explain. The reason may lead to change the dataset or preprocessing. In the screenshot above this, barbell detection, eventhough barbell is correctly spotted the probability score of 0.447 indicates a dataset issue, not model issue. Because in actual training data there is no gym image with white or plain background. Training datasets were images captured from real gym itself. That’s why image 1 and 3 gives good probability score. This is one of the most adopted method to win kaggle solutions and build better models in production. Train a model and see where it fails and see how we can change the dataset so that we can significantly reduce the errors . Sometimes we might need to weigh more some features cos a particular feature is not getting enough contribution. Similarly change the feature transform or change loss function itself. Eg: if the model is failing cos of lot of outliers, use a loss fn which can handle outliers better.This testing is so hard which can take days or months.

![](media/6f7f0554bbe032147027157f072b31c5.png)

In this,say log loss is the cv loss which is the model performance whereas recall is the business metric. We can see that log loss is reducing which is good , but recall is almost zero. Why ? It can be the classification model with 0 &1. If all the values i.e P(yi\|xi)\<0.5 for some reason and if we set the threshold of a class label belonging to class 1 is strictly of 0.5, the recall will be zero. This threshold issue is due to data imbalance. This can be rectified by either change the classification threshold or recaliberate the model using platt scaling or isotonic regression. Another approach is using AUC is allowed or use multiple but sensible metrics to debug.

![](media/20251887920565740a33ab3c81387d3c.png)

![](media/f96168e251c0dd0624deb120f05278fb.png)

![](media/b26365a7dceb879cf82bd5d5b6ca9d1e.png)

In productionization, say I designed my model through an API (in a computer of having 4 cores perse) which means the user sends me an xi, model return the yi. Simple way of performance and load testing is write a simple code in one computer, which sends 1 request ie 1 xi at a time and waits for the reply and measure the latency for 1 QPS (query per second) and check whether under the permissible time limit. Next send 10 QPS. X1 x2 x3 …. HTTP server takes each xi and process. Calculate the avg time taken for the results. This is load testing ie measuring how much load this model can take on condition of return time. Saying we need result in 100ms. And we found out 200 QPS is the limit. Plot we make is QPS Vs 99 percentile of latency (ie 99 percentage of inputs returns). Sending this query is easy using a for loop sending http request. We can use multithreaded or multicore code if we want.

![](media/9bd753a86c64089a60beadeac30fe971.png)

A/B testing is for ensuring the efficiency of new model.

![](media/71b56e5e55b3112ccb377c15d430467f.png)

Another imp aspect is in model maintenance. Everytime when we retrain our model we have Dtrain_old and Dtrain_new. We can check whether both training data are significantly different using statistical differences. In production pipelines this is part of daily testing mechanism, done every night. What happens is probably there is some error in new training data, somewhere something in the data pipeline broke. That data can be from multiple sources, join them using query, anything can break in between. If there is significant difference we can raise issue and say we are not deploying the new model.
