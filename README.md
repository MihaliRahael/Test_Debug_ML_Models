**Testing & Debugging of ML Systems**

**Model testing**

![image](https://user-images.githubusercontent.com/106816732/213533777-3da22816-1ac8-4d96-8ce5-b1fbb13fb759.png)

![image](https://user-images.githubusercontent.com/106816732/213533806-111e344a-7e68-491c-91c8-5682c44381b3.png)

Suppose in the dataset feature f1 is an ordinal feature like day of week. So we will be giving 1-7 for each day. So we can give a simple sanity check by using group by operation and check whether labels are only ranges from 1-7. We can do this test for every ordinal features. Skew check can be done using ‘group by count’. If we have more data on sun and less data on Friday then model will perform better in Sundays. We should be aware of this fact.

Another feature which is categorical not ordered, like state. Some label would be AP or Andrapradesh etc. So we need to arrange everything in proper format. We need to understand the distribution of categories to understand skewness. Suppose more data are there for Maharastra, bihar and UP and less data on Sikkim, kerala, then this distributional skew shows model performs better on former states than later.

![image](https://user-images.githubusercontent.com/106816732/213533919-a55d5846-ac8c-4f0c-84bb-eb60748923db.png)

![image](https://user-images.githubusercontent.com/106816732/213533943-bf52fa3a-dd14-4eca-8580-8fdc48e2dd30.png)

In English we know that power law distribution of words, we can check the data whether its true. Another issue is misspellings, we can use the dictionary and test the words which are not part of the dict.

![image](https://user-images.githubusercontent.com/106816732/213533974-df133a68-5f36-4ad6-bd92-615f0814fc4a.png)

This is one of the major issue. Splitting the data for train, CV and test. We should ensure that train set is so different from test. Otherwise the model will show high performance . We should do a statistical difference test. How?

![image](https://user-images.githubusercontent.com/106816732/213533996-4085cafa-fb41-4151-9561-520da25539d1.png)

Take an example, we need to know check whether training data is similar how different from cv data. Procedure is to combine the original xi and yi together to create xi’ and yi’ be 1 for training data. Similarly combine xi and yi of cv together to create new xi’ and yi’ be 0. Now we need to build a classification model like RF which can classify a pair of xi and yi belong to training data or cv data. If it classify very efficiently then both datas are well separated.

![image](https://user-images.githubusercontent.com/106816732/213534023-f8482910-e54f-4ebe-84af-ca12dc9eba27.png)

Next aspect is while doing feature transforms. Say while we do data normalization by substracting mean and divide by variance we should make sure that variance is not zero. Then division by zero can mess up the whole model itself. Zero variance is possible if all xi’s of particular feature are same. As a remedy either we add a small value to variance or avoid the feature where variance is zero cos that feature is irrelevant. Similar to div-by zero there are others problems like log(0), log(-ve) etc while we encounter numerical features, which we need to take care.

Another important thing is while a data is missing , doing imputation. we need to make sure that imputation is happening in equal amount on class zero and class 1

![image](https://user-images.githubusercontent.com/106816732/213534052-d6cfe424-e448-40fe-be25-d24dfba26f34.png)

While we plot the histogram and detect some outliers we can clip them since the presence of outliers may worsen the performance. We learned this in gradient clipping in DL. Next is when we do one hot encoding just do a sanity check whether only one category has assigned ‘1’ (say for C1C2C3C4C5 00100). Next thing is normalization and standardization. After doing these preprocessing just check whether mean =0 and var =1 if normalization.

![image](https://user-images.githubusercontent.com/106816732/213534080-f756f220-71eb-4ed7-b8f8-d2e05533d3f1.png)

We need to make sure our word vectorizer is working well by giving some words and check the similar and dissimilar words . In BOW , length of vector should be equal to \# of words.

**Model Debugging**

![image](https://user-images.githubusercontent.com/106816732/213534119-ed26e805-6ec6-4d00-800b-a2c5b8a7a654.png)

Always do univariate analysis which helps in identifying best and worst features and discuss about these features with domain expert. Using these best features design a baseline model like linear or logistic regression. Using a metric we will have a performance figure of this model. So after this any complex model’s metric should be greater than this baseline model. Otherwise our new model is useless. Second thing is suppose we are creating a regression baseline model and metric is MAPE (Mean absolute percentage error) and say model gives a 10% of MAPE. Means our new complex model should give a value \<10% . Also In case of classification problem we usually creates random –model with metric of AUC and in case of regression we use mean-model (which returns just mean of all numericals; which will be the worst possible result.)

![image](https://user-images.githubusercontent.com/106816732/213534227-7ce3e51f-ba1b-46aa-9f63-7a7822c2140c.png)

Say we have an imbalanced dataset of 90% of zeros and remaining 1s. Then one of the very basic model is to return a dominant class where it always return class 0 and then we can measure the metric. Similarly we can use random model where we have roughly 50-50 split of class label. By adopting dominant class or random model strategy we can know how worse a model can be. After then we build a simple model like LR using the best univariate feature. Then go for all complex models. Performance metric value should be improved each stages.

![image](https://user-images.githubusercontent.com/106816732/213534287-78560dad-b115-44f7-bddd-50e2365f5cb4.png)

Take some ideal datapoints; say for classification. Implement it without using any regularizer ie lambda=0.Then check whether its overfitting. Similarly in case of a simple MLP, give constant weight of 0 or 1 and check the output. We know what the output already if we give this input. We can easily find the output using a pen and paper. These are sanity check to check whether the code implementation is correct. Similarly in gradient based problems always check the gradients whether the value is decaying.

![image](https://user-images.githubusercontent.com/106816732/213534323-4f2c6323-ca96-4352-a6cd-b6699f324b1e.png)

Once we checked without regularization slowly increase regularization by increasing data and lambda value. In case of DL increase dropouts. Plot loss vs epochs. And confirm that we are underfitting the model with large regularization and overfitting with low reg. In the plots of loss vs epochs or iterations in SGD based models (LR,Log reg, SVM, DL etc) ; we can see a sudden drop at particular epochs. This drop confirms that the learning rate changes as expected. After each SGD step learning rate will reduce which will make model to converge more to the dip.

Another observation in optimization based systems is as shown below. We can see an oscillation of CV loss after particular epoch. This is because in optimization the value is not reaching the actual minima and its oscillating back and forth around the minima. In that case we can understand we need to reduce learning rate.

![image](https://user-images.githubusercontent.com/106816732/213534366-1b228044-fc54-4d4b-a0bc-0553bcd3ebe7.png)

![image](https://user-images.githubusercontent.com/106816732/213534396-cf265aff-a2cd-472e-bee0-9b223b5643c4.png)

This type of oscillation is also common which occurs if we doesn’t handle NaN properly or div-by zero or exploding gradient etc.

![image](https://user-images.githubusercontent.com/106816732/213534425-da96a616-cc31-4074-a311-df1b498da664.png)

This is important. Usually what happens is in each epochs, we take random subset of data, update the weights and go on. i.e we are sending mini batches. Issue is whatever the batch which increases the loss, batch which decreases the loss, the same batches are repeating. The solution to this is simple. After every epoch, randomly shuffle the whole data once.

![image](https://user-images.githubusercontent.com/106816732/213534453-93988c84-5e46-4738-84d4-fdb4cf5ab9ce.png)

Unit test means , take a unit of code basically a fn or a class , test it. For a sigmoid fn, give an input of zero, a large value and a small value. We know the output . Check whether the fn is performing as expected. Same incase of derivative sigmoid, Relu, maxpool etc. If we are implementing a max pool layer give a simple 3X3 input of maximum value 8 , the test should give back 8.

!![image](https://user-images.githubusercontent.com/106816732/213534500-7779376d-a79c-488b-ba7b-e31c5abd5eb2.png)

This we have learned . In deep NN, check the distribution of input to the 10th layer or so. If there is consistant change in it, then add batch normalization layer infront.

Solution of div-by zero issue where adding a small value to variance , is also done in batch normalization layer.

![image](https://user-images.githubusercontent.com/106816732/213534527-c308c110-7353-4492-bc25-cbf8a55c7cf9.png)

![image](https://user-images.githubusercontent.com/106816732/213534560-929daba7-8527-42a1-a5ba-741757f41dcd.png)

Sometimes for say in case of CNNs, we might wont get a predicted output label as expected. We might need to find the reason why the model didn’t classify it correctly and explain. The reason may lead to change the dataset or preprocessing. In the screenshot above this, barbell detection, eventhough barbell is correctly spotted the probability score of 0.447 indicates a dataset issue, not model issue. Because in actual training data there is no gym image with white or plain background. Training datasets were images captured from real gym itself. That’s why image 1 and 3 gives good probability score. This is one of the most adopted method to win kaggle solutions and build better models in production. Train a model and see where it fails and see how we can change the dataset so that we can significantly reduce the errors . Sometimes we might need to weigh more some features cos a particular feature is not getting enough contribution. Similarly change the feature transform or change loss function itself. Eg: if the model is failing cos of lot of outliers, use a loss fn which can handle outliers better.This testing is so hard which can take days or months.

![image](https://user-images.githubusercontent.com/106816732/213534598-c53f607e-6fc9-4700-88b7-a6862e7573d0.png)

In this,say log loss is the cv loss which is the model performance whereas recall is the business metric. We can see that log loss is reducing which is good , but recall is almost zero. Why ? It can be the classification model with 0 &1. If all the values i.e P(yi\|xi)\<0.5 for some reason and if we set the threshold of a class label belonging to class 1 is strictly of 0.5, the recall will be zero. This threshold issue is due to data imbalance. This can be rectified by either change the classification threshold or recaliberate the model using platt scaling or isotonic regression. Another approach is using AUC is allowed or use multiple but sensible metrics to debug.

![image](https://user-images.githubusercontent.com/106816732/213534649-31756330-3ce5-415c-aa69-b450df4adc55.png)

![image](https://user-images.githubusercontent.com/106816732/213534671-37450e3a-c60d-41a5-8abd-4ded58b53b7a.png)

![image](https://user-images.githubusercontent.com/106816732/213534707-125e517e-11b6-48b5-8719-136b1f3e04c1.png)

In productionization, say I designed my model through an API (in a computer of having 4 cores perse) which means the user sends me an xi, model return the yi. Simple way of performance and load testing is write a simple code in one computer, which sends 1 request ie 1 xi at a time and waits for the reply and measure the latency for 1 QPS (query per second) and check whether under the permissible time limit. Next send 10 QPS. X1 x2 x3 …. HTTP server takes each xi and process. Calculate the avg time taken for the results. This is load testing ie measuring how much load this model can take on condition of return time. Saying we need result in 100ms. And we found out 200 QPS is the limit. Plot we make is QPS Vs 99 percentile of latency (ie 99 percentage of inputs returns). Sending this query is easy using a for loop sending http request. We can use multithreaded or multicore code if we want.

![image](https://user-images.githubusercontent.com/106816732/213534747-576d4adc-315c-40d4-85ff-3589726b0756.png)

A/B testing is for ensuring the efficiency of new model.

![image](https://user-images.githubusercontent.com/106816732/213534789-e01d9f8b-c22b-433b-893b-b3fb2eb5c318.png)

Another imp aspect is in model maintenance. Everytime when we retrain our model we have Dtrain_old and Dtrain_new. We can check whether both training data are significantly different using statistical differences. In production pipelines this is part of daily testing mechanism, done every night. What happens is probably there is some error in new training data, somewhere something in the data pipeline broke. That data can be from multiple sources, join them using query, anything can break in between. If there is significant difference we can raise issue and say we are not deploying the new model.
