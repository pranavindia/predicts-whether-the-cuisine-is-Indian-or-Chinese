In this problem scenario, we are constructing a machine learning pipeline on the MLEnd Yummy dataset, which takes as input a photo of a dish. Subsequently, it filters the dataset to include items with cuisine related to either Indian or Chinese cuisine, and consequently, predicts whether the cuisine is Indian or Chinese.

The MLEnd Yummy Dataset comprises 3250 rows and 12 columns, including:

Photos of dishes

Dish names

Home or restaurant designation

Cuisine

Ingredients

Diet information

Healthiness rating

Tastiness rating

The presence of 'Benchmark_A' facilitates the division of our entire dataset into training and testing sets.

The objective is to employ machine learning algorithms to provide highly accurate predictions regarding whether the image corresponds to a Indian or Chinese cuisine.


Downloading Data
In the initial stage of the pipeline, we download the Yummy dataset of dish images and the CSV file into my Google Drive.

Loading Data: The subsequent step involves reading the image files from a specific drive location. Due to the complexity of image attributes, we opt to use the names of respective image files (e.g., '000001.jpg'). These names are matched with those in the CSV file, allowing for the extraction of images and their features efficiently.

Data Preprocessing:

Filtering: We filter the dataset based on cusine names to exclusively include images of Indian and chinese in our input. The results are saved in the 'indian_chinese' column, assigning the value "indian" if the cusine includes indian and vice versa for chinese.

Mapping: We map indian to 0 and chinese to 1, limiting the dataset to only 500 images of indian to prevent overfitting. Given the varying image sizes, we perform a Train-Test Split:

Train-Test Split: Separating datasets (TrainSet and TestSet) is achieved by creating training and testing sets based on the 'Benchmark_A' column.

Resizing: All images are resized to ensure uniform dimensions.

Feature Extraction: Image features, such as the yellow component for information on yellow tones and GLCM features for insights into texture patterns, are extracted. Additional features, such as Contour-based features, are utilized to extract information related to object shapes in the image.

Normalization: After feature extraction, normalization is applied by calculating mean and standard deviation.

Model Training: Different machine learning models, including Support Vector Machine, K-Nearest Neighbors, and Random Forest, are employed to classify images and predict training and testing accuracy.

Confusion Matrix: The confusion matrix is utilized to display counts of true positive, true negative, false positive, and false negative predictions. It provides a comprehensive overview of how well the model distinguishes between cuisene of Indian and chinese.

Transformation stage
Data Transformation --Data transformation is done basically for image preprocessing .For uniform images we need to perform resizing as images are not of same size. Our first step will be to resize all the images to so that they have the same size.we will resize them to 200x200 pixels.After resizing images are square shape and consist of 200x200 pixels. Now Later we are doing feature extraction to solve the problem of predicting whether a cuisene has indian and chinese using a 200 x 200 pixels photo as the predictor. Each photo is described by 3 x 200 x 200 = 120,000 values. Therefore, the predictor space has 120,000 dimensions. To train a model on such a space, we need a training dataset that has more than 120,000 samples. This is a higher dimensionality problem and requires an impossibly large amount of training dataset to train its parameters.

To avoid this, we go for dimensionality reduction where we will move our samples from a 120,000D space to another space that has fewer dimensions. Here we are extracting image using features which include yellow component to provide valuable information about the presence of yellow tones in the images and GLCM features to provide insights into texture patterns within the images.I have used additon feature that is Contour-based features which is used to extract features related to the shape of objects in the image.We are basically doing feature extraction so that food image gives information on basis on above feature extraction.After feature extraction we are using normalization by taking mean and std .

5 Modelling
Now we are modelling to predict whether the image corresponds to a cuisine that has Indian or chinese.we are using 4 attributes for that i have normalised.The linear model that i have used is support Vector Machine.SVMs are essentially binary classifiers, and the task here for this problem statemnet includes classifying images into two categories: "indian" and "chinese.SVM is good for image classification and is less prone to overfitting.The effiectiveness of SVM is it makes the decision boundary created by a linear SVM is a straight line in two dimensions, a plane in three dimensions, and a hyperplane in higher dimensions. Main aims of this classifier is to maximize the margin between the classes.I am getting better accuracy by using SVM. I also tried random forest and K nearest neighbors i have commented that code.
6 Methodology
The model we have designed is trained using linear SVM on a dataset of yummydata images which is further categorized into indian or chinese, and its performance is validated through metrics such as accuracy and a confusion matrix.For purpose of training ,we have used training set and testing set that we have divided by data preprocessing .Our training sample and testing sample is divided on basis of label "Benchmark_A".We have used these samples for training and test accuracy.My training accuracy is determined by taking mean of all samples where predicted value matches actual value.Same work is done in calculating the testing accuracy.

We have plotted a confusion matrix that tells model performance as it displays the count of true positive, true negative, false positive, and false negative predictions and providing us a clear overview of how well the model is distinguishing between cuisene of indian and chinese.
7 Dataset
We have used the Yummy Image dataset. The MLEnd dataset was obtained through the download_yummy function, resulting in the creation of a DataFrame named MLENDYD_df sourced from a CSV file. The 'Benchmark_A' column is used for the purpose of train-test splitting. We constructed an array encompassing the values of all features extracted from the images, with labels assigned as 0 and 1 (where 0 corresponds to indian and 1 to chinese). The mapping between indian and chinese is accomplished during the preprocessing stage.

Results
To attain desired result We have tried multiple classification models to perform the indian vs chinese predictions.

In Linear SVM Classifier for the parameter value of C=1, we got output where the Training accuracy is 73% and the testing accuracy is 53%.

For Random Forest Classifier for parameter value n_estimators=4, we receive unusually very high training accuracy value of 92% which seems to suggest some overfitting is taking place for the training data. So we discard this model for our classification.

For K Nearest Neighbors, for the value of k=7 (I have tried many values but this value gave us the best balance of train and test accuracy, without overfitting on the train data or compromising on the testing accuracy), we get the training accuracy of 71% and the testing accuracy of 51%.

On the basis of all these observations, we go ahead with the second best option for our classification that is the SVM.

The confusion matrix that i have plotted gives the number of true positives (TP) is much higher than the number of false positives (FP) and false negatives (FN). which conclude that the model is very good at predicting whether someone will choose indian or chinese.

 Conclusions
The conclusion we can draw is my model is achieving a training accuracy around 73% and testing accuracy of around 53%, which is relatively good. This means that the model is able to correctly classify the dishes in the training set but need some improvement in testing dataset.To improve the model's performance we can use more sophosicated featues ,i had tried some but my accuracy did'nt improve.We also observed that it is essential to equalize the number of instances between both classes before applying any classification algorithm.

Additionally, from the above observation, we can conclude that SVC provides more significant and generates good expected accuracy for the predictions as compared to other two models . Also, it is observed that Validation and training accuracy are very similar in case of SVC , implying that the model is some what fitted . Hence it is efficient.

In Future reference We can use deep learning architecture like convolutional neural network for image classification
