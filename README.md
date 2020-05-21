# heart.io-backend

[![Devpost | heart.io](https://badges.devpost-shields.com/get-badge?name=heart.io&id=heart-io&type=big-logo&style=flat)](https://devpost.com/software/heart-io)

![heart.io Screenshots](https://www.stewartdulaney.com/wp-content/uploads/sites/7/2019/01/heart.io_.gif "heart.io")

## Inspiration

Not everyone has access to the best doctors. In fact, 20% of Americans go without any healthcare at all. However, 96% of Americans have access to an internet connected device. As a team, we recognize this disparity in access to medical advice and as a result with our application heart.io we intend to give access to an expert opinion that some would otherwise never get.

## What it does

By integrating convolutional neural networks (CNN) with a user friendly web application, heart.io delivers a personalized health analysis in less than one second. This is achieved by having users simply upload one image file of a suspected carcinogenic blemish (i.e. a mole, bodily discoloration, etc.).  Give it a try! 
 
## How we built it

By utilizing HAM10000, a dataset published by the University of Vienna consisting of imageset of size 10,015. We use a Tesla K80 GPU provided by the Google Colaboratory service to train a CNN model for use in the diagnosis potentially carcinogenic blemishes. 
The steps involved in the model creation include:
* Feature Engineering - Discerning what features could be dropped without sacrificing predictive power.
* Label Binarization - Involved binarization of the categorical variable shown in the HAM10000 dataset.
* Cross Validation - Ensures that model performance is not solely based on one experiment but a robust series of tests.
* Normalization of Data - We normalize the imageset in order to reduce training time and general runtime, and to grant a better relation between our training and test set.
* Data Augmentation - We use Data Augmentation in order to make the most of the HAM10000 dataset.
* Hyper parameters - we optimized our model accuracy by iterating over hyper parameters and testing various NN architectures.

After starting with a Flask back end, we decided to move it to Google Cloud’s serverless platform in order to take advantage of the ability to instantly scale. Deploying cloud services at the level of a single function using Google Cloud Functions will allow us to easily add back end functionality for additional diseases in our plans to expand the heart.io database to be the image recognition equivalent of WebMD.

## Challenges we ran into
Due to computing constraints we were unable to utilize transfer learning. We struggled to match the high resolution of images needed for many pre-trained models such as ResNet50 & MobileNet. We attempted to bypass this using batch sizing, for loops and ample usage of both the Google Cloud Platform and local systems in order to take advantage of faster write speeds. When that was finally solved, no time was left for iterating and optimizing the accuracy.

Preprocessing large datasets took longer than expected even on solid state drives. We gathered X-ray data that we would’ve liked to utilize but time constraints did not allow that.

We initially were considering to deploy our backend on Google’s Kubernetes Engine for the scalability. However, due to the Python 3.7 restriction (Keras and Tensorflow only support up to Python 3.6) we had difficulties in implementing an end-to-end solution using Kubernetes, we tried once again to deploy our backend to GCP’s serverless cloud functions and eventually were successful. 


## Accomplishments that we're proud of
We were able to adapt our website to have an optimal user experience. Many people decide to forego visiting a healthcare professional simply due to inconvenience. In order to overcome this hurdle we delivered a pleasant user experience and deliver easy-to-understand results in a timely manner to users.

We were proud to be able to unify various branches of computer science that are not typically integrated into a single project. With Machine Learning being such a new field, we were proud to learn more about this exciting field.

## What we Learned
As the creators of a skin cancer detection program, we learned how to take our skills in deep/machine learning and utilize them in a full end to end solution. Many members of our team were highly specialized, but were able to learn more about other roles involved in the development of a web app. We improved our understanding of the importance of the preprocessing step, and how it can greatly affect model accuracy.

We gained a better understanding of how our datasets can affect our model's accuracy, and probability of overfitting, and the importance of having diverse data points. In regards to website deployment, we learned how to efficiently utilize Flask in conjunction with Keras/Tensorflow.

## What's next for heart.io

After seeing our success with 6 diseases. We will be adding more diseases for our platform to diagnose. We understand the importance of one’s own health awareness, to that end we are committed to setting up a one stop shop for accurate diagnoses. 

It sometimes takes one gentle push for a person to change their lifestyle for the better. In the spirit of self improvement, we want heart.io to do just that.

## Check it out
- [https://2b1s.github.io/heart.io-frontend/](https://2b1s.github.io/heart.io-frontend/)
- [https://devpost.com/software/heart-io](https://devpost.com/software/heart-io)
- [https://github.com/2B1S/heart.io-frontend](https://github.com/2B1S/heart.io-frontend)
- [https://github.com/2B1S/heart.io-backend](https://github.com/2B1S/heart.io-backend)
- [https://www.youtube.com/watch?v=QKcd7p2ow-Y](https://www.youtube.com/watch?v=QKcd7p2ow-Y)

## Contributors
- [sdulaney](https://github.com/sdulaney)
- [oshtontsen](https://github.com/oshtontsen)
- [tejashah88](https://github.com/tejashah88)
- [edmondnemsingh](https://github.com/edmondnemsingh)
- [ohadmich](https://github.com/ohadmich)
