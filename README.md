# Face-Recognition
1-This is a face-recognition project using K-Nearest Neighbours algorithm.
### KNN ALGORITHM
-KNN basically find the Euclidean distance between the query point and all the training points.
-Euclidean Distance between two points ((x,y),(a,b)) is given as  = √(x - a)² + (y - b)²
-Then using these distances we find the k nearest points to the query point.
-And then we classify the query point to a class whose points are more closer to query point.
-There is no time for training. So order for training is O(1).
-All the time required is to find the distance of query point from all the points. So if there are N points and Q query points , then total time complexity of this algorithm will be O(NQ).
###### Example
Suppose Q is a query point and there are two classes namely 0 and 1.
a,b,c,d and e are five nearest points to Q. (a,0),(b,1),(c,1),(d,0),(e,1).This means 'a' belongs to class 0 and 'b' belongs to class 1 and so on.
Since 3 points from class 1 are closer to Q while 2 from class 0. So it is clear that points from class 1 are more closer hence the probability of Q belonging to class 1 is more.
Hence Q belongs to class 1.
This is how KNN Algorithm classifies.

2-There are two python script.
-One is for generating the training data by capturing the images.-
-Second one is written to predict the name of the person by using KNN.-

### Haarcascade Classifier
In this a predefined classifier is used. Its name is Haarcascade Classifier. In this it has been used to read the face.
Here is the link for full documentation of Haarcascade.  "https://docs.opencv.org/3.4.3/d7/d8b/tutorial_py_face_detection.html"


3 In this all the images are 3 layered (BGR - Blue Green Red).  So if we store this image as an numpy array , it will be a 3D array.
  This 3D array is flatten into a single row array so that Euclidean distance can be calculated.
4 After this using KNN, predicition is made. 
