## 0. Objective:

In this project we are going to develop a model to predict if a person is diabetic or not, based on attributes of Pima-Indians diabetes data from UCI repository. Since, our target is a binary decision (Yes or No), we are going to use supervised classification algorithms.
Based on online studies and papers, we are targeting to achieve 75+ % or above model classification accuracy (https://www.ijedr.org/papers/IJEDR1703069.pdf)  with minimum false negative (High Recall) of the model. 

Minimum False Negative:  So that we should not miss any patient with a diabetic. For this classification we can tolerate some false positive(Patients misclassified as diabetic). If we mis-classify a non-diabetic person as a diabetic then he/she will go for further testing where they will get to know that they are non-diabetic.

**Data source:** 

[Pima Indians Diabetes](https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes)

**Attribute Information:**

   1. Pregnancies : Number of times pregnant - 
   2. Glucose : Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
   3. Blood Pressure: Diastolic blood pressure (mm Hg)
   4. SkinThickness : Triceps skin fold thickness (mm) 
   5. Insulin : 2-Hour serum insulin (mu U/ml)
   6. BMI : Body mass index (weight in kg/(height in m)^2)
   7. DiabetesPedigreeFunction : Diabetes pedigree function 
   8. Age : Age

**Target** 9. data.columns(0 or 1) - "class_variable" 1 is yes and 0 is no.*

**Python Library Used:**

- numpy
- pandas
- seaborn
- matplotlib.pyplot
- plotly
- time

**Visualizations**
![1](https://github.com/kwankhede/Diabetes_Prediction/blob/master/1.png)
![2](https://github.com/kwankhede/Diabetes_Prediction/blob/master/2.png)



**Conclusion**

In this study, we investigate various classification algorithms to get best prediction accuracy with minimum false negative instances of diabetes. We explored various data-preprocessing techniques to find the best results. 
After all the activities we performed, we came to know that Extreme Gradient Boosting Classifier and Gradient Boosting classifier are the best performing algorithms on this dataset. So we would recommend to use these model for predicting diabetes. 

**Scope for further investigation**

As we have not used all the available machine learning classification algorithms and related techniques, there are strong possibilities to get better results. I would also recommend to explore deep learning to see if we can get better results.

