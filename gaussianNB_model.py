from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

app = FastAPI()

#To define request body, need to create a class which inherits BaseModel and defines the features as attributes of that class along with their type hints (float etc)

class request_body(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float


#Loading Iris Dataset
iris = load_iris() #Loading the iris dataset from sci-kit learn to classify iris flowers

X = iris.data
Y = iris.target
Z = iris.target_names

print(X)
print(Y)
print(Z)



#Fitting our Model on the dataset
clf = GaussianNB() #Gaussian Naive Bayes model
clf.fit(X,Y)

#print(clf)

#####created model, now can use it to make predictions#####

#Need to define our request body: data sent from API to the client (a client is a user program that connects to a server to access a service)
#Using BaseModel (part of the pydantic module) to define the format of the data to send to the API

#Creating an endpoint that will predict the class and return it as a response
@app.post('/predict')

def predict(data : request_body):
    #Making the data in a form suitable for prediction
    test_data = [[
        data.sepal_length, 
        data.sepal_width, 
        data.petal_length, 
        data.petal_width 
]]

    #Predicting the class
    class_idx = clf.predict(test_data)[0]
    print (class_idx)
    #Return the result
    return { 'class' : iris.target_names[class_idx]}
