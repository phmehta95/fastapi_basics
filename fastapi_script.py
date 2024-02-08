from typing import Union #Union is the typing module
from fastapi import FastAPI #imports FastAPI class from the fastapi module
from pydantic import BaseModel #BaseModel class is used by Pydantic to define the structure of the data sent and recieved by the application
import pycaret

from pycaret.datasets import get_data
from pycaret.regression import *
#pycaret.regression -> module set up to use data for a regression analysis (predicts a continous outcome variable based on the value of one or mulptiple predictor variables (x)
data = get_data('insurance')
s = setup(data,target = "charges") #Initialises the experiment in PyCaret and controls all the preprocessing procedures -> setup function performs various preprocessing steps and preparing the data for modelling

best = compare_models()

create_api(best, 'insurance_prediction_model')


print (data)
print (best)
app = FastAPI() #creates an instance of the FastAPI class and assigns it to the variable app


#Create BaseModel class
#FastAPI will check name is a string, price is a float, and is_offer is a bool
class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/") #decorator used to define a route for the root URL of the web app

def read_root(): #read_root function os called when a GET request is made to the URL

    return {"Hello": "World"}


#Returns a dictionary with the key Hello and the value World
#When running fastapi_script:app --reload, it will show {"Hello":"World"} in the API which is up and running in the browser

@app.get("/items/{item_id}/") #decorator used to define a route for URLs that include an item ID parameter

def read_item(item_id: int, q: Union[str, None] = None):#this function returns a dictionary with the item_id and q values
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}/") #decorator defines an endpoint that takes an item_id parameter and an item parameter of type item, and returns a JSON object with the item_name and item_id values

def update_item(item_id: int, item: Item):#this function returns a dictionary with the item_id and q values - the app.put request takes multiple inputs of different data types
    return {"item_name": item.name, "item_id": item_id}

