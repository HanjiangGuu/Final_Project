#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:28:23 2021

@author: guhanjiang
"""

import streamlit as st
import pandas as pd
import altair as alt
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

st.title("House Price in Bodega Bay Area with respect to multi-variables")

st.write("Hanjiang Gu's Math 10 Final Project")
st.write("Student ID#:48061816")
st.write("Data_Link: https://drive.google.com/file/d/1brl70j18M8fBJ7EBdZmtmF6Sp0XXvH6H/view?usp=sharing")

st.title("Part 1:Read & Filter the Data.Keep the numeric columns.")
df = pd.read_csv("houseprice.csv",na_values = " ")

df2 = pd.DataFrame(df, columns = ["PRICE","SQUARE FEET","BEDS","BATHS"])

df3 = df2[df2.notna().all(axis=1)].copy()
df3

# Part 1: Basic Altair
st.write("Based on the numeric data, I use altair to show the relationship between different variables.")

st.write("Chart 1: Set # of beds as x-axis, # of bathrooms as y-axis, pirce as color, square feet as size. In such a way, we could better visualize how does # of bathroom and beds affect the hose price.")
st.write("By Altair, it shows the most expansive house is equipped with 4 beds and 3.5 bathrooms, the size is about standard. Also, 4 beds seems like a threshold of luxury house.")
chart1 = alt.Chart(df3).mark_circle().encode(
            x ="BEDS",
            y = "BATHS",
            color = 'PRICE',
            size = 'SQUARE FEET',
            tooltip = ['PRICE','SQUARE FEET','BEDS','BATHS'])

st.altair_chart(chart1)


st.write("Chart 2: Set Sqft as x-axis, price as y-axis,# of baths as color, # of beds as size. In such a way, we could better visualize the relationship between House sqft & Price")
st.write("By Altair, it's obvious that the smaller house always come with low price. Most house area is from 1,000 to 2,500 sqft. Some larger houses are also in low price, which reveals that house area is not the only factor affects house price, but in general, larger house has higher price.")
chart2 = alt.Chart(df3).mark_circle().encode(
        x ="SQUARE FEET",
        y = "PRICE",
        color = 'BATHS',
        size = 'BEDS',
        tooltip = ['PRICE','SQUARE FEET','BEDS','BATHS'])

st.altair_chart(chart2)


#Part 2: Train the Data & Plot
st.title("Part 2: Try to Train the Data & Plot")

st.write("Here, I train the data. The smaller the learning rate, the more accurate the result, the slower the training model. As the learning rate increases, the iteratino increases as well.")
st.write("I use keras training to make a linear regression like prediction model. The idea situation is to use sqft, beds, baths to predict the house pirce.(ex: when we have a new house, input:sqft & beds & baths, and use model.predict() to predict the house price.)") 
X_train = df3.drop("PRICE", axis=1)
y_train = df3["PRICE"]


model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (3,)),
        keras.layers.Dense(10, activation="sigmoid"),
        keras.layers.Dense(10, activation="sigmoid"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.SGD(learning_rate=0.001),
    #metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=100, validation_split = 0.2, verbose=False)

fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')

st.pyplot(fig)

model.summary()

st.write("Usually, as # of training increases, the loss would decrease, and the model would be more accurate. However, the loss is still huge in this case, which means keras may not fit the research question.")
st.write("Therefore, it would be better to use linear regression.")

# Part 3: Scikit-Learn:Linear Regression 

st.title("Part 3:Scikit-Learn:Linear Regression")
st.write("The coding work is in the streamlit.")
st.write("Since Keras does not well, so I try scikit_learn. The folloing chart is 3 coefficients, and the single number is the intercept.")

reg = LinearRegression()
reg.fit(X_train,y_train)
coef= reg.coef_
intercept = reg.intercept_
st.write(coef)
intercept
st.write("Then we make bulid the following function by defination:")
st.write("Price = intercept + coef[0] × x1 + coef[1] × x2 + coef[2] × x3")
st.write("(Note: x1 represents sqft, x2 represents beds, x3 represents baths)")

# Part 4: KMEANS & Cluster & Plot 
st.title("Part 4: Kmeans & Cluster & Plot")
st.write("In order to idnetify the house type(low, middle, high), I use Kmeans to divide the house into 3 groups. For the new data frame, I add a new column called cluster range from 0-2. The order of number does not matter, it's used to classify the groups, and we could observe the result by graph." )
kmeans = KMeans(3)
kmeans.fit(df3)
kmeans.predict(df3)
df3["cluster"]=kmeans.predict(df3)
df3

st.write("Plot out the Kmeans result. It's obvious that the houses are divided into 3 levels since we set the degree of Kmeans as 3. By such graph, we could know how many houses are considered as low level, middle level, and high level.")
chart3 = alt.Chart(df3).mark_circle().encode(
    x = "PRICE",
    y = "SQUARE FEET",
    color = "cluster:O"
)
st.altair_chart(chart3)

st.write("To better visualize the graph, I use 3 different colors, Blue,Orange,and Red. Here, we could easily see how does the house price distribute with respect to house area, and  ")
chart4 = alt.Chart(df3).mark_circle().encode(
    x = "PRICE",
    y = "SQUARE FEET",
    color = "cluster:N"
)
st.altair_chart(chart4)


#Part 5: StandradScaler the Data
# Use StandradScaler to nrmalize the data

st.title("Part 5:Use StandardScaler & Normalize Data")
st.write("It's nature to use Standard Scaler in this situation. I bulid a new chart by Standard Scaler, and then plot the new data.")

scaler = StandardScaler()
scaler.fit(df3)
df4 = scaler.transform(df3)
df5 = pd.DataFrame(df4)
df5.columns = ["PRICE",'SQUARE FEET','BEDS',"BATHS",'cluster']
df5

st.write("StandardScaler helps remove the possible error values. It resize the distribution of values. This chart is mostly the same as before, but the axises are standarlized. ")

chart5 = alt.Chart(df5).mark_circle().encode(
    x = "PRICE",
    y = "SQUARE FEET",
    color = "cluster:N"
)
st.altair_chart(chart5)
