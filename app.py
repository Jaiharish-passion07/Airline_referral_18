import streamlit as st
import numpy as np
import data_preprocess as dt
import model_call
#setting title
st.title('Airline Passenger Referral Prediction')

#ratings data of all features
seat_comfort=[1.0, 2.0, 3.0, 4.0, 5.0]
cabin_service=[1.0, 2.0, 3.0, 4.0, 5.0]
food_bev=[1.0, 2.0, 3.0, 4.0, 5.0]
entertainment=[1.0, 2.0, 3.0, 4.0, 5.0]
ground_service=[1.0, 2.0, 3.0, 4.0, 5.0]
ratings_features=[]

col1, col2 = st.columns(2)

with col1:
    #sub header of all features ratings
    st.subheader('Seat Comfort')
    #setting slider to get input values
    seat_comfort_value = st.select_slider(
         'Select a Seat comfort ratings',
         options=seat_comfort)
    #storing the values in list
    ratings_features.append(seat_comfort_value)

    st.subheader('Cabin Service')
    #setting slider to get input values
    cabin_service_value = st.select_slider(
         'Select a Cabin Service ratings',
         options=cabin_service)
    #storing the values in list
    ratings_features.append(cabin_service_value)

with col2:
    st.subheader('Food Beverages')
    #setting slider to get input values
    food_bev_value = st.select_slider(
         'Select a Food Beverage ratings',
         options=food_bev)
    #storing the values in list
    ratings_features.append(food_bev_value)

    st.subheader('Entertainment')
    #setting slider to get input values
    entertainment_value = st.select_slider(
         'Select a Entertainment ratings',
         options=entertainment)
    #storing the values in list
    ratings_features.append(entertainment_value)


st.subheader('Ground Service')
#setting slider to get input values
ground_service_value = st.select_slider(
     'Select a Ground service ratings',
     options=entertainment)
#storing the values in list
ratings_features.append(ground_service_value)

#grab categorical data
st.sidebar.subheader('Travelling type')
option_1 = st.sidebar.selectbox(
     'Select the Travelling type you prefered',dt.x_)


st.sidebar.subheader('Cabin type')
option_2 = st.sidebar.selectbox(
     'Select the Cabin type you prefered',dt.y_)

datas=dt.encdoding_data(a=option_1, b=option_2)

ratings_features.extend(map(int,datas))
user_data=np.array(ratings_features,dtype=float)
#st.write("This is My user_ratings",user_data)

#button to predict the result
if st.button('Predict'):
     #predict the user_data 
     result=model_call.grad_boost_model.predict([user_data])
     if result[0]==1:
         st.subheader("Yes,The Passenger will Recommend")
         st.balloons()
     else:
         st.subheader("The Passenger will Not Recommend")
else:
     pass
