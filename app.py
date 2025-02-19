import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import streamlit as st
import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time

st.set_page_config(page_title='Stock Price Predictor', page_icon=":bar_chart:")
st.title('Stock Price Predictor :bar_chart:')

#st.write("Stock symbols can found on Yahoo Finance [here](https://finance.yahoo.com/)")
stock_name = st.text_input('Enter the Stock ID (eg, GOOG)', "GOOG")

col1, col2, col3 = st.columns([3,1,3])



with col2:
    submit_button = st.button('Submit')


st.text('')

if submit_button:
    if stock_name:
        try:
            end_date = datetime.now()
            start_date = '2010-01-01'
            df = yf.download(stock_name, start=start_date, end=end_date)
            if df.empty:
                st.error("No data found for the given stock ID, Please check the ID you've entered")
            else:

                try:
                    model = keras.models.load_model('keras_model.keras')
                except Exception as e:
                    st.error("Failed to load the model : \n",e)
                    st.stop()
                
                with st.spinner('Processing data and predicting prices...'):
                    #Preprocessing Steps
                    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
                    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])

                    scaler = MinMaxScaler(feature_range=(0,1))
                    training_fitted = scaler.fit_transform(data_training)


                    x = []
                    y = []
                    for i in range(100, training_fitted.shape[0]):
                        x.append(training_fitted[i-100:i])
                        y.append(training_fitted[i][0])

                    x = np.array(x)
                    y = np.array(y)

                    past_100_days = data_training.tail(100)
                    testing_data = pd.concat([past_100_days, data_testing], ignore_index=False)

                    testing_data_fitted = scaler.fit_transform(testing_data)

                    x_test = []
                    y_test = []

                    for i in range(100, testing_data_fitted.shape[0]):
                        x_test.append(testing_data_fitted[i-100:i])
                        y_test.append(testing_data_fitted[i][0])


                    x_test = np.array(x_test)
                    y_test = np.array(y_test)

                    #Prediction
                    y_pred = model.predict(x_test)

                    y_predicted_transformed = scaler.inverse_transform(y_pred.reshape(-1, 1))
                    y_test_transformed = scaler.inverse_transform(y_test.reshape(-1, 1))


                st.toast("Prediction complete!")

                st.subheader('Stock Data')
                st.dataframe(df, width=750)
                #st.write(df)

                st.text('')
                st.text('')

                st.subheader('Stock Price Over Time')
                fig = plt.figure(figsize=(20,10))
                plt.plot(df['Close'])
                st.pyplot(fig=fig)

                st.text('')
                st.text('')

                #Plot 100 days and 200 days Moving Average
                st.subheader('Stock Price with 100-Day & 200-Day MA')
                ma_100_days = df['Close'].rolling(100).mean()
                ma_200_days = df['Close'].rolling(200).mean()
                fig = plt.figure(figsize=(20,10))
                plt.plot(df['Close'], label='Stock Price')
                plt.plot(ma_100_days, 'g-', label='MA 100 day')
                plt.plot(ma_200_days, 'r-', label='MA 200 day')
                plt.legend(loc='lower right')
                st.pyplot(fig=fig)

                st.text('')
                st.text('')

                #Predicted Results
                st.subheader(':red[Actual Closing Price vs. Predicted Closing Price]')
                fig = plt.figure(figsize=(20, 10))
                plt.plot(y_test_transformed, 'g-', label='Actual Price' )
                plt.plot(y_predicted_transformed, 'r-', label='Predicted Price')
                plt.legend(loc='lower right')
                st.pyplot(fig=fig)

        
        except Exception as e:
            st.error("An exception occurred. Please see the exception details : \n",e)
    
    else:
        st.error('Please enter a valid stock ID')