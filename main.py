import pandas as pd 
import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
from datetime import datetime
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly



def style():
    st.markdown("""<style>
    
    .css-3mnucz e16nr0p30{
        text-align: center;
        }
    
    
    
    </style>""", unsafe_allow_html=True)


class Coronavirus():
    def __init__(self):
        self.url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
        self.df = pd.read_csv(self.url)
        self.url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
        self.df_death = pd.read_csv(self.url_death)
        self.url_recov = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
        self.df_recov = pd.read_csv(self.url_recov)
        self.data = pd.DataFrame(columns = ['Date', 'Cases'])

    def create_luxembourg_cases_dataframe(self):
        self.lux_df = self.df[self.df['Country/Region'] == 'Luxembourg']
        self.lux_df = self.lux_df.drop(columns=['Lat', 'Long', 'Province/State'])
        self.date = []
        self.case = []

        for self.cases in self.lux_df.columns[1:]:
            self.case.append(self.lux_df[self.cases].values[0])
            self.date.append(self.lux_df.columns[1:].values)



        for dates in self.date:
            self.data['Date'] = dates
            self.data['Cases'] = self.case

        self.data['Date'] = pd.to_datetime(self.data['Date']).dt.strftime('%Y-%m-%d')
        self.data = self.data.set_index('Date')


    def create_luxembourg_deaths_dataframe(self):

        self.lux_death = self.df_death[self.df_death['Country/Region'] == 'Luxembourg']
        self.lux_death = self.lux_death.drop(columns=['Lat', 'Long', 'Province/State'])
        self.date_death = []
        self.case_death = []
        print(self.case_death)

        for deaths in self.lux_death.columns[1:]:
            self.case_death.append(self.lux_death[deaths].values[0])
            self.date_death.append(self.lux_death.columns[1:].values)

        self.data['Deaths'] = self.case_death


    def create_luxembourg_recoveries_dataframe(self):
        
        self.lux_recov = self.df_recov[self.df_recov['Country/Region'] == 'Luxembourg']
        self.lux_recov = self.lux_recov.drop(columns=['Lat', 'Long', 'Province/State'])
        self.date_recov = []
        self.case_recov = []
        print(self.case_recov)

        for recov in self.lux_recov.columns[1:]:
            self.case_recov.append(self.lux_recov[recov].values[0])
            self.date_recov.append(self.lux_recov.columns[1:].values)

        self.data['Recovered'] = self.case_recov

        


    
    def plot_data(self, column, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.data.index, y = self.data[column]))
        fig.layout.update(title_text = f'{title}: {self.data[column][-1]}')
        fig.update_layout(xaxis_tickformat="%b %Y")
        st.plotly_chart(fig)

    @st.cache
    def forecasting(self):
        index = self.data.reset_index()

        time_series = pd.DataFrame(columns = ['Date', 'Cases'])
        time_series['Date'] = index['Date']
        time_series['Cases'] = index['Cases']

        
        period = 2 * 365
        self.data_train = time_series[['Date', 'Cases']]
        self.data_train = self.data_train.rename(columns = {'Date': 'ds', 'Cases': 'y'})


        self.model = Prophet()
        self.model.fit(self.data_train)
        self.forecast = self.model.make_future_dataframe(periods = period)
        self.prediction = self.model.predict(self.forecast)


    
    def world_map(self):
        pass



    def main_gui(self):
        st.image('cl.jpg')
        st.title('COVID-19 LUXEMBOURG')
        st.subheader('Last 5 days')
        st.table(self.data.tail())
        self.plot_data('Cases', 'Total Cases')
        self.plot_data('Deaths', 'Total Deaths')
        self.plot_data('Recovered', 'Total Recoveries')


        st.subheader('Future Cases (Predictions)')
        fig = plot_plotly(self.model, self.prediction)
        fig.update_layout(width = 700, height = 500)
        st.plotly_chart(fig)
        st.text('Source: https://github.com/CSSEGISandData/COVID-19')

style()
app = Coronavirus()
app.create_luxembourg_cases_dataframe()
app.create_luxembourg_deaths_dataframe()
app.create_luxembourg_recoveries_dataframe()
app.forecasting()
app.main_gui()
