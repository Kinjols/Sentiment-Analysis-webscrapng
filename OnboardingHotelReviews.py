import pandas as pd
import plotly.express as px

hotels = pd.read_csv('Individual assignment/Hotel_Reviews.csv')

hotels.head(10)
hotels.describe()
hotels.dtypes

hotels.isnull().sum()

fig = px.histogram(hotels, x='Reviewer_Nationality')
fig.show()

fig = px.histogram(hotels.head(20), x='Reviewer_Nationality')
fig.show()


fig = px.histogram(hotels.head(10), x='Reviewer_Nationality')
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


import plotly.express as px
fig = px.histogram(hotels, x="Review_Total_Negative_Word_Counts")
fig.show()

(
    hotels[['Review_Total_Negative_Word_Counts','Negative_Review','Reviewer_Score']]
    .loc[lambda df: df['Review_Total_Negative_Word_Counts'] == 2]
    .head(5)
)

hotelsfiltered=hotels[(hotels['Review_Total_Negative_Word_Counts']>=5) & (hotels['Review_Total_Positive_Word_Counts']>=5)]

# to a new db

from sqlalchemy import create_engine

# create db first in MySQL
engine = create_engine('mysql+mysqlconnector://root:root@localhost/hotels')

hotelsfiltered.to_sql(name='hotelreviewscleaned',con=engine,if_exists='fail',index=False, chunksize=1000) 


