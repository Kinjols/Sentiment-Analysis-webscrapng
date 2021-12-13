import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# creating a handwritten dataframe
# List1  
Name = ['tom', 'krish', 'arun', 'juli']  
  
# List2  
review = [95, 63, 54, 47]

# list 3
rating = [95, 63, 54, 47]

sentiment = [0, 1, 1, 0]

source=['handwritten', 'handwritten', 'handwritten', 'handwritten']  
#  two lists.  
# and merge them by using zip().  
handwritten_reviews = list(zip(Name, review, rating, sentiment, source))  
  
# Assign data to tuples.  
print(handwritten_reviews)  
  
# Converting lists of tuples into  
# pandas Dataframe.  
dframe = pd.DataFrame(handwritten_reviews, columns=['title', 'review', 'rating', 'sentiment', 'source'])  
  
# Print data.  
print(dframe)  

# readig the csv files and renaming some collumns to match with the other DF's
keagle_positive_reviews_df = pd.read_csv('Hotel_Reviews.csv',sep=',').rename(columns={"Positive_Review":"review","Hotel_Name":"title","Reviewer_Score":"rating"})
keagle_negative_reviews_df = pd.read_csv('Hotel_Reviews.csv',sep=',').rename(columns={"Negative_Review":"review","Hotel_Name":"title","Reviewer_Score":"rating"})
trip_advisor_reviews_df = pd.read_csv('trip_advisor_review.csv',sep=',')
booking_reviews_df = pd.read_csv('booking_reviews.csv',sep=',')

# adding collumns with correct sentiment
trip_advisor_reviews_df['sentiment'] = np.where(trip_advisor_reviews_df['rating']<35,0,1)
keagle_positive_reviews_df['sentiment'],keagle_positive_reviews_df['source']=[1,'keagle']
keagle_negative_reviews_df['sentiment'],keagle_negative_reviews_df['source']=[0,'keagle']
trip_advisor_reviews_df['source']='trip advisor'

trip_advisor_reviews_df.head(50)

# dropping collumns that are unncesary
columns_pos = ["Average_Score","Hotel_Address","Additional_Number_of_Scoring",
            "Review_Date","Reviewer_Nationality","Review_Total_Negative_Word_Counts"
            ,"Total_Number_of_Reviews","Review_Total_Positive_Word_Counts",
            "Total_Number_of_Reviews_Reviewer_Has_Given","Tags",
            "days_since_review","lat","lng","Negative_Review"]

columns_neg = ["Average_Score","Hotel_Address","Additional_Number_of_Scoring",
            "Review_Date","Reviewer_Nationality","Review_Total_Negative_Word_Counts"
            ,"Total_Number_of_Reviews","Review_Total_Positive_Word_Counts",
            "Total_Number_of_Reviews_Reviewer_Has_Given","Tags",
            "days_since_review","lat","lng","Positive_Review"]

keagle_negative_reviews_df.drop(columns_neg, inplace=True, axis=1)
keagle_positive_reviews_df.drop(columns_pos, inplace=True, axis=1)

# fusing all reviews and shuffling them
df = keagle_positive_reviews_df.append(keagle_negative_reviews_df).append(booking_reviews_df).append(trip_advisor_reviews_df).sample(frac = 1).reset_index(drop=True)
df.head(50)

# send to MySQL database
engine = create_engine('mysql+mysqlconnector://root:root@localhost/hotels')
df.to_sql(name='onboardinghotelreviews', con=engine, if_exists='fail', index=False,chunksize=1000)


