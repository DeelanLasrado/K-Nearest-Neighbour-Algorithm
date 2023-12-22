import numpy as np
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

movies = pd.read_csv("C:\\Users\\deela\\Downloads\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\deela\\Downloads\\tmdb_5000_credits.csv")

print(movies.info())
      
      
for i,k in zip(movies.genres,range(len(movies.genres))):
  c=pd.read_json(i)
  movies.genres[k] = [c.loc[j]['name'] for j in range(len(c ))]  
  #iterates through the dictionaries within the evaluated JSON data, extracts the 'name' field from each dictionary, and creates a list of genre names.
    #this is also right
  #movies.genres[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

print(movies.genres)

# movies.keywords[0]
for i,k in zip(movies.keywords,range(len(movies.keywords))):
  movies.keywords[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

# movies.production_companies[0]
for i,k in zip(movies.production_companies,range(len(movies.production_companies))):
  movies.production_companies[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

# movies.production_countries[0]
for i,k in zip(movies.production_countries,range(len(movies.production_countries))):
  movies.production_countries[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

# movies.spoken_languages[0]
for i,k in zip(movies.spoken_languages,range(len(movies.spoken_languages))):
  movies.spoken_languages[k] = [eval(i)[j]['name'] for j in range(len(eval(i)))]

print(movies.isnull().sum())

movies.drop('homepage', axis=1, inplace=True)
movies.drop('tagline', axis=1, inplace=True)

cop = movies.copy()
print(cop.isnull().sum())
#op= runtime  2null values

print(cop[cop.runtime.isnull()])

movies.dropna(inplace=True)

print(movies.info())

# release_date -> release_day, release_month, release_year
movies['Year'] = pd.DatetimeIndex(movies.release_date).year
movies['Month'] = pd.DatetimeIndex(movies.release_date).month
movies['Day'] = pd.DatetimeIndex(movies.release_date).day

movies.drop('release_date', axis=1, inplace=True)

print(credits.columns)
#Index(['movie_id', 'title', 'cast', 'crew'], dtype='object')


m = movies.copy()
c = credits.copy()

movies = pd.merge(movies, credits, left_on = "id", right_on = "movie_id")
#left_on and right_on: These parameters specify the columns in the left (movies) and right (credits) DataFrames that serve as the merge keys.


for i, k in zip(movies.cast,range(len(movies.cast))):
  movies.cast[k]=[eval(i)[k]['name'] for k in range(len(eval(i)))]

m = movies.copy()
print(m.crew[0])

a =[]
for x, y in zip(m.crew, range(len(m.crew))):
   b = [eval(x)[w]["department"] for w in range(len(eval(x)))]
   a.append(b)

c =[]
for x, y in zip(m.crew, range(len(m.crew))):
  b = [eval(x)[w]["name"] for w in range(len(eval(x))) if eval(x)[w]["job"]=='Director']
  c.append(b)


for i, k in zip(movies.crew,range(len(movies.crew))):
  movies.crew[k]=[eval(i)[k]['name'] for k in range(len(eval(i)))]

movies['Director'] = c
movies['Department'] = a

print(movies.isnull().sum())

print(movies)

#v only need these cols
#[budget, genres, keywords, popularity, revenue, runtime, status, 
# title_x, vote_count,	Year,	Month, Day, Director, cast]