#!/usr/bin/env python
# coding: utf-8

# ### Team Rating

# In[1]:


import requests
import json

# API endpoint and parameters
url = "https://api-football-v1.p.rapidapi.com/v3/standings"
querystring = {"season": "2020", "league": "140"}

# Headers including the API key and host
headers = {
    "X-RapidAPI-Key": "91684b1af8mshc806588b84de4d8p11df56jsn20a49a480b92",
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# Make the request
response = requests.get(url, headers=headers, params=querystring)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON and save it to a file
    data = response.json()
    with open('standings_2020.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Data has been saved to standings.json")
else:
    print(f"Failed to retrieve data: {response.status_code}")


# In[2]:


import json
import pandas as pd

# Load the data from your JSON file
with open('standings_2017.json', 'r') as file:
    data = json.load(file)
    
import pandas as pd

# Assuming 'data' is your JSON response
standings_data = data['response'][0]['league']['standings'][0]

# Normalize the data and create a dataframe
df_17 = pd.json_normalize(standings_data)

df_17 = df_17.drop(['group', 'status', 'description', 'update', 'team.logo'], axis=1)


# In[3]:


import json
import pandas as pd

# Load the data from your JSON file
with open('standings_2018.json', 'r') as file:
    data = json.load(file)
    
import pandas as pd

# Assuming 'data' is your JSON response
standings_data = data['response'][0]['league']['standings'][0]

# Normalize the data and create a dataframe
df_18 = pd.json_normalize(standings_data)


# In[4]:


import json
import pandas as pd

# Load the data from your JSON file
with open('standings_2019.json', 'r') as file:
    data = json.load(file)
    
import pandas as pd

# Assuming 'data' is your JSON response
standings_data = data['response'][0]['league']['standings'][0]

# Normalize the data and create a dataframe
df_19 = pd.json_normalize(standings_data)


# In[5]:


import json
import pandas as pd

# Load the data from your JSON file
with open('standings_2020.json', 'r') as file:
    data = json.load(file)
    
import pandas as pd

# Assuming 'data' is your JSON response
standings_data = data['response'][0]['league']['standings'][0]

# Normalize the data and create a dataframe
df_20 = pd.json_normalize(standings_data)


# In[6]:


df_20 = df_20.drop(['group', 'status', 'description', 'update', 'team.logo'], axis=1)


# In[7]:


df17 = df_17.copy()
df18 = df_18.copy()
df19 = df_19.copy()
df20 = df_20.copy()


# In[8]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Your DataFrames are df2016, df2017, df2018, df2019
# They all have 'team', 'points', and 'goalsDiff' as columns.

# Initialize the MinMaxScaler
minmax_scaler = MinMaxScaler()

# Updated function to include year in the normalized column names
def normalize_features(df, year, scaler):
    df[[f'points_normalized_{year}', f'goalsDiff_normalized_{year}']] = scaler.fit_transform(df[['points', 'goalsDiff']])
    return df

# Apply normalization to each DataFrame with year labels
df2017 = normalize_features(df17, 2017, minmax_scaler)
df2018 = normalize_features(df18, 2018, minmax_scaler)
df2019 = normalize_features(df19, 2019, minmax_scaler)
df2020 = normalize_features(df20, 2020, minmax_scaler)

# You would continue with the rest of your code as before,
# making sure to adjust column names in the merging and weighting steps accordingly.
df2020


# In[9]:


teams_2020 = df2020['team.name'].unique()
df_2019_filtered = df2019[df2019['team.name'].isin(teams_2020)]
df_2018_filtered = df2018[df2018['team.name'].isin(teams_2020)]
df_2017_filtered = df2017[df2017['team.name'].isin(teams_2020)]


# In[10]:


df_2020_norm = df2020[['team.name', 'points_normalized_2020', 'goalsDiff_normalized_2020']]

# Selecting normalized columns for 2018
df_2019_norm = df_2019_filtered[['team.name', 'points_normalized_2019', 'goalsDiff_normalized_2019']]

# Selecting normalized columns for 2017
df_2018_norm = df_2018_filtered[['team.name', 'points_normalized_2018', 'goalsDiff_normalized_2018']]

# Selecting normalized columns for 2016
df_2017_norm = df_2017_filtered[['team.name', 'points_normalized_2017', 'goalsDiff_normalized_2017']]


# In[11]:


df_2020_norm


# In[12]:


df_2017_norm


# In[13]:


df_merged = df_2020_norm.merge(df_2019_norm, on='team.name', how='left')
df_merged = df_merged.merge(df_2018_norm, on='team.name', how='left')
df_merged = df_merged.merge(df_2017_norm, on='team.name', how='left')


# In[14]:


import pandas as pd
import numpy as np

# Assuming df_merged is your DataFrame after merging all the season data.

# Fill NaN values for teams not present in a season with 0
df_merged.fillna(0, inplace=True)

# Normalize the sums of points and goal difference within each season


# In[15]:


for year in ['2017', '2018', '2019','2020']:
    points_col = f'points_normalized_{year}'
    goaldiff_col = f'goalsDiff_normalized_{year}'
    sum_col = f'sum_{year}'
    
    # Sum the points and goal differences for the season
    df_merged[sum_col] = df_merged[points_col] + df_merged[goaldiff_col]
    
    # Normalize the sum between 0 and 1
    max_value = df_merged[sum_col].max()
    df_merged[f'norm_sum_{year}'] = df_merged[sum_col] / max_value if max_value != 0 else 0

# Apply exponential weighting
weights = [1/(2.5**i) for i in range(4)][::-1]  # Correct exponential weights

# Now calculate the weighted sums for each year
for year, weight in zip(['2017', '2018', '2019','2020'], weights):
    df_merged[f'weighted_sum_{year}'] = df_merged[f'norm_sum_{year}'] * weight

# Calculate the overall team rating
weighted_columns = [f'weighted_sum_{year}' for year in ['2017', '2018', '2019','2020']]
df_merged['team_rating'] = df_merged[weighted_columns].sum(axis=1) / sum(weights)

# Display the DataFrame with the team ratings


# In[16]:


df_merged


# In[17]:


import requests
import json

# API endpoint and parameters
url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

querystring = {"league":"140","season":"2020"}
headers = {
    "X-RapidAPI-Key": "91684b1af8mshc806588b84de4d8p11df56jsn20a49a480b92",
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# Make the request
response = requests.get(url, headers=headers, params=querystring)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response JSON and save it to a file
    data = response.json()
    with open('fixtures_2020.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Data has been saved to standings.json")
else:
    print(f"Failed to retrieve data: {response.status_code}")


# In[18]:


import pandas as pd
import json

# Load JSON data from a file
with open('fixtures_2020.json', 'r') as f:
    data = json.load(f)

# Extracting fixtures information
fixtures_data = []
for fixture in data['response']:
    # Concatenating home and away full-time goals to create the scoreline
    scoreline = str(fixture['score']['fulltime']['home']) + ' - ' + str(fixture['score']['fulltime']['away'])
    fixture_info = {
        'date': fixture['fixture']['date'],
        'home_team': fixture['teams']['home']['name'],
        'away_team': fixture['teams']['away']['name'],
        'full_time_score': scoreline  # Adding the scoreline
    }
    fixtures_data.append(fixture_info)

# Creating DataFrame
df_fixtures = pd.DataFrame(fixtures_data)


# In[19]:


df_fixtures_2020 = df_fixtures.copy()


# In[20]:


# Convert the 'date' column to datetime
df_fixtures_2020['date'] = pd.to_datetime(df_fixtures_2020['date'])
df_fixtures_2020['date'] = df_fixtures_2020['date'].dt.date
df_fixtures_2020.sort_values(by='date', inplace=True)


# In[21]:


import pandas as pd
from datetime import datetime

# Dictionary mapping matchdays to date ranges (YYYY-MM-DD format)
matchday_dict = {
    1: ("2020-09-12", "2020-09-13"),
    2: ("2020-09-19", "2020-09-20"),
    3: ("2020-09-26", "2020-09-27"),
    4: ("2020-09-29", "2020-10-01"),
    5: ("2020-10-03", "2020-10-04"),
    6: ("2020-10-17", "2020-10-18"),
    7: ("2020-10-23", "2020-10-26"),
    8: ("2020-10-30", "2020-11-02"),
    9: ("2020-11-06", "2020-11-08"),
    10: ("2020-11-20", "2020-11-23"),  # You may need to adjust this range
    11: ("2020-11-27", "2020-11-30"),
    12: ("2020-12-04", "2020-12-07"),
    13: ("2020-12-11", "2020-12-14"),
    14: ("2020-12-18", "2020-12-20"),
    15: ("2020-12-22", "2020-12-23"),
    16: ("2020-12-29", "2020-12-31"),
    17: ("2021-01-02", "2021-01-04"),
    18: ("2021-01-08", "2021-01-11"),
    19: ("2021-01-19", "2021-01-21"),
    20: ("2021-01-22", "2021-01-25"),
    21: ("2021-01-29", "2021-02-01"),
    22: ("2021-02-05", "2021-02-08"),
    23: ("2021-02-12", "2021-02-15"),
    24: ("2021-02-19", "2021-02-22"),  # This range seems incorrect, might need adjustment
    25: ("2021-02-26", "2021-03-01"),
    26: ("2021-03-05", "2021-03-08"),
    27: ("2021-03-12", "2021-03-15"),
    28: ("2021-03-19", "2021-03-21"),
    29: ("2021-04-02", "2021-04-07"),
    30: ("2021-04-09", "2021-04-12"),
    31: ("2021-04-21", "2021-04-22"),
    32: ("2021-04-24", "2021-04-26"),
    33: ("2021-04-28", "2021-04-29"),
    34: ("2021-04-30", "2021-05-03"),
    35: ("2021-05-07", "2021-05-10"),
    36: ("2021-05-11", "2021-05-13"),
    37: ("2021-05-16", "2021-05-16"),
    38: ("2021-05-21", "2021-05-23")
}


# Convert matchday_dict to a DataFrame
matchday_df = pd.DataFrame([
    {"matchday": md, "start_date": pd.Timestamp(sd), "end_date": pd.Timestamp(ed)}
    for md, (sd, ed) in matchday_dict.items()
])



# Function to find matchday for each date
def find_matchday(game_date):
    for index, row in matchday_df.iterrows():
        if row['start_date'] <= pd.Timestamp(game_date) <= row['end_date']:
            return row['matchday']
    return None

# Map each game to its corresponding matchday number
df_fixtures_2020['matchday'] = df_fixtures_2020['date'].apply(find_matchday)



# In[22]:


df_fixtures_2020


# In[23]:


df_fixtures_2020[(df_fixtures_2020['date'] == '2021-04-18')]


# In[24]:


# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['date'] == pd.Timestamp('2021-04-18'))
)

# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[25]:


df_fixtures_2020[df_fixtures_2020['matchday'].isna()]


# In[26]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Sevilla') & 
    (df_fixtures_2020['away_team'] == 'Elche')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 2  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[27]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Atletico Madrid') & 
    (df_fixtures_2020['away_team'] == 'Athletic Club')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 18  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[28]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Athletic Club') & 
    (df_fixtures_2020['away_team'] == 'Barcelona')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 2  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[29]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Barcelona') & 
    (df_fixtures_2020['away_team'] == 'Elche')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 1 # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[30]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Real Madrid') & 
    (df_fixtures_2020['away_team'] == 'Getafe')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 1  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[31]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Atletico Madrid') & 
    (df_fixtures_2020['away_team'] == 'Sevilla')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 1  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[32]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Granada CF') & 
    (df_fixtures_2020['away_team'] == 'Osasuna')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 4  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[33]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Barcelona') & 
    (df_fixtures_2020['away_team'] == 'Real Sociedad')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 19  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[34]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Real Madrid') & 
    (df_fixtures_2020['away_team'] == 'Athletic Club')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 2  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[35]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Levante') & 
    (df_fixtures_2020['away_team'] == 'Atletico Madrid')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 2  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[36]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Alaves') & 
    (df_fixtures_2020['away_team'] == 'Huesca')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[37]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Atletico Madrid') & 
    (df_fixtures_2020['away_team'] == 'Eibar')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[38]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Getafe') & 
    (df_fixtures_2020['away_team'] == 'Real Madrid')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[39]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Levante') & 
    (df_fixtures_2020['away_team'] == 'Villarreal')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[40]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Real Betis') & 
    (df_fixtures_2020['away_team'] == 'Valencia')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[41]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Cadiz') & 
    (df_fixtures_2020['away_team'] == 'Celta Vigo')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[42]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Osasuna') & 
    (df_fixtures_2020['away_team'] == 'Elche')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[43]:


# Assuming df_fixtures is your DataFrame

# Specify the conditions for the row you want to update
conditions = (
    (df_fixtures_2020['home_team'] == 'Real Sociedad') & 
    (df_fixtures_2020['away_team'] == 'Sevilla')
)



# Set the matchday for the specified row, replacing 'matchday_value' with the correct matchday
matchday_value = 33  # Replace with the actual matchday value if different
df_fixtures_2020.loc[conditions, 'matchday'] = matchday_value

# Assuming you've handled all NaN values in the matchday column, convert the matchday column to integer


# In[44]:


# Assuming df_fixtures is your DataFrame with the matchday column
df_fixtures_2020['matchday'] = df_fixtures_2020['matchday'].astype(int)
df_ratings = df_merged[['team.name', 'team_rating']]


# In[45]:


df_fixtures_2020[df_fixtures_2020.isna().any(axis=1)]


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have two DataFrames: fixtures_df and ratings_df
# fixtures_df contains columns: ['date', 'home_team', 'away_team', 'full_time_score', 'matchday']
# ratings_df contains columns: ['team.name', 'team_rating']

# Merge the fixtures dataframe with the ratings dataframe for both home and away teams
df_fixtures_2020 = df_fixtures_2020.merge(df_ratings, left_on='home_team', right_on='team.name', how='left')
df_fixtures_2020 = df_fixtures_2020.merge(df_ratings, left_on='away_team', right_on='team.name', how='left', suffixes=('_home', '_away'))


# In[47]:


# Drop the duplicate 'team.name' columns from the merge
df_fixtures_2020.drop(columns=['team.name_home', 'team.name_away'], inplace=True)

# Function to plot the ratings for a given team
def plot_ratings_for_team(team, df):
    # Filter matches where the team is either home or away
    df_team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
    
    # Determine opponent ratings based on whether the team is home or away
    df_team_matches['opponent_rating'] = df_team_matches.apply(
        lambda row: row['team_rating_away'] if row['home_team'] == team else row['team_rating_home'], axis=1
    )
    
    # Sort the matches by date
    df_team_matches.sort_values('date', inplace=True)
    
    # Plotting the line graph
    plt.figure(figsize=(15, 5))
    plt.plot(df_team_matches['date'], df_team_matches['opponent_rating'], marker='o', linestyle='-')
    plt.title(f'Team Rating of Opponents for {team} Over Time')
    plt.xlabel('Date')
    plt.ylabel('Opponent Team Rating')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout() # This ensures that the plot is neatly laid out
    plt.show()

# Now you can call this function for any team. For example:
plot_ratings_for_team('Barcelona', df_fixtures_2020)


# ### Team Form

# In[48]:


import pandas as pd

# Assuming you have a DataFrame named df with columns: ['date', 'team', 'opponent', 'goals', 'opponent_rating']
df20_tf = df_merged
# Sort the DataFrame by date in descending order


# In[49]:


import pandas as pd
import json

# Load JSON data from a file
with open('fixtures_2020.json', 'r') as f:
    data = json.load(f)


# In[50]:


import pandas as pd

# Assuming you have loaded the JSON data into a variable called 'data'

# Extracting fixtures information and calculating goal difference
fixtures_data = []
for fixture in data['response']:
    # Extracting home and away goals
    home_goals = fixture['goals']['home']
    away_goals = fixture['goals']['away']

    # Calculating goal difference
    goal_difference = home_goals - away_goals

    fixture_info = {
        'date': fixture['fixture']['date'],
        'home_team': fixture['teams']['home']['name'],
        'away_team': fixture['teams']['away']['name'],
        'goal_difference': goal_difference  # Adding the goal difference
    }
    fixtures_data.append(fixture_info)

# Creating DataFrame
df_fixtures = pd.DataFrame(fixtures_data)


# In[51]:


df_fix_2020 = df_fixtures.copy()


# In[52]:


# Convert the 'date' column to datetime
df_fix_2020['date'] = pd.to_datetime(df_fix_2020['date'])
df_fix_2020['date'] = df_fix_2020['date'].dt.date


# In[53]:


fixtures_2020 = df_fixtures_2020.copy()


# In[54]:


import pandas as pd

# Assuming you have df_fixtures and second_df dataframes

# Merge the dataframes based on 'date,' 'home_team,' and 'away_team'
fixtures_2020 = fixtures_2020.merge(df_fix_2020[['date', 'home_team', 'away_team', 'goal_difference']], 
                              on=['date', 'home_team', 'away_team'], 
                              how='left')

# Now merged_df contains the 'goal_difference' column from the second dataframe


# In[55]:


fixtures_2020 = fixtures_2020.sort_values(by='date')
fixtures_2020_5f = fixtures_2020.copy() 


# In[56]:


import pandas as pd

def get_last_5_matches(df):
    # Convert the date column to datetime, if not already in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a new DataFrame to store the last 5 matches for each team
    last_5_matches = pd.DataFrame()
    
    # Get unique teams from both home and away columns
    teams = pd.unique(df[['home_team', 'away_team']].values.ravel('K'))
    
    # For each team, get the last 5 matches
    for team in teams:
        # Get matches where the team was either home or away
        team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
        # Sort the matches by date in descending order and take the top 5
        most_recent_5 = team_matches.sort_values(by='date', ascending=False).head(5)
        
        # Append these matches to the last_5_matches DataFrame
        last_5_matches = pd.concat([last_5_matches, most_recent_5], ignore_index=True)
    
    # Since teams will have duplicates in the last_5_matches DataFrame (as they will appear in both home and away),
    # we need to remove these duplicates by keeping the first occurrence
    #last_5_matches = last_5_matches.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
    
    return last_5_matches

# To use this function, you would call it with your dataframe:
last_5_df = get_last_5_matches(fixtures_2020_5f)

# Replace 'your_full_matches_df' with the variable name of your dataframe containing all matches.
#last_5_df.to_csv('last_5.csv')
# print(last_5_df)


# In[57]:


teams = pd.unique(last_5_df[['home_team', 'away_team']].values.ravel('K'))

# Create a dictionary to hold dataframes for each team
team_dfs = {team: pd.DataFrame() for team in teams}

# Split the dataframe into individual team dataframes
for team in teams:
    team_home_games = last_5_df[last_5_df['home_team'] == team]
    team_away_games = last_5_df[last_5_df['away_team'] == team]
    # Combine the home and away games into one dataframe per team
    team_dfs[team] = pd.concat([team_home_games, team_away_games]).drop_duplicates()


# In[58]:


import pandas as pd

def update_goal_difference(df):
    # Determine the team that occurs the most in 'home_team' and 'away_team' columns
    most_common_team = df['home_team']._append(df['away_team']).mode().iloc[0]

    # Check and update the 'goal_difference' column based on 'full_time_score'
    for index, row in df.iterrows():
        score_parts = row['full_time_score'].split('-')
        if most_common_team == row['home_team']:
            goal_difference = int(score_parts[0]) - int(score_parts[1])
        else:
            goal_difference = int(score_parts[1]) - int(score_parts[0])

        df.at[index, 'goal_difference'] = goal_difference

    return df


# In[59]:


import warnings
warnings.filterwarnings("ignore")

# Assuming you have a dictionary 'team_dfs' containing DataFrames for each team
for team, df in team_dfs.items():
    updated_df = update_goal_difference(df)
    team_dfs[team] = updated_df


# In[60]:


for team in teams:
    globals()[f"{team.replace(' ', '_')}_df"] = team_dfs[team]


# In[61]:


import warnings
warnings.filterwarnings("ignore")

# Assuming you have a dictionary 'team_dfs' containing DataFrames for each team
for team, df in team_dfs.items():
    updated_df = update_goal_difference(df)
    team_dfs[team] = updated_df


# In[62]:


df


# In[63]:


import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def update_form_column(df, most_common_team):
    """
    Update the 'form' column of a DataFrame based on the given team.

    Args:
        df (pd.DataFrame): The DataFrame to update.
        most_common_team (str): The team to consider when calculating the 'form' column.

    Returns:
        pd.DataFrame: The DataFrame with the 'form' column updated.
    """
    def calculate_form(row):
        if most_common_team == row['home_team']:
            return row['goal_difference'] * row['team_rating_away']
        else:
            return row['goal_difference'] * row['team_rating_home']

    df['form'] = df.apply(calculate_form, axis=1)
    return df

# Usage example:
# Assuming you have a DataFrame 'Barcelona_df' and a 'most_common_team' variable
for team, df in team_dfs.items():
    new_df = update_form_column(df, team)
    team_dfs[team] = new_df


# In[64]:


final_form_df = pd.DataFrame()

# Iterate through each team's DataFrame and calculate the sum of 'form' values
for team, df in team_dfs.items():
    total_form = df['form'].sum()
    final_form_value = total_form / len(df)  # Divide by the number of matches
    final_form_df.loc[team, 'final_form'] = final_form_value

# Reset the index to have the team names as a column
final_form_df.reset_index(inplace=True)
final_form_df.rename(columns={'index': 'team'}, inplace=True)

# Display the final_form_df DataFrame
# final_form_df


# In[65]:


scaler = MinMaxScaler()

# Scale the 'final_form' column
final_form_df['final_form'] = scaler.fit_transform(final_form_df[['final_form']])

# Display the scaled final_form_df
# final_form_df


# In[66]:


# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# import pandas as pd

# # Assuming final_form_df and df_fixtures_2019 are your DataFrames
# # Replace these with your actual DataFrame loading if necessary

# app = dash.Dash(__name__)

# # Function to create traces for each team
# def create_trace_for_team(team, df):
#     df_team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
#     df_team_matches['opponent_rating'] = df_team_matches.apply(
#         lambda row: row['team_rating_away'] if row['home_team'] == team else row['team_rating_home'], axis=1
#     )
#     df_team_matches.sort_values('date', inplace=True)
    
#     hover_text = [
#         f"Current Team: {team}<br>"
#         f"Opponent Team: {row['home_team'] if row['away_team'] == team else row['away_team']}<br>"
#         f"Matchday: {row['matchday']}<br>"
#         f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
#         f"Score: {row['full_time_score']}<br>"
#         f"Team Rating: {row['opponent_rating']*100:.2f} %"
#         for _, row in df_team_matches.iterrows()
#     ]
    
#     return [
#         go.Scatter(
#             x=df_team_matches['date'],
#             y=df_team_matches['opponent_rating'] * 100,
#             name=team,
#             mode='lines+markers',
#             line=dict(width=3, color='lightcoral'),
#             marker=dict(size=12, symbol='circle', opacity=0.7, color='lightcoral'),
#             hoverinfo='text',
#             text=hover_text
#         )
#     ]

# @app.callback(
#     [
#         Output('form-output', 'children'),
#         Output('team-graph', 'figure')
#     ],
#     [Input('team-dropdown', 'value')]
# )
# def update_output(selected_team):
#     # Update the current form text
#     form_text = ''
#     if selected_team is not None:
#         team_form = final_form_df[final_form_df['team'] == selected_team]['final_form'].values[0]
#         form_text = f'Current form: {team_form:.2%}'

#     # Update the figure to show the selected team's data
#     fig = make_subplots(rows=1, cols=1)
#     for team in df_fixtures_2020['home_team'].unique():
#         traces = create_trace_for_team(team, df_fixtures_2020)
#         for trace in traces:
#             trace['visible'] = (team == selected_team)
#             fig.add_trace(trace, row=1, col=1)
#     fig.update_layout(
#         title_text="Team Rating of Opponents Over Time",
#         xaxis_title="Date",
#         yaxis_title="Opponent Team Rating",
#         yaxis=dict(range=[0, 100], showgrid=True, gridcolor='black'),
#         xaxis=dict(showgrid=False),
#         paper_bgcolor='white',
#         plot_bgcolor='white'
#     )

#     # Here we set the visibility of the traces depending on the selected team
#     fig.for_each_trace(
#         lambda trace: trace.update(visible=True) if trace.name == selected_team else trace.update(visible=False)
#     )

#     return form_text, fig


# app.layout = html.Div([
#     dcc.Dropdown(
#         id='team-dropdown',
#         options=[{'label': team, 'value': team} for team in final_form_df['team'].unique()],
#         value=final_form_df['team'][0]
#     ),
#     html.Div(id='form-output'),
#     dcc.Graph(id='team-graph')
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True, port=8049, use_reloader=False)


# ### Stacked bar chart for goals scored and conceded
# 

# In[67]:


fixtures_5_g = df_fixtures_2020.copy()


# In[68]:


teams = pd.unique(fixtures_5_g[['home_team', 'away_team']].values.ravel('K'))


# In[69]:


# Create a dictionary to hold dataframes for each team
team_datafs = {team: pd.DataFrame() for team in teams}

# Split the dataframe into individual team dataframes
for team in teams:
    team_home_games = fixtures_5_g[fixtures_5_g['home_team'] == team]
    team_away_games = fixtures_5_g[fixtures_5_g['away_team'] == team]
    # Combine the home and away games into one dataframe per team
    team_datafs[team] = pd.concat([team_home_games, team_away_games]).drop_duplicates()


# In[70]:


def split_goals(df, team_name):
    df['goals_scored'] = df.apply(lambda x: int(x['full_time_score'].split(' - ')[0]) if x['home_team'] == team_name else int(x['full_time_score'].split(' - ')[1]), axis=1)
    df['goals_conceded'] = df.apply(lambda x: int(x['full_time_score'].split(' - ')[1]) if x['home_team'] == team_name else int(x['full_time_score'].split(' - ')[0]), axis=1)
    return df[['matchday', 'goals_scored', 'goals_conceded']]

# Prepare the data for plotting
team_goals_data = {}
for team_name, df in team_dfs.items():
    team_goals_data[team_name] = split_goals(df, team_name)


# In[71]:


for team in teams:
    globals()[f"{team.replace(' ', '_')}_df"] = team_datafs[team]


# In[72]:


import pandas as pd
import plotly.graph_objs as go

# Assuming 'teams' is a list of team names and each team's data is in a DataFrame


# Define your color scheme for the bars
goals_scored_color = '#1f77b4'  # Example color for goals scored
goals_conceded_color = '#ff7f0e'  # Example color for goals conceded

# Create an empty figure
fig = go.Figure()

# Add a pair of traces (goals scored and goals conceded) for each team
for team in teams:
    team_df = globals()[f"{team.replace(' ', '_')}_df"]
    team_df[['goals_scored', 'goals_conceded']] = team_df['full_time_score'].str.split('-', expand=True).astype(int)
    
    # Trace for goals scored
    fig.add_trace(
        go.Bar(
            x=team_df['matchday'],
            y=team_df['goals_scored'],
            name=f'{team} Goals Scored',
            marker_color=goals_scored_color,
            visible=False,
            hoverinfo='y+name',
            hovertemplate='Matchday: %{x}<br>Goals Scored: %{y}<extra></extra>'
        )
    )
    
    # Trace for goals conceded
    fig.add_trace(
        go.Bar(
            x=team_df['matchday'],
            y=team_df['goals_conceded'],
            name=f'{team} Goals Conceded',
            marker_color=goals_conceded_color,
            visible=False,
            hoverinfo='y+name',
            hovertemplate='Matchday: %{x}<br>Goals Conceded: %{y}<extra></extra>'
        )
    )

# Create a list for the dropdown options
buttons = []

for i, team in enumerate(teams):
    # Set the visibility of all traces to False
    visibility = [False] * len(teams) * 2  # We have two traces per team
    # Set the visibility of the current team's traces to True
    visibility[i*2] = True  # Goals scored trace
    visibility[i*2 + 1] = True  # Goals conceded trace
    buttons.append(
        dict(
            label=team,
            method='update',
            args=[{'visible': visibility},
                  {'title': f'Stacked Bar Chart for {team}'}]
        )
    )

# Update the layout with the dropdown
fig.update_layout(
    updatemenus=[{
        'buttons': buttons,
        'direction': 'down',
        'pad': {'r': 10, 't': 10},
        'showactive': True,
        'x': 0.1,
        'xanchor': 'left',
        'y': 1.1,
        'yanchor': 'top'
    }],
    barmode='stack',
    paper_bgcolor='#f8f9fa',  # Set the paper background color
    plot_bgcolor='#f8f9fa',  # Set the plot background color
    font=dict(
        family='Arial, Helvetica, sans-serif',  # Match the font family
        size=20,  # Match the font size
    ),
    title='Select a team to display the goals by matchday'
)

# Optionally set the first team to be visible by default
# Here we make the first team's traces visible
fig.data[0].visible = True
fig.data[1].visible = True

# Show the plot
fig.show()


# ### for the position

# In[73]:


df_2020 = df2020.copy()


# In[74]:


def update_output(selected_team):
    # Initialize the form text to be displayed
    form_text = ''
    
    # Check if a team has been selected
    if selected_team is not None:
        # Find the rank of the selected team
        team_rank = df_2020[df_2020['team.name'] == selected_team]['rank'].values[0]
        # Update the form text with the rank information
        form_text = f'Rank: {team_rank}'
    
    # Return the updated form text
    return form_text


# In[75]:


# import dash
# from dash import html, dcc
# from dash.dependencies import Input, Output

# # Assuming 'final_form_df' is your DataFrame with the 'team' and 'rank' columns

# # Initialize the Dash app
# app = dash.Dash(__name__)

# # Define the layout of the app
# app.layout = html.Div([
#     # Dropdown for selecting a team
#     dcc.Dropdown(
#         id='team-dropdown',
#         options=[{'label': team, 'value': team} for team in df_2020['team.name']],
#         value=None  # No default value, require user to select an option
#     ),
#     # Div to display the rank
#     html.Div(id='rank-output')
# ])

# # Set up the callback
# @app.callback(
#     Output('rank-output', 'children'),
#     [Input('team-dropdown', 'value')]
# )
# def update_rank(selected_team):
#     return update_output(selected_team)

# # Run the appa
# if __name__ == '__main__':
#     app.run_server(debug=True)


# ### goals scored avg

# In[76]:


import pandas as pd


df = df_fixtures_2020.copy()

# Splitting the 'full_time_score' into two columns and converting them to numeric
df[['home_goals', 'away_goals']] = df['full_time_score'].str.split(' - ', expand=True).apply(pd.to_numeric)

# Calculating the total and average goals for home games
home_goals = df.groupby('home_team')['home_goals'].agg(['sum', 'count'])
home_goals['average_goals'] = home_goals['sum'] / home_goals['count']

# Calculating the total and average goals for away games
away_goals = df.groupby('away_team')['away_goals'].agg(['sum', 'count'])
away_goals['average_goals'] = away_goals['sum'] / away_goals['count']


# In[77]:


# Combining the home and away goals and counts
total_goals = home_goals['sum'].add(away_goals['sum'], fill_value=0)
total_matches = home_goals['count'].add(away_goals['count'], fill_value=0)

# Calculating the overall average goals per team
average_goals_per_team = total_goals / total_matches

average_goals_per_team.sort_values(ascending=False)


# In[78]:



# Creating a DataFrame from the average goals per team
average_goals_df = average_goals_per_team.reset_index()
average_goals_df.columns = ['Team', 'Goals Scored Avg']

# Sorting the teams by 'Average Goals' and adding a 'Rank' column
average_goals_df = average_goals_df.sort_values(by='Goals Scored Avg', ascending=False).reset_index(drop=True)
average_goals_df['Rank'] = average_goals_df.index + 1


# ### goals conceded avg

# In[79]:


# Calculating the total and average goals conceded for home games
home_goals_conceded = df.groupby('home_team')['away_goals'].agg(['sum', 'count'])
home_goals_conceded['average_goals_conceded'] = home_goals_conceded['sum'] / home_goals_conceded['count']

# Calculating the total and average goals conceded for away games
away_goals_conceded = df.groupby('away_team')['home_goals'].agg(['sum', 'count'])
away_goals_conceded['average_goals_conceded'] = away_goals_conceded['sum'] / away_goals_conceded['count']

# Combining the home and away goals conceded and counts
total_goals_conceded = home_goals_conceded['sum'].add(away_goals_conceded['sum'], fill_value=0)
total_matches_conceded = home_goals_conceded['count'].add(away_goals_conceded['count'], fill_value=0)

# Calculating the overall average goals conceded per team
average_goals_conceded_per_team = total_goals_conceded / total_matches_conceded

# Creating a DataFrame from the average goals conceded per team
average_goals_conceded_df = average_goals_conceded_per_team.reset_index()
average_goals_conceded_df.columns = ['Team', 'Average Goals Conceded']

# Sorting the teams by 'Average Goals Conceded' and adding a 'Rank' column
average_goals_conceded_df = average_goals_conceded_df.sort_values(by='Average Goals Conceded', ascending=True).reset_index(drop=True)
average_goals_conceded_df['Rank'] = average_goals_conceded_df.index + 1

 # Displaying the top 10 teams as an example


# #### clean sheets 

# In[80]:


# Calculating the number of clean sheets for home and away games
# Clean sheet is when no goals are conceded. Therefore, away_goals = 0 for home clean sheets, and home_goals = 0 for away clean sheets.

# Counting clean sheets for home games
home_clean_sheets = df[df['away_goals'] == 0].groupby('home_team').size()

# Counting clean sheets for away games
away_clean_sheets = df[df['home_goals'] == 0].groupby('away_team').size()

# Combining the home and away clean sheets
total_clean_sheets = home_clean_sheets.add(away_clean_sheets, fill_value=0)

# Creating a DataFrame from the total clean sheets per team
clean_sheets_df = total_clean_sheets.reset_index()
clean_sheets_df.columns = ['Team', 'Clean Sheets']

# Sorting the teams by 'Clean Sheets' and adding a 'Rank' column
clean_sheets_df = clean_sheets_df.sort_values(by='Clean Sheets', ascending=False).reset_index(drop=True)
clean_sheets_df['Rank'] = clean_sheets_df.index + 1

clean_sheets_df.head(10)  # Displaying the top 10 teams as an example


# In[81]:


# To calculate the average clean sheets, we first need the total number of matches played by each team

# Counting the total matches played by each team (both home and away)
total_matches_played = df.groupby('home_team').size().add(df.groupby('away_team').size(), fill_value=0)

# Calculating the average clean sheets per team
average_clean_sheets_per_team = total_clean_sheets / total_matches_played

# Creating a DataFrame from the average clean sheets per team
average_clean_sheets_df = average_clean_sheets_per_team.reset_index()
average_clean_sheets_df.columns = ['Team', 'Average Clean Sheets']

# Sorting the teams by 'Average Clean Sheets' and adding a 'Rank' column
average_clean_sheets_df = average_clean_sheets_df.sort_values(by='Average Clean Sheets', ascending=False).reset_index(drop=True)
average_clean_sheets_df['Rank'] = average_clean_sheets_df.index + 1

average_clean_sheets_df.head(10)  # Displaying the top 10 teams as an example


# In[82]:


# Assuming average_goals_df, average_goals_conceded_df, and average_clean_sheets_df are your dataframes

# Merge the DataFrames on the 'Team' column
combined_df = average_goals_df.merge(average_goals_conceded_df, on='Team', suffixes=('_scored', '_conceded'))
combined_df = combined_df.merge(average_clean_sheets_df, on='Team')

# Rename the columns to match the desired structure
combined_df.rename(columns={
    'Average Goals_scored': 'Goals Scored Avg',
    'Rank_scored': 'Goals Scored Rank',
    'Average Goals Conceded': 'Goals Conceded Avg',
    'Rank_conceded': 'Goals Conceded Rank',
    'Average Clean Sheets': 'Clean Sheets Avg',
    'Rank': 'Clean Sheets Rank'
}, inplace=True)

# Ensure the 'Rank' columns are integers
combined_df['Goals Scored Rank'] = combined_df['Goals Scored Rank'].astype(int)
combined_df['Goals Conceded Rank'] = combined_df['Goals Conceded Rank'].astype(int)
combined_df['Clean Sheets Rank'] = combined_df['Clean Sheets Rank'].astype(int)

# Now you can use the combined_df with the Plotly code provided earlier


# In[83]:


combined_df


# ### Final Dashboard

# # Sreesh Code

# In[84]:


dfs1 = pd.read_csv('2020standings(36).csv') 
dfs2 = pd.read_csv('2020points(36).csv')


# In[85]:


# Convert the DataFrame back to a dictionary with the first column as keys and the second as values
teams_positions = pd.Series(dfs1['Positions'].values, index=dfs1['Team Name']).to_dict()
teams_points = pd.Series(dfs2['Points'].values, index=dfs2['Team Name']).to_dict()


# In[86]:


# Extract only the second value from each list
matchday_list = [value[1] for value in matchday_dict.values()]

teams_positions


# In[87]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Assume teams_positions is already defined
# Number of matchdays
matchdays = list(range(1, 37))

team_colors = {
    'Real Madrid': '#fcbf00',
    'Atletico Madrid': '#262f61',
    'Barcelona': '#a50044',
    'Athletic Club': '#ee2523',
    'Valencia': '#ee3524',
    'Osasuna': '#0a346f',
    'Real Betis': '#00954c',
    'Cadiz': '#fde607',
    'Getafe': '#004fa3',
    'Real Sociedad': '#143c8b',
    'Villarreal': '#005187',
    'Alaves': '#004fa3',
    'Granada CF': '#e60d2e',
    'Celta Vigo': '#e5254e',
    'Sevilla': '#d70f21',
    'Valladolid': '#178555',
    'Huesca': '#ffed00',
    'Eibar': '#1c2f6d',
    'Elche': '#05642c',
    'Levante': '#b2b2b2'
   
}

# Create subplots
fig = make_subplots(rows=1, cols=1)
val = 0
# Plotting each team's positions across matchdays
for team, positions in teams_positions.items():
    # You can add more details within the hover text such as the team's name and matchday.
    hover_text = [f"{team}<br>Matchday {matchday}<br>Date: {matchday_list_value}<br>Position: {position}" for matchday, position, matchday_list_value in zip(matchdays, json.loads(positions), matchday_list)]
    fig.add_trace(
        go.Scatter(
            x=matchdays,
            y=json.loads(positions),
            mode='lines+markers',
            name=team,
            text=hover_text,  # This is where you set the hover text
            hoverinfo='text', 
            line=dict(color=team_colors.get(team, '#000')),  # Use the team's color, default to black if not found
            marker=dict(color=team_colors.get(team, '#000')), # Display the text on hover
            visible='legendonly'
        )
    )
    if val == len(matchday_list) - 1:
        val =-1 
    val += 1

# Highlighting y-values from 1 to 4 with a green background
fig.add_hrect(y0=0.5, y1=4.5, line_width=0, fillcolor="green", opacity=0.15)
fig.add_hrect(y0=4.5, y1=6.5, line_width=0, fillcolor="blue", opacity=0.15)
fig.add_hrect(y0=17.5, y1=20.5, line_width=0, fillcolor="red", opacity=0.15)




# Update y-axis
fig.update_yaxes(autorange="reversed", tickmode='array', tickvals=list(range(1, 21)), title='Position')

# Update x-axis
fig.update_xaxes(tickmode='array', tickvals=list(range(1, 37)), title='Matchday')

# Update layout
fig.update_layout(
    title={
        'text': "Team Positions Over Matchdays (2020)",
        'y':0.9,
        'x':0.47,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(x=1.05, y=1, title='Clubs'),
    hovermode="closest"
)

# Update the size of the figure (optional)
fig.update_layout(width=1000, height=600)

# Show figure
fig.show()


# In[88]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Assume teams_points and team_colors are already defined
# team_colors should be a dictionary mapping team names to their color strings

# Create subplots
fig = make_subplots(rows=1, cols=1)

# Plotting each team's points across matchdays
for team, points in teams_points.items():
    val = 0
    # Ensure matchdays is defined, e.g., matchdays = range(1, len(points) + 1)
    hover_text = [f"{team}<br>Matchday {matchday}<br>Date: {matchday_list[-2]}<br>Points: {point}" for matchday, point in enumerate(json.loads(points), start=1)]
    fig.add_trace(
        go.Scatter(
            x=list(range(1, 37)),  # Assuming matchdays are a sequential list starting at 1
            y=json.loads(points),
            mode='lines+markers',
            name=team,
            text=hover_text,  # This is where you set the hover text
            hoverinfo='text',
            line=dict(color=team_colors.get(team, '#000')),  # Use the team's color
            marker=dict(color=team_colors.get(team, '#000')),  # Set marker color
            visible='legendonly'  # Start with the trace not shown
        )
    
    )
    val += 1

# Update y-axis
fig.update_yaxes(title='Total Points')

# Update x-axis
# Assuming the number of matchdays is known, otherwise adjust accordingly
fig.update_xaxes(tickmode='array', tickvals=list(range(1, 37)), title='Matchday')

# Update layout
fig.update_layout(
    title={
        'text': "Team Points Over Matchdays (2021)",
        'y':0.9,
        'x':0.5,  # Set to 0.5 to center the title
        'xanchor': 'center',
        'yanchor': 'top'
    },
    legend=dict(x=1.05, y=1, title='Clubs'),
    hovermode="closest"
)

# Update the size of the figure (optional)
fig.update_layout(width=1000, height=600)

# Show figure
fig.show()


# In[89]:


# Now, create a list from a column of the dataframe
teams_list = combined_df['Team'].tolist()


# In[90]:


teams_list


# In[91]:


teams_stats = {'Atletico Madrid': [1, 2, 1, 2, 1],
 'Real Madrid': [2, 3, 2, 4, 2],
 'Sevilla': [3, 7, 4, 1, 3],
 'Barcelona': [4, 1, 3, 3, 1, 5],
 'Real Sociedad': [5, 5, 5, 7, 4],
 'Athletic Club': [6, 11, 9, 6, 11],
 'Getafe': [7, 20, 6, 9, 6],
 'Villarreal': [8, 4, 8, 8, 8],
 'Osasuna': [9, 13, 7, 10, 7],
 'Real Betis': [10, 8, 11, 13, 13],
 'Eibar': [11, 19, 10, 12, 10],
 'Huesca': [12, 17, 15, 14, 9],
 'Valencia': [13, 9, 14, 11, 20],
 'Elche': [14, 16, 13, 16, 19],
 'Alaves': [15, 14, 12, 15, 12],
 'Celta Vigo': [16, 6, 18, 18, 15],
 'Levante': [17, 12, 19, 17, 16],
 'Valladolid': [18, 18, 17, 5, 17],
 'Cadiz': [19, 15, 16, 20, 9, 18],
 'Granada CF': [20, 10, 20, 19, 14]}


# In[92]:



for key, value in teams_stats.items():
    if value:  # Check if the list is not empty
        value.append(value[0])
teams_stats


# In[93]:


for key, values in teams_stats.items():
    teams_stats[key] = [abs(value-21) for value in values]

teams_stats


# In[94]:


for key, values in teams_stats.items():
    teams_stats[key] = [value/20 for value in values]
teams_stats


# In[95]:


import plotly.graph_objects as go


categories = ['Defence','Attack','vs Big 3',
              'Win Streak','Clean Sheet']



categories.append(categories[0])

fig = go.Figure()

for val in teams_stats:
    fig.add_trace(go.Scatterpolar(
          r=teams_stats[val],
          theta=categories,
          fill='toself',
          name= val,
        showlegend=val in teams_list, visible = True))
    

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True
)

fig.show()


# In[96]:


# Assuming teams_positions is your dictionary and teams_list is your list
missing_teams = [team for team in teams_stats if team not in teams_list]

# This will give you a list of teams that are in teams_positions but not in teams_list


# In[97]:


missing_teams


# In[98]:


# Extracting 'name' from the 'team' dictionary
#df2020['team.name'] = df2020['team'].apply(lambda x: x['name'])

# Selecting only the desired columns
new_df = df2020[['rank',  'team.name', 'goalsDiff','points']]
new_df = new_df.rename(columns={'rank': '#', 'points': 'Points', 'goalsDiff':'GD', 'team.name': 'Team'})

new_df


# In[104]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import dash_table

# Assuming final_form_df and df_fixtures_2019 are your DataFrames
# Replace these with your actual DataFrame loading if necessary

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            'Team Performance Dashboard (2019-2020 Season)',
            style={'textAlign': 'center', 'fontFamily': 'Arial, Helvetica, sans-serif'}
        ),
        dcc.Dropdown(
            id='team-dropdown',
            options=[{'label': team, 'value': team} for team in final_form_df['team'].unique()],
            value=final_form_df['team'][0],
            style={'width': '50%', 'margin': '10px auto', 'fontFamily': 'Arial, Helvetica, sans-serif'}
        ),
        html.Div(id='rank-output', 
            style={
                'textAlign': 'center',
                'padding': '20px',
                'marginTop': '20px',
                'background-color': '#1c0d2d',
                'color': 'white',
                'border-radius': '5px',
                'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                'fontSize': '20px',
                'fontFamily': 'Arial, Helvetica, sans-serif'
            }
        ),
#         html.Div(id='form-output',  # New div for current form
#             style={
#                 'textAlign': 'center',
#                 'padding': '20px',
#                 'marginTop': '20px',
#                 'background-color': '#1c0d2d',
#                 'color': 'white',
#                 'border-radius': '5px',
#                 'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
#                 'fontSize': '20px',
#                 'fontFamily': 'Arial, Helvetica, sans-serif'
#             }
#         ),
#         html.Div([dash_table.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} for i in new_df.columns],
#         data=new_df.to_dict('records'),
#         style_header={
#             'fontWeight': 'bold',  # Make header bold
#             'backgroundColor': 'rgb(230, 230, 230)',  # Optional: add background color
#             'textAlign': 'center'  # Optional: align text to center
#         },
#         # Other optional styling
#         style_cell={'padding': '5px'},
#         style_data={
#             'whiteSpace': 'normal',
#             'height': 'auto'
#         },
#     )
#         ]),
        # New div for side-by-side layout
        html.Div([
            # Div for the table
            html.Div([
                dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in new_df.columns],
                    data=new_df.to_dict('records'),
                    # (Rest of your DataTable properties)
                )
            ], style={'width': '60%', 'display': 'inline-block', 'padding': '30px', 'fontFamily': 'Arial, Helvetica, sans-serif'}),

            # Div for the form output
            html.Div(id='form-output', 
                style={
                    'width': '40%', 
                    'display': 'inline-block', 
                    'padding': '10px',
                    'textAlign': 'center',
                     'padding': '20px',
                     'marginTop': '70px',
                    'height': '150px',
                     'background-color': '#1c0d2d',
                     'color': 'white',
                    'lineHeight': '150px',
                     'border-radius': '5px',
                     'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
                     'fontSize': '20px',
                     'fontFamily': 'Arial, Helvetica, sans-serif'
                    # (Rest of your form-output styles)
                }
            ),
        ], style={'display': 'flex', 'justifyContent': 'center'}),

        # ... (rest of your components)
    
        dcc.Graph(id='team-graph'),
        dcc.Graph(id='bar-chart'),
        dcc.Graph(id='positions-graph'),
        dcc.Graph(id='points-graph'),
        dcc.Graph(id='team-stats-graph'),
        dcc.Graph(id='star-graph')
    ],
    style={'padding': '20px', 'fontFamily': 'Arial, Helvetica, sans-serif'}
)

def create_team_positions_graph(selected_team, teams_positions, matchdays):
    # Create subplots
    fig = make_subplots(rows=1, cols=1)

    # Add the trace for the selected team
    if selected_team in teams_positions:
        positions = json.loads(teams_positions[selected_team])
        hover_text = [f"{selected_team}<br>Matchday {matchday}<br>Position: {position}" for matchday, position in zip(matchdays, positions)]
        fig.add_trace(
            go.Scatter(
                x=matchdays,
                y=positions,
                mode='lines+markers',
                name=selected_team,
                text=hover_text,
                hoverinfo='text',
                #line=dict(color=team_colors.get(selected_team, '#000')),
                #marker=dict(color=team_colors.get(selected_team, '#000')),
            )
        )

    return fig


# Function to create traces for each team for scatter plot
def create_scatter_trace_for_team(team, df):
    df_team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
    df_team_matches['opponent_rating'] = df_team_matches.apply(
        lambda row: row['team_rating_away'] if row['home_team'] == team else row['team_rating_home'], axis=1
    )
    df_team_matches.sort_values('date', inplace=True)
    
    hover_text = [
        f"Current Team: {team}<br>"
        f"Opponent Team: {row['home_team'] if row['away_team'] == team else row['away_team']}<br>"
        f"Matchday: {row['matchday']}<br>"
        f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
        f"Score: {row['full_time_score']}<br>"
        f"Team Rating: {row['opponent_rating']*100:.2f} %"
        for _, row in df_team_matches.iterrows()
    ]
    
    return [
        go.Scatter(
            x=df_team_matches['date'],
            y=df_team_matches['opponent_rating'] * 100,
            name=team,
            mode='lines+markers',
            line=dict(width=3, color='lightcoral'),
            marker=dict(size=12, symbol='circle', opacity=0.7, color='lightcoral'),
            hoverinfo='text',
            text=hover_text
        )
    ]


# Function to create traces for each team for bar chart
def create_bar_traces_for_team(team, df):
    traces = []
    team_df = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()

    # Add a column to team_df to indicate the number of goals scored by the selected team in each match
    team_df['goals_scored'] = team_df.apply(
        lambda row: int(row['full_time_score'].split('-')[0]) if row['home_team'] == team else int(row['full_time_score'].split('-')[1]),
        axis=1
    )

    # Add a column to team_df to indicate the number of goals conceded by the selected team in each match
    team_df['goals_conceded'] = team_df.apply(
        lambda row: int(row['full_time_score'].split('-')[1]) if row['home_team'] == team else int(row['full_time_score'].split('-')[0]),
        axis=1
    )

    # Trace for goals scored
    traces.append(
        go.Bar(
            x=team_df['matchday'],
            y=team_df['goals_scored'],
            name='Goals Scored',
            hoverinfo='y+name',
            hovertemplate='Matchday: %{x}<br>Goals Scored: %{y}<extra></extra>'
        )
    )
    
    # Trace for goals conceded
    traces.append(
        go.Bar(
            x=team_df['matchday'],
            y=team_df['goals_conceded'],
            name='Goals Conceded',
            hoverinfo='y+name',
            hovertemplate='Matchday: %{x}<br>Goals Conceded: %{y}<extra></extra>'
        )
    )
    
    return traces

def calculate_average_goals(df):
    average_goals_data = {}
    for matchday in range(1, 39):
        matchday_goals = []
        matchday_games = df[df['matchday'] == matchday]
        
        for _, row in matchday_games.iterrows():
            goals = row['full_time_score'].split('-')
            goals = [int(goal) for goal in goals]
            matchday_goals.extend(goals)  # Adding both home and away goals for each game

        # Calculate the average if there are any goals, otherwise set to 0
        average_goals_data[matchday] = sum(matchday_goals) / len(matchday_goals) if matchday_goals else 0

    return average_goals_data

def add_team_stats_to_figure(team_data, fig):
    def get_color_by_rank(rank):
        if 1 <= rank <= 7:
            return 'darkgreen'
        elif 8 <= rank <= 15:
            return 'darkorange'  # Using dark orange instead of dark yellow for better visibility
        elif 16 <= rank <= 20:
            return 'darkred'
        else:
            return 'grey' 
    if not team_data.empty:
        # Goals Scored Average
        goals_scored_rank = team_data['Goals Scored Rank'].iloc[0]
        goals_scored_color = get_color_by_rank(goals_scored_rank)
        fig.add_trace(go.Indicator(
            mode="number",
            value=team_data['Goals Scored Avg'].iloc[0],
            number={'font': {'color': goals_scored_color}},
            title={'text': f"<b>Goals Per Game</b><br><sup>Rank: {goals_scored_rank}</sup>",
                   'font': {'size': 30}},
            domain={'row': 0, 'column': 0}
        ))

        # Goals Conceded Average
        goals_conceded_rank = team_data['Goals Conceded Rank'].iloc[0]
        goals_conceded_color = get_color_by_rank(goals_conceded_rank)
        fig.add_trace(go.Indicator(
            mode="number",
            value=team_data['Goals Conceded Avg'].iloc[0],
            number={'font': {'color': goals_conceded_color}},
            title={'text': f"<b>Conceded Per Game</b><br><sup>Rank: {goals_conceded_rank}</sup>",
                   'font': {'size': 30}},
            domain={'row': 0, 'column': 1}
        ))

        # Clean Sheets Average
        clean_sheets_rank = team_data['Clean Sheets Rank'].iloc[0]
        clean_sheets_color = get_color_by_rank(clean_sheets_rank)
        fig.add_trace(go.Indicator(
            mode="number",
            value=team_data['Clean Sheets Avg'].iloc[0],
            number={'font': {'color': clean_sheets_color}},
            title={'text': f"<b>Clean Sheets</b><br><sup>Rank: {clean_sheets_rank}</sup>",
                   'font': {'size': 30}},
            domain={'row': 0, 'column': 2}
        ))
        


@app.callback(
    [Output('rank-output', 'children'),
     Output('form-output', 'children'),
     Output('team-graph', 'figure'),
     Output('bar-chart', 'figure'),
     Output('positions-graph', 'figure'),
     Output('points-graph', 'figure'),
     Output('star-graph', 'figure'),
     #Output('table', 'style_data_conditional'),
     Output('team-stats-graph', 'figure')],  # Additional output for the team stats graph
    [Input('team-dropdown', 'value')]
)
def update_output(selected_team):
    rank_text = None
    form_text = None
#     table_val = None
#     style_data_conditional = []

    if selected_team is not None:
        
        # Get the rank of the selected team from df_2019
        team_rank = df_2020[df_2020['team.name'] == selected_team]['rank'].iloc[0]
        # Create a styled rank text using html.Div and html.Span
        rank_text = html.Div(
            children=[
                html.Span('Position: ', style={'fontSize': '24px', 'color': 'white'}),
                html.Span(f'{team_rank}', style={'fontSize': '80px', 'color': 'lightblue', 'opacity': '0.75'})
            ],
            style={'textAlign': 'center'}
        )
       
        
        # Get the current form of the selected team from final_form_df
        team_form = final_form_df[final_form_df['team'] == selected_team]['final_form'].iloc[0]
        # Create a styled text element for form_text
        form_text = html.Div(
            children=f'Current form: {team_form:.2%}',
            style={'fontSize': '24px', 'color': 'white', 'textAlign': 'center'}  # Increase font size here
        )
        
        # Add DataTable styling based on the selected team
        
        
#         # Add the conditional style as a dictionary inside a list
#         style_data_conditional.append({
#             'if': {
#                 'filter_query': f"{{Team}} = '{selected_team}'",
#                 'column_id': 'Team'
#             },
#             'backgroundColor': '#FFDDC1',  # Highlight color
#             'fontWeight': 'bold'
#         })
       
        
    matchdays = list(range(1,37))
    test_fig = make_subplots(rows=1,cols=1)
    # Plotting each team's positions across matchdays
    for team, positions in teams_positions.items():
        if team == selected_team:
            hover_text = [f"{team}<br>Matchday {matchday}<br>Date: {matchday_list_value}<br>Position: {position}" for matchday, position, matchday_list_value in zip(matchdays, json.loads(positions), matchday_list)]
            test_fig.add_trace(
                go.Scatter(
                    x=matchdays,
                    y=json.loads(positions),
                    mode='lines+markers',
                    name=team,
                    text=hover_text,
                    hoverinfo='text',
                    line=dict(color=team_colors.get(team, '#000')),
                    marker=dict(color=team_colors.get(team, '#000')),
                    visible=True  # Set visible to True for the selected team
                )
            )
        # You can add more details within the hover text such as the team's name and matchday.
        else:
            hover_text = [f"{team}<br>Matchday {matchday}<br>Date: {matchday_list_value}<br>Position: {position}" for matchday, position, matchday_list_value in zip(matchdays, json.loads(positions), matchday_list)]
            test_fig.add_trace(
                go.Scatter(
                    x=matchdays,
                    y=json.loads(positions),
                    mode='lines+markers',
                    name=team,
                    text=hover_text,  # This is where you set the hover text
                    hoverinfo='text', 
                    line=dict(color=team_colors.get(team, '#000')),  # Use the team's color, default to black if not found
                    marker=dict(color=team_colors.get(team, '#000')), # Display the text on hover
                    visible='legendonly'
                )
            )
        
        # Highlighting y-values from 1 to 4 with a green background
    test_fig.add_hrect(y0=0.5, y1=4.5, line_width=0, fillcolor="green", opacity=0.15)
    test_fig.add_hrect(y0=4.5, y1=6.5, line_width=0, fillcolor="blue", opacity=0.15)
    test_fig.add_hrect(y0=17.5, y1=20.5, line_width=0, fillcolor="red", opacity=0.15)




    # Update y-axis
    test_fig.update_yaxes(autorange="reversed", tickmode='array', tickvals=list(range(1, 21)), title='Position')

    # Update x-axis
    test_fig.update_xaxes(tickmode='array', tickvals=list(range(1, 37)), title='Matchday')

    # Update layout
    test_fig.update_layout(
        title={
            'text': "Position",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Helvetica",
                size=24,
                color="black"
            )
        },
        xaxis=dict(
            title="Matchday",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
        ),
        yaxis=dict(
            title="Points",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
            range=[1, 21],
        ),
        legend=dict(x=1.05, y=1, title='Clubs'),margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest"
    )

    # Update the size of the figure (optional)
    test_fig.update_layout(height=600)

    
    
    points_fig = make_subplots(rows=1, cols=1)
    # Plotting each team's points across matchdays
    for team, points in teams_points.items():
        if team == selected_team:
            hover_text = [f"{team}<br>Matchday {matchday}<br>Points: {point}" for matchday, point in enumerate(json.loads(points), start=1)]
            points_fig.add_trace(
                go.Scatter(
                    x=list(range(1, 37)),  # Assuming matchdays are a sequential list starting at 1
                    y=json.loads(points),
                    mode='lines+markers',
                    name=team,
                    text=hover_text,  # This is where you set the hover text
                    hoverinfo='text',
                    line=dict(color=team_colors.get(team, '#000')),  # Use the team's color
                    marker=dict(color=team_colors.get(team, '#000')),  # Set marker color
                    visible=True  # Start with the trace not shown
                )
                

            )
        else:
            # Ensure matchdays is defined, e.g., matchdays = range(1, len(points) + 1)
            hover_text = [f"{team}<br>Matchday {matchday}<br>Points: {point}" for matchday, point in enumerate(json.loads(points), start=1)]
            points_fig.add_trace(
                    go.Scatter(
                        x=list(range(1, 37)),  # Assuming matchdays are a sequential list starting at 1
                        y=json.loads(points),
                        mode='lines+markers',
                        name=team,
                        text=hover_text,  # This is where you set the hover text
                        hoverinfo='text',
                        line=dict(color=team_colors.get(team, '#000')),  # Use the team's color
                        marker=dict(color=team_colors.get(team, '#000')),  # Set marker color
                        visible='legendonly'  # Start with the trace not shown
                    )

                )


    # Update y-axis
    points_fig.update_yaxes(title='Total Points')

    # Update x-axis
    # Assuming the number of matchdays is known, otherwise adjust accordingly
    points_fig.update_xaxes(tickmode='array', tickvals=list(range(1, 37)), title='Matchday')

    # Update layout
    points_fig.update_layout(
        title={
            'text': "Points",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Helvetica",
                size=24,
                color="black"
            )
        },
        xaxis=dict(
            title="Matchday",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
        ),
        yaxis=dict(
            title="Points",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
            range=[0, 90],
        ),
        legend=dict(x=1.05, y=1, title='Clubs'),margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest"
    )

    # Update the size of the figure (optional)
    points_fig.update_layout(height=600)




    star_fig = make_subplots(rows=1, cols=1)

    for val in teams_stats:
        team_color = team_colors.get(val, 'grey')
        # Define custom hover text
        hover_text = f"{val}<br>" + "<br>".join([f"{cat}: {stat}" for cat, stat in zip(categories, teams_stats[val])])
        
        if val == selected_team:
            star_fig.add_trace(go.Scatterpolar(
              r=teams_stats[val],
              theta=categories,
              fill='toself',
              name= val,
              line=dict(color=team_color),  # Set the color of the line
              marker=dict(color=team_color),
            hoverinfo='text',  # Use custom hover text
            hovertext=hover_text,  # Set the hover text# Set the color of the markers
              showlegend=val in teams_list, visible = True))
        else:
            star_fig.add_trace(go.Scatterpolar(
                  r=teams_stats[val],
                  theta=categories,
                  fill='toself',
                  line=dict(color=team_color),  # Set the color of the line
                  marker=dict(color=team_color),
                hoverinfo='text',  # Use custom hover text
            hovertext=hover_text,  # Set the hover text# Set the color of the markers
                  name= val,
                showlegend=val in teams_list, visible = 'legendonly'))


    star_fig.update_layout(
        title={
            'text': "Team Comparison",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Helvetica",
                size=24,
                color="black"
            )
        },
        legend=dict(x=1.05, y=1, title='Clubs', bordercolor='black',  # Set the border color
        borderwidth=2,),margin=dict(
            l=50,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,
        #height = 1000,
        #width = 1000,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest"
    )
    
    
    


    scatter_fig = make_subplots(rows=1, cols=1)
    for team in df_fixtures_2020['home_team'].unique():
        traces = create_scatter_trace_for_team(team, df_fixtures_2020)
        for trace in traces:
            trace['visible'] = (team == selected_team)
            scatter_fig.add_trace(trace, row=1, col=1)
    scatter_fig.update_layout(
        title={
            'text': "Team Rating of Opponents Over Time",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Helvetica",
                size=24,
                color="black"
            )
        },
        xaxis=dict(
            title="Date",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
        ),
        yaxis=dict(
            title="Opponent Team Rating (%)",
            titlefont=dict(
                family="Helvetica",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='grey',
            linecolor='grey',
            linewidth=2,
            tickfont=dict(
                family="Helvetica",
                size=12,
                color="black"
            ),
            range=[0, 100],
        ),
        legend=dict(
            x=0.1,
            y=1.0,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='grey'
        ),
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        height=600
    )

    # Update trace colors for scatter plot
    for trace in scatter_fig.data:
        trace.line.color = 'navy'
        trace.marker.color = 'darkred'
    # Update the bar chart figure
    # Update the bar chart figure
    bar_chart_fig = go.Figure()
    if selected_team:
        bar_traces = create_bar_traces_for_team(selected_team, df_fixtures_2020)
        for trace in bar_traces:
            bar_chart_fig.add_trace(trace)

    # Calculate and add the average goals data
    average_goals_data = calculate_average_goals(df_fixtures_2020)
    bar_chart_fig.add_trace(
    go.Scatter(
        x=list(average_goals_data.keys()),
        y=list(average_goals_data.values()),
        mode='lines+markers',
        name='Average Goals Scored',
        marker=dict(color='black'),
        hoverinfo='y+name',
        hovertemplate='Matchday: %{x}<br>Average Goals Scored: %{y:.2f}<extra></extra>'
    )
)

    bar_chart_fig.update_layout(
        title={
            'text': f"Goals Per Game: {selected_team}",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                family="Arial, sans-serif",
                size=24,
                color="black"
            )
        },
        xaxis=dict(
            title="Matchday",
            titlefont=dict(
                family="Arial, sans-serif",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=False,
            linecolor='black',
            linewidth=2,
            tickfont=dict(
                family="Arial, sans-serif",
                size=14,
                color="black"
            ),
        ),
        yaxis=dict(
            title="Goals",
            titlefont=dict(
                family="Arial, sans-serif",
                size=18,
                color="black"
            ),
            showline=True,
            showgrid=True,
            gridcolor='lightgrey',
            linecolor='black',
            linewidth=2,
            tickfont=dict(
                family="Arial, sans-serif",
                size=14,
                color="black"
            ),
        ),
        legend=dict(
            x=0.5,
            y=-0.15,
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            font=dict(size=16, color='black')
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        barmode='stack',
        height = 600# Setting barmode to stack
    )
    
    team_stats_fig = go.Figure()
    team_data = combined_df[combined_df['Team'] == selected_team]
    add_team_stats_to_figure(team_data, team_stats_fig)
    team_stats_fig.update_layout(
        grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
        # Add common styling options here
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=5, b=20)
        # Include any other layout properties that you used in your original figure
    )


    
    

    return rank_text, form_text, scatter_fig, bar_chart_fig, test_fig, points_fig, team_stats_fig, star_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, use_reloader=False)


# In[ ]:




