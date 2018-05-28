
# coding: utf-8

# In[252]:

import numpy as np
# import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# import statsmodels.formula.api as smf
import pandas as pd
# import pyodbc
# import glob
import datetime
import glob
import pylab
import math
import operator
import statsmodels

# from numpy import linalg as LA
import seaborn as sns
sns.set_style("whitegrid")

from scipy.stats import wilcoxon, mannwhitneyu, entropy


# In[189]:

weight_goals = pd.read_csv("/remote/mgord/under_armour/derived_data/single_tables/mfp_weight_loss_goal_by_date.csv")


# In[38]:

weight_measurements = pd.read_csv("/remote/mgord/under_armour/derived_data/summary_tables/20180513_weights.csv")


# In[190]:

weight_goals["created_at"] = pd.to_datetime(weight_goals["created_at"])


# In[188]:

weight_goals[weight_goals["user_id"] == "dd0441ad-1c18-4c8d-85f8-68fe06b4dfee"]


# In[191]:

weight_goals = weight_goals.sort_values(["user_id", "created_at"])
weight_goals["prev_goal"] = weight_goals.groupby("user_id")["goal_weight"].shift(1)
weight_goals_filt = weight_goals.loc[weight_goals["prev_goal"] != weight_goals["goal_weight"]]


# # Filtering goals by criteria:
# 
# * Must have changed goal at least once
# * only users who's goal is lower than initial weight
# * max time between goals: 365 days
# * min time between goals: 30 days
# * goal change must be >= 1 lb

# In[192]:

weight_goals_filt.head()


# In[193]:

weight_goals_filt = weight_goals_filt.sort_values(["user_id", "created_at"])


# In[194]:

weight_goals_filt["next_goal_date"] = weight_goals_filt.groupby("user_id").created_at.shift(-1)
weight_goals_filt["next_goal_diff"] = weight_goals_filt.next_goal_date - weight_goals_filt.created_at


# In[195]:

losers = weight_goals_filt.loc[weight_goals_filt["goal_weight"] < weight_goals_filt["initial_weight"]]


# In[196]:

losers


# In[197]:

losers["next_goal"] = losers.groupby("user_id").goal_weight.shift(-1)


# In[198]:

losers = losers.loc[np.abs(losers["next_goal"] - losers["goal_weight"]) > 1.0]


# In[199]:

losers = losers.loc[losers["next_goal_diff"].between("30 days", "366 days")]


# In[200]:

losers.groupby("user_id").size().hist(bins=range(0, 10))


# In[15]:

losers.to_csv("/remote/mgord/under_armour/derived_data/single_tables/cs230_2018-05-08_weight_goals", index=False)


# In[37]:

# user_group = "00"
# # weights = pd.read_csv("/remote/mgord/under_armour/derived_data/weight_measurements_by_userid/weight_measurements_%s.csv" %user_group)
# weights = pd.read_csv("/remote/mgord/under_armour/derived_data/single_tables/mfp_weight_measurements.csv")


# In[26]:

# weights["updated_at"] = pd.to_datetime(weights["updated_at"])


# In[38]:

# weights.head()


# In[ ]:

# for index, row in losers.head(100).iterrows():
#     start_date = row["created_at"]
#     end_date = row["next_goal_date"]
#     user_id = row["user_id"]
#     print(user_id)
#     goal_weights = weights.loc[(weights["user_id"] == user_id)]
# #     & (weights["updated_at"].between(start_date, end_date, inclusive=True)
#     print(goal_weights)


# In[202]:

len(weight_measurements)


# In[203]:

filt_weights = weight_measurements.loc[weight_measurements["weight_goal_id"].isin(losers["id"])]


# In[205]:

len(filt_weights)


# In[206]:

filt_weights["last_weight_date"] = pd.to_datetime(filt_weights["last_weight_date"])
filt_weights["first_weight_date"] = pd.to_datetime(filt_weights["first_weight_date"])
filt_weights["updated_at"] = pd.to_datetime(filt_weights["updated_at"])


# In[207]:

filt_weights["period_length"] = filt_weights["last_weight_date"] - filt_weights["first_weight_date"]


# In[208]:

filt_weights["day_of_period"] = filt_weights["updated_at"] - filt_weights["first_weight_date"]
filt_weights["day_of_period_int"] = filt_weights["day_of_period"].dt.days


# In[209]:

filt_weights["week_of_period"] = np.floor(filt_weights["day_of_period_int"] / 7) + 1


# In[210]:

filt_weights["period_length_int"] = filt_weights["period_length"].dt.days


# In[211]:

filt_weights["period_length_int_bin"] = pd.cut(filt_weights["period_length_int"], bins=[0, 5, 10, 20, 30, 60, 90, 120, 200, 366])


# In[212]:

filt_weights.groupby(["weight_goal_id", "period_length_int_bin"]).size()


# In[213]:

filt_weights.groupby("weight_goal_id").size().hist(bins=range(0, 200))


# In[214]:

filt_weights["updated_at"] = pd.to_datetime(filt_weights["updated_at"])


# In[576]:

first_last = filt_weights.groupby("weight_goal_id").value.agg(['first', 'last', 'count']).reset_index()


# In[577]:

filt_weights_deduped = filt_weights.drop_duplicates(subset="weight_goal_id")


# In[578]:

first_last = first_last.merge(filt_weights_deduped[["weight_goal_id", "period_length_int"]], on="weight_goal_id")


# In[579]:

first_last


# In[ ]:

ts = filt_weights.set_index('updated_at')


# In[ ]:

resampled = ts.groupby("weight_goal_id").value.resample('W').mean()


# In[ ]:

interpolated = resampled.interpolate()


# In[ ]:

interpolated


# In[ ]:

interpolated = interpolated.reset_index()
interpolated["ct"] = interpolated.groupby("weight_goal_id").cumcount()


# In[393]:

interpolated


# In[379]:

A = interpolated.groupby(["weight_goal_id", "ct"]).value.mean().unstack(level=-1)


# In[380]:

A = A.reset_index()


# In[381]:

A = A.sort_values(["weight_goal_id"])


# In[425]:

A


# In[426]:

weights_x = A.copy()


# In[257]:

filt_losers = losers.loc[losers["id"].isin(filt_weights.weight_goal_id)]


# In[258]:

filt_losers = filt_losers.sort_values("id")


# In[378]:

filt_losers


# In[225]:

# TODO maybe???: fix the weights so that there is no such thing as an active period


# In[259]:

y = filt_losers["next_goal"] > filt_losers["goal_weight"]


# In[388]:

filt_losers["raised_goal"] = filt_losers["next_goal"] > filt_losers["goal_weight"]
raised_goals = filt_losers.loc[filt_losers["raised_goal"] == True]
lowered_goals = filt_losers.loc[filt_losers["raised_goal"] == False]


# In[394]:

raisers = interpolated.loc[interpolated["weight_goal_id"].isin(raised_goals["id"])]
lowerers = interpolated.loc[interpolated["weight_goal_id"].isin(lowered_goals["id"])]


# In[409]:

raisers['week_max'] = raisers.groupby(['weight_goal_id'])['ct'].transform(max)
lowerers['week_max'] = lowerers.groupby(['weight_goal_id'])['ct'].transform(max)


# In[396]:

import seaborn as sns; 


# In[525]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(5, 5)].head(500000), ci=None, x_bins=range(0, 40))


# In[526]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(6, 6)].head(500000), ci=None, x_bins=range(0, 40))


# In[527]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(7, 7)].head(500000), ci=None, x_bins=range(0, 40))


# In[520]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(10, 10)].head(500000), ci=None, x_bins=range(0, 40))


# In[522]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(15, 15)].head(500000), ci=None, x_bins=range(0, 40))


# In[521]:

sns.lmplot(x="ct", y="value", data=raisers.loc[raisers["week_max"].between(20, 20)].head(500000), ci=None, x_bins=range(0, 40))


# In[415]:

sns.lmplot(x="ct", y="value", data=lowerers.loc[lowerers["week_max"].between(19, 20)].head(500000), ci=None, x_bins=range(0, 40))


# In[244]:

users = pd.read_csv("/remote/althoff/under_armour/derived_data/single_tables/users_20170306.csv")


# In[262]:

A = A.merge(filt_losers[["id", "user_id", "goal_weight"]], left_on="weight_goal_id", right_on="id")


# In[263]:

A = A.merge(users[["age_years", "gender", "bmi_initial", "common_user_id"]], left_on="user_id", right_on="common_user_id")


# In[266]:

X = A
del X["id"]
del X["user_id"]
del X["common_user_id"]
del X["weight_goal_id"]


# In[424]:

X


# In[342]:

y.value_counts()


# In[269]:

from sklearn.cross_validation import train_test_split


# In[274]:

X["gender"] = (X["gender"] == 'f') * 1


# In[283]:

X_np = X.values
y_np = y.values


# In[287]:

y_np = y_np * 1


# In[289]:

(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.10, random_state=42)


# In[277]:

from keras.models import Sequential
from keras.layers import Dense, Activation


# In[416]:

X_np


# In[290]:

X_np.shape


# In[546]:

# TODO: convert things to categorial


# In[580]:

first_last_X = first_last.copy()
del first_last_X['weight_goal_id']
first_last_X = first_last_X.values


# In[588]:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=4))
model.add(Dense(300, activation='relu', input_dim=32))
# model.add(Dense(100, activation='relu', input_dim=100))
# model.add(Dense(50, activation='relu', input_dim=100))
# model.add(Dense(10, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(first_last_X[0:100000], y_np[0:100000], epochs=5, validation_split=0.1)


# In[340]:

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=58))
model.add(Dense(100, activation='relu', input_dim=32))
model.add(Dense(100, activation='relu', input_dim=100))
model.add(Dense(10, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(X_np_copy[0:10000], y_np[0:10000], epochs=4, batch_size=32, validation_split=0.1)


# In[354]:

X_np[60].T


# In[ ]:

plt.plot(X_np[60][0:50].T)


# In[370]:

plt.plot(X_np[90][0:50].T)
plt.xlabel("Weeks")
plt.ylabel("Weight (in lbs)")
plt.title("Current goal: 120. Changed goal: made harder")


# In[374]:

i=500
plt.plot(X_np[i][0:50].T)
plt.xlabel("Weeks")
plt.ylabel("Weight (in lbs)")
plt.title("Current goal: 115. Changed goal: made harder")
print(y_np[i])


# In[369]:

y_np[90]


# In[373]:

X_np[500]


# In[367]:

plt.plot(X_np[62][0:50].T)
plt.xlabel("Weeks")
plt.ylabel("Weight (in lbs)")
plt.title("Current goal: 150. Changed goal: made easier")


# In[334]:

X_np_copy = np.nan_to_num(X_np)


# In[337]:

np.array(model.layers[0].get_weights())[0].shape


# In[338]:

np.array(model.layers[0].get_weights())[0]


# In[321]:

for layer in model.layers:
    weights = layer.get_weights()


# In[322]:

weights


# In[341]:

for i in range(100):
    print(model.predict_classes(X_np[i].reshape(1, 58)))


# In[317]:

ynew


# # LSTM

# In[419]:

X_np_copy


# In[428]:

del weights_x["weight_goal_id"]


# In[441]:

weights_x_arr = weights_x.values


# In[444]:

shortened_lists = []
for arr in weights_x_arr:
    shortened_lists.append(arr[~np.isnan(arr)])


# In[445]:

shortened_lists


# In[446]:

from keras.preprocessing.sequence import pad_sequences
# define sequences
# pad sequence
padded = pad_sequences(shortened_lists, padding='post')
print(padded)


# In[501]:

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking


# In[491]:

padded_reshaped = padded.reshape(351010, 54, 1)


# In[492]:

padded_reshaped[:, 0:1000, :].shape


# In[493]:

padded_reshaped.shape


# In[ ]:

# Make it work on1 example
# Then try two examples of two different classes


# In[ ]:

model = Sequential()
M = Masking(mask_value=0, input_shape=(54, 1))
model.add(M)
model.add(LSTM(500, input_dim=1, activation="tanh"))
model.add(Dense(128, activation='relu', input_dim=500))
model.add(Dense(32, activation='relu', input_dim=128))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(32, activation='relu', input_dim=58))
# model.add(Dense(100, activation='relu', input_dim=32))
# model.add(Dense(100, activation='relu', input_dim=100))
# model.add(Dense(10, activation='relu', input_dim=100))
# model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(padded_reshaped[0:5000, :, :], y_np[0:5000], epochs=10, batch_size=32, validation_split=0.0)


# In[516]:

for i in range(100):
    print(model.predict_classes(padded_reshaped[i].reshape(1, 54, 1)))


# In[ ]:

# Try next: only look at people between 10 and 20 weeks. Same for y.


# In[528]:

padded_reshaped


# In[ ]:



