import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

# Load data
data = pd.read_csv("instagram_analysis.csv", encoding='latin1')
print(data.head())

# Drop missing values
data = data.dropna()

# Info about the data
data.info()

# Distribution plots (update sns.distplot to sns.histplot)
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.histplot(data['From Home'], kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.histplot(data['From Hashtags'], kde=True)
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.histplot(data['From Explore'], kde=True)
plt.show()

# Pie chart of Impressions
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()

# Word cloud for Captions
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Word cloud for Hashtags
text = " ".join(i for i in data.Hashtags)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Scatter plots showing relationships
figure = px.scatter(data_frame=data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title="Relationship Between Likes and Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title="Relationship Between Comments and Total Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols", 
                    title="Relationship Between Shares and Total Impressions")
figure.show()

figure = px.scatter(data_frame=data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols", 
                    title="Relationship Between Post Saves and Total Impressions")
figure.show()

# Correlation calculation (exclude non-numeric columns)
numeric_data = data.select_dtypes(include=[np.number])
correlation = numeric_data.corr()
print(correlation["Impressions"].sort_values(ascending=False))

# Conversion rate
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)

# Scatter plot for profile visits and follows
figure = px.scatter(data_frame=data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title="Relationship Between Profile Visits and Followers Gained")
figure.show()

# Train model
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

# Model 
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model_score = model.score(xtest, ytest)
print(f"Model score: {model_score}")

# Predict with sample features
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
prediction = model.predict(features)
print(f"Prediction: {prediction}")
