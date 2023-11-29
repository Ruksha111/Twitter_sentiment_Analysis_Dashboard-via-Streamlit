import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


st.title('Twitter Sentiment Analysis')


def load_data():
    df = pd.read_csv("train.csv", encoding='ISO-8859-1')
    return df

df = load_data()


if st.checkbox('Dataframe'):
    st.write(df)

st.write("Shape of DataFrame: ", df.shape)


# Bar Chart
st.subheader("Data Distribution for Each Class")
data_dist = list(df.groupby(by=["sentiment"]).count()["text"])
labels = df['sentiment'].sort_values().unique()

# Create the bar plot
fig, ax = plt.subplots()
ax.bar(labels, data_dist)

# Add text labels above the bars
for index, value in enumerate(data_dist):
    ax.text(index, value + 50, str(value), ha='center')

# Set titles and labels
ax.set_title("Data Distribution for each Class")
ax.set_xlabel("Sentiment")
ax.set_ylabel("No. of Distinct Utterances")

# Display the plot in Streamlit
st.pyplot(fig)

# Adding WordList and WordCount columns
df["WordList"] = df["text"].str.split(' ')
df["WordCount"] = df["WordList"].str.len()

# Display DataFrame in Streamlit
if st.checkbox('Show DataFrame with WordList and WordCount'):
    st.write(df)

# Plot histogram of word counts
st.subheader("Histogram of word count")
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(df["WordCount"])

plt.title("Histogram of word count")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
# Show plot
plt.show()

st.pyplot(fig)

# Plot a histogram of the 'Age of User' column
st.subheader("Age Distribution of Users")
plt.figure(figsize=(10, 6))
plt.hist(df['Age of User'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Users')
plt.xlabel('Age')
plt.ylabel('Number of Users')
plt.grid(True)
plt.show()
st.pyplot(plt)

st.subheader("Sentiment Distribution Over Time by Age Group")
#Plotting the Sentiment Distribution Over Time by Age Group
sentiments = ['negative', 'neutral', 'positive']

plt.figure(figsize=(15, 5))
# Loop through each sentiment, create a subplot for each
for i, sentiment in enumerate(sentiments, 1):
    plt.subplot(1, 3, i)
    subset = df[df['sentiment'] == sentiment]

    if not subset.empty:  # If the subset is not empty, create a count plot
        sns.countplot(data=subset, x='Time of Tweet', hue='Age of User')
        plt.title(f'Tweets for {sentiment} sentiment')# Create a countplot for 'Time of Tweet' categorized
        if i == 3:  #   For the last subplot, add a legend; for others, remove the legend
            plt.legend(title='Age of User', loc='upper right')
        else:
            plt.legend().remove()
    else: 
        # If the subset is empty, display a text message in the subplot
        plt.text(0.5, 0.5, f'No data for {sentiment} sentiment', horizontalalignment='center')

plt.tight_layout()
plt.show()
st.pyplot(plt)

st.subheader("Tweets over time")
# Plotting the Tweets over Time
sentiments = df['sentiment'].unique()## Retrieve unique sentiment values from the 'sentiment' column of the DataFrame

plt.figure(figsize=(20, 6))
# Loop through each unique sentiment value
for i, sentiment in enumerate(sentiments, 1):
    subset = df[df['sentiment'] == sentiment]
     # Group the subset by 'Time of Tweet' and count the number of tweets in each group
    tweet_counts = subset.groupby('Time of Tweet').size()

    plt.subplot(1, len(sentiments), i)
     #Plot the tweet counts as a pie chart
    tweet_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140,
                      colors=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title(f'Tweets Distribution for {sentiment.capitalize()} Sentiment')
    plt.ylabel('')  # Hide the ylabel

plt.tight_layout()
plt.show()
st.pyplot(plt)

st.subheader("Tweet Distribution by Top 10 Countries for Different Sentiments")
sentiments = df['sentiment'].unique()
# Setting up a figure with a dynamic height based on the number of sentiment categories
plt.figure(figsize=(20, 5 * len(sentiments)))
#Looping through each sentiment to create a separate plot
for i, sentiment in enumerate(sentiments, 1):
    subset = df[df['sentiment'] == sentiment]
     # Identifying the top 10 countries with the highest tweet counts for the current sentimen
    top_countries_subset = subset['Country'].value_counts().head(10).index
     # Creating a subplot for each sentiment with appropriate dimensions
    plt.subplot(len(sentiments), 1, i)
    sns.countplot(data=subset, x='Country', order=top_countries_subset)
    plt.title(f'Number of Tweets by Top 10 Countries for {sentiment.capitalize()} Sentiment')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
st.pyplot(plt)

st.subheader("Most Frequently Occuring Words - Top 30")
from sklearn.feature_extraction.text import CountVectorizer

# Handle missing values 
df['text'] = df['text'].fillna('')
# Initialize a CountVectorizer object with English stop words.
cv = CountVectorizer(stop_words = 'english')
# Apply the vectorizer to the 'text' column of the dataframe to convert text data into a numeric format.
words = cv.fit_transform(df.text)
#Sum the occurrences of each word in the corpus.
sum_words = words.sum(axis=0)
# Pair each word with its frequency in the corpus.
words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
#Sort the word-frequency pairs in descending order of frequency.
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

#Convert the sorted word-frequency pairs into a DataFrame.
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
plt.title("Most Frequently Occuring Words - Top 30")
st.pyplot(plt)

st.subheader("Most Frequently Occuring Positive Words - Top 10")
positive_texts = df[df['sentiment'] == 'positive']['text'] 



# Fit and transform the data
word_counts = cv.fit_transform(positive_texts)

# Summarize word counts
sum_words = word_counts.sum(axis=0)

# Create a frequency dictionary
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

# Sort words by frequency
sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Select top words
top_positive_words = sorted_words_freq[:10]  # Adjust the number as needed

# Prepare data for plotting
top_words = [word for word, freq in top_positive_words]
frequencies = [freq for word, freq in top_positive_words]

# Plotting
plt.figure(figsize=(15, 7))
plt.bar(top_words, frequencies, color='blue')
plt.title("Most Frequently Occuring Positive Words - Top 10")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
st.pyplot(plt)

st.subheader("Most Frequently Occuring Negative Words - Top 10")



negative_texts = df[df['sentiment'] == 'negative']['text']  
# Fit and transform the data
word_counts = cv.fit_transform(negative_texts)

# Summarize word counts
sum_words = word_counts.sum(axis=0)

# Create a frequency dictionary
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

# Sort words by frequency
sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Select top words
top_positive_words = sorted_words_freq[:10]  # Adjust the number as needed

# Prepare data for plotting
top_words = [word for word, freq in top_positive_words]
frequencies = [freq for word, freq in top_positive_words]

# Plotting
plt.figure(figsize=(15, 7))
plt.bar(top_words, frequencies, color='blue')
plt.title("Most Frequently Occuring Negative Words - Top 10")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
st.pyplot(plt)

st.subheader("Sentiment Analysis Trend Across Time Periods")
sentiment_over_time = df.groupby(['Time of Tweet', 'sentiment']).size().reset_index(name='tweet_count')

plt.figure(figsize=(15, 7))
sns.lineplot(data=sentiment_over_time, x='Time of Tweet', y='tweet_count', hue='sentiment', marker='o')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Time of Tweet')
plt.ylabel('Number of Tweets')
plt.legend(title='Sentiment')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
st.pyplot(plt)

st.subheader("Tweet Activity Distribution Throughout the Day")

tweet_counts = df['Time of Tweet'].value_counts()

# Plotting the counts
plt.figure(figsize=(10, 5))
tweet_counts.plot(kind='bar', title='Tweet Volume by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=0)  # Rotates the x-axis labels to be horizontal
plt.show()
st.pyplot(plt)


st.subheader("Word Clouds for Sentiments")
sentiment = st.radio("Choose sentiment", ('positive', 'neutral', 'negative'))

plt.figure(figsize=(20, 10))
wc = WordCloud(max_words=2000, width=1600, height=800, background_color='white').generate(" ".join(df[df.sentiment == sentiment].text))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

st.markdown("**Analysis:** The visual data analysis reveals a trend where tweets are mostly neutral, with lesser instances of positive and negative sentiments. When examining the tweet lengths, a significant number of tweet contain between 0 to 20 words. In terms of user , the age distribution is relatively even, suggesting broad engagement across age groups. Further analysis of sentiment-related tweet patterns indicates that younger users (0-20 years) tend to post negative tweets around noon, while also being the most active in the morning for neutral tweets. Interestingly, users aged 60-70 are more inclined to share positive tweets at midday. Overall, the distribution of tweets by sentiment remains fairly consistent throughout the day, with slight variations in frequency across different times.")

