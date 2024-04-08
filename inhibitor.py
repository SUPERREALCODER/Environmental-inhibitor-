#importing all dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from google.colab import drive
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl

'''data preprocessing begins'''
#reading the original datasheet
url="https://raw.githubusercontent.com/SUPERREALCODER/Environmental-inhibitor-/main/AL_AA2024%20-%20Sheet1.csv"
data = pd.read_csv(url)

data.head()

#dropping the data which were not got
from google.colab import data_table
data1 = data.drop([5,19,20,21,22,23,24,25,26,31,32,33,34,35,40,41,51,91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334,259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 335, 336, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,29])
dt = data_table.DataTable(data1)
data2 = data1
display(dt)


#finding the mean and median of the dropped  datasheet
data1['Cost/100 gm'] = pd.to_numeric(data1['Cost/100 gm'], errors='coerce')
mean = data1['Cost/100 gm'].mean()
print("mean",mean)
median = data1['Cost/100 gm'].median()
print("median",median)
plt.plot(data1['Cost/100 gm'],marker='o')
plt.xlabel('Index')
plt.ylabel('Cost/100 gm')
plt.title('Numeric Values of Cost/100 gm Column')
plt.show()
'''data preprocessing ends  and the mean medians are used to fill the values 
of the cost of the inhibitors which we did not get'''

#text preprocessing for toxicity
url1="https://raw.githubusercontent.com/SUPERREALCODER/Environmental-inhibitor-/main/Copy%20of%20AL_AA2024%20for%20machine%20learning%20preprocessing%20-%20Sheet1.csv"
data_pre = pd.read_csv(url1)

#converting text to lower case and tokenizing text
def preprocess_text(text):
  #convert text to lowercase
  text = text.lower()

  #remove special characters and digits using regular expressions
  text = re.sub(r'\d+', '',text)#remove digits
  text = re.sub(r'[^\w\s]', '',text)#remove special characters
   #tokenize text
  tokens = nltk.word_tokenize(text)

  return(tokens)

#removing stopwords
def remove_stopwords(tokens):
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word.lower() not in stop_words ]
  return(filtered_tokens)

#performing lemmatzation
def perform_lemmatzation(tokens):
  lemmatizer = nltk.WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
  return(lemmatized_tokens)

#cleaning text
def clean_text(text):
  tokens = preprocess_text(text)
  filtered_tokens = remove_stopwords(tokens)
  lemmatized_tokens = perform_lemmatzation(filtered_tokens)
  clean_text = ' '.join(lemmatized_tokens)
  return clean_text
for i in range(343):
  text1=data_pre['Toxicity'][i]
  if isinstance(text1, str):
    data_pre['Toxicity'][i]=clean_text(text1)

#importing text processed file to google drive for ml application

drive.mount('/content/drive')
d0 = pd.DataFrame(data_pre)
d0.to_csv('/content/drive/My Drive/mydata.csv', index=False)

#again dropping the rows for which we have got toxicity
data_predrop = data_pre.drop([5,19,20,21,22,23,24,25,26,31,32,33,34,35,40,41,51,91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334])
df_pre = data_table.DataTable(data_predrop)
display(df_pre)

#removing the empty errors of the data sheet
empty_values = data_predrop['Toxicity'].isna().sum()

print(empty_values)
#troubleshooting some errors in datasheet for row index 16 ,54 and 55
data_predrop['Toxicity'][16] = "damaging fertility unborn child skin irritation"
print(data_predrop['Toxicity'][16] )


'''performing the k means clustering'''

#performing elbow method to find k-means clustering 


# Read the CSV data sheet into a Pandas DataFrame
df = pd.read_csv('https://raw.githubusercontent.com/SUPERREALCODER/Environmental-inhibitor-/main/adjusted%20data%20with%20median%20and%20mode%20-%20Sheet1.csv')

# Select the specific column to use
column = 'Toxicity'

# Perform one-hot encoding on the column
one_hot_encoder = OneHotEncoder(sparse=False)
encoded_values = one_hot_encoder.fit_transform(df[[column]])

# Create a list of values for the number of clusters to try
k_range = range(1, 10)

# Create an empty list to store the WCSS values for each number of clusters
wcss = []

# Iterate over the list of k values, and for each value, fit a KMeans model to the data and calculate the WCSS
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(encoded_values)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values against the number of clusters
plt.plot(k_range, wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.savefig('elbow.jpg')
files.download('elbow.jpg')


vectorizer = TfidfVectorizer()
# Fit the vectorizer to the data
vectorizer.fit(data_predrop['Toxicity'])
# Transform the data into tf-idf vectors
X = vectorizer.transform(data_predrop['Toxicity'])
# Create a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans object to the data
kmeans.fit(X)

# Predict the cluster labels for each document
labels = kmeans.predict(X)

# Print the cluster labels
print(labels)


#clustering data based on toxicity with labels with k=4
column = data_predrop['Toxicity']
le = LabelEncoder()
le.fit(column)
column_encoded = le.transform(column)
column_encoded = column_encoded.reshape(-1, 1)
kmeans = KMeans(n_clusters=4)
kmeans.fit(column_encoded)
cluster_labels = kmeans.labels_
data_predrop['cluster_label'] = cluster_labels
data_predrop.to_csv('clustered_data.csv', index=False)


'''plotting the scattered graph of k-means clustering'''
column = data_predrop['Toxicity']
le = LabelEncoder()
le.fit(column)
column_encoded = le.transform(column)
column_encoded = column_encoded.reshape(-1, 1)
kmeans = KMeans(n_clusters=4)
kmeans.fit(column_encoded)
cluster_labels = kmeans.labels_
data_predrop['cluster_label'] = cluster_labels
data_predrop.to_csv('clustered_data.csv', index=False)


clustered_data1 = pd.read_csv('clustered_data1.csv')
df_cluster = data_table.DataTable(clustered_data1)
display(df_cluster)



# Assuming 'data_predrop' is your DataFrame containing the data
column = clustered_data['Toxicity']

# Encode the 'Toxicity' column using LabelEncoder
le = LabelEncoder()
column_encoded = le.fit_transform(column)
column_encoded = column_encoded.reshape(-1, 1)

# Perform K-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(column_encoded)
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
clustered_data['cluster_label'] = cluster_labels

# Plot the clusters
plt.figure(figsize=(8,6))
for cluster_label in range(4):
    cluster_data1 = clustered_data[clustered_data['cluster_label'] == cluster_label]
    plt.scatter(cluster_data1.index, cluster_data1['Toxicity'], label=f'Cluster {cluster_label}')

plt.title('K-means Clustering of Toxicity Data')
plt.xlabel('Index')
plt.ylabel('Inhibitor')
plt.legend()
plt.grid(False)
plt.show()

# Save the clustered data to a CSV file
clustered_data.to_csv('clustered_data1.csv', index=False)



#displaying the clustered data sheet
clustered_data = pd.read_csv('clustered_data.csv')
df_cluster = data_table.DataTable(clustered_data)
display(df_cluster)

#saving the clustered data sheet for further use

drive.mount('/content/drive')
d0 = pd.DataFrame(clustered_data)
d0.to_csv('/content/drive/My Drive/mydata.csv', index=False)


'''plotting efficiency vs Cost/100 gm of the cluster which is most environment friendly to identify the most efficient and economic feasible one  '''
# Read the CSV file into a Pandas DataFrame
d_plot = pd.read_csv('https://raw.githubusercontent.com/SUPERREALCODER/Environmental-inhibitor-/main/1%20cluster%20data%20sheet%20-%20Sheet1.csv')

# Select the two columns that you want to plot
d_plot['Cost/100 gm'] = d_plot['Cost/100 gm'].str.replace(',', '')

d_plot['Cost/100 gm']=pd.to_numeric(d_plot['Cost/100 gm'])
d_plot['Efficiency']=pd.to_numeric(d_plot['Efficiency'])
x = d_plot['Cost/100 gm']
y = d_plot['Efficiency']

# Create a scatter plot
plt.scatter(x, y)

# Add the specific marked index on each point
for i, point in enumerate(x):
    plt.text(point, y[i], str(d_plot['Unnamed: 0'][i]))

# Set the title and labels of the plot
plt.title('Plot of Efficiency vs Cost/100 gm')
plt.xlabel('Cost/100 gm')
plt.ylabel('Efficiency')

# Display the plot
plt.show()

