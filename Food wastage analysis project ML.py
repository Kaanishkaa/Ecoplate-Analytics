#!/usr/bin/env python
# coding: utf-8

# 

# In[72]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly.express as px


# In[73]:


#importing dataset
df=pd.read_csv("Data.csv")


# In[74]:


df.info()


# In[75]:


plt.figure(figsize=(10, 6))  # Adjust the size as per your preference
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of DataFrame')
plt.show()


# In[76]:


df.describe()


# In[77]:


df.head()


# Here we can see that the loss percentage and loss percentage original column is object datatype. We will convert it to numeric for any calculations ahead. The column loss_percentage_original has % in every variable and loss_quantity has kgs, we need to get rid of that as well.

# In[78]:


# Changing datatype of columns and removing units


df["loss_percentage"]=pd.to_numeric(df["loss_percentage"],errors='coerce')
df["loss_percentage_original"]=pd.to_numeric(df["loss_percentage_original"].str.rstrip('%'),errors='coerce')
df["loss_quantity"]=pd.to_numeric(df["loss_quantity"].str.rstrip('kgs'),errors='coerce')


# In[79]:


df.info()


# In[80]:


# Cleaning the data
# Deleting duplicate rows
df.dropna()
df


# As we can see in the original dataset we had 27743 data entries and after dropping duplicate rows we still got 27743 rows thereby showing that the dataset has all unique data entries.

# In[81]:


#Calculating how many missing values are there in each columns
missing_values=df.isnull().sum()
missing_values


# All these missing values need to be replaced with some values and drop unnecessary columns.

# In[82]:


# Dropping unnecessary columns
drop_columns=['loss_percentage_original','sample_size','url','notes','loss_quantity']
df=df.drop(drop_columns,axis=1)
df.head()


# In[83]:


# Replacing NaN values in the columns 'region','activity','cause_of_loss','treatment','reference'

df['region'].fillna('Unknown',inplace=True)
df['activity'].fillna('Unknown',inplace=True)
df['cause_of_loss'].fillna('Unknown',inplace=True)
df['treatment'].fillna('Unknown',inplace=True)
df['reference'].fillna('Unknown',inplace=True)
df['method_data_collection'].fillna('Unknown',inplace=True)

# Replacing missing values in food_supply_stage with 'Storage' as default value

df['food_supply_stage'].fillna('Storage',inplace=True)

# Replacing missing values in treatment with 'To be determined'

df['treatment'].fillna('To be determined',inplace=True)


# In[84]:


#Checking for missing values once more
missing_values=df.isnull().sum()
missing_values


# In[85]:


df.info()
df.to_csv('CleanData.csv', index=False)  


# In[86]:


# Box plot for "loss_percentage" column
plt.figure(figsize=(10,8))
sns.boxplot(x='loss_percentage',data=df)
plt.title("Box plot for Loss Percentage")
plt.show()


# In[87]:


# Scatter plot for "loss_percentage" column against 'year'

plt.figure(figsize=(10,8))
sns.scatterplot(x='year',y='loss_percentage',data=df)
plt.title("Scatter plot for loss_percentage column against year")
plt.show()


# In[88]:


# Correlation heatmap for the numerical columns

correlation_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[89]:


# Handling outliers (you can adjust the threshold based on your specific case)
#outlier_threshold = 2.5
#df_no_outliers = df[df['loss_percentage'] < outlier_threshold]


# In[90]:


#Q1 = df['loss_percentage'].quantile(0.25)
#Q3 = df['loss_percentage'].quantile(0.75)
#IQR = Q3 - Q1

#df_no_outliers = df[(df['loss_percentage'] >= (Q1 - 1.5 * IQR)) & (df['loss_percentage'] <= (Q3 + 1.5 * IQR))]


# In[91]:


# Display the cleaned dataset after removing outliers
#print("\nCleaned Dataset (without outliers):")
#print(df_no_outliers)


# In[92]:


# Display the shape of the cleaned dataset
#print("\nShape of Cleaned Dataset:")
#print(df_no_outliers.shape)


# In[93]:


"""
from scipy import stats

z_scores = stats.zscore(df['loss_percentage'])
df_no_outliers1 = df[(z_scores < 3) & (z_scores > -3)]
print("\nShape of Cleaned Dataset:")
print(df_no_outliers1.shape)
"""


# In[94]:


# Pairplot for numerical columns

sns.pairplot(df[['loss_percentage','year']],height=3)
plt.suptitle('Pairplot for loss percentage and year',y=1.05)
plt.show()


# In[95]:


# Distribution of loss_percentage column

plt.figure(figsize=(10,6))
sns.histplot(df['loss_percentage'],bins=30,kde=True,color='red')
plt.title("Distribution of Loss Percentage")
plt.xlabel("Loss Percentage")
plt.ylabel("Frequency")
plt.show()


# In[96]:


plt.figure(figsize=(20, 10))
sns.boxplot(x='region', y='loss_percentage', data=df, palette='viridis')
plt.title('Boxplot of Loss Percentage by Region', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=12)  # Adjust rotation angle and font size
plt.ylabel('Loss Percentage', fontsize=14)
plt.xlabel('Region', fontsize=14)
plt.show()


# In[97]:


plt.figure(figsize=(15, 8))  # Adjust the aspect ratio of the figure
sns.boxplot(x='region', y='loss_percentage', data=df, palette='viridis')
plt.title('Boxplot of Loss Percentage by Region', fontsize=18)  # Increase title font size
plt.xticks(rotation=45, ha='right', fontsize=14)  # Adjust rotation angle and increase font size
plt.ylabel('Loss Percentage', fontsize=16)  # Increase ylabel font size
plt.xlabel('Region', fontsize=16)  # Increase xlabel font size
plt.show()


# In[98]:


# Bar plot for the top 10 countries with highest amount of food loss percentage
top_countries = df.groupby('country')['loss_percentage'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_countries.index,y=top_countries.values,palette='muted')
plt.title('Top 10 Countries with Highest Average Loss Percentage')
plt.xticks(rotation=45,ha='right')
plt.ylabel('Average Loss Percentage')
plt.show()


# In[99]:


# Line plot for loss percentage over the year 1965-2022
plt.figure(figsize=(12,6))
sns.lineplot(x='year',y='loss_percentage',data=df,ci=None,color='orange')
plt.title('Loss Percentage over the years')
plt.xlabel('Year')
plt.ylabel('Loss Percentage')
plt.show()


# In[100]:


# Count plot for the distribution of food_supply_stage

plt.figure(figsize=(12,6))
sns.countplot(x='food_supply_stage',data=df,palette='pastel')
plt.title('Distribution of food supply stages')
plt.xticks(rotation=45,ha='right')
plt.xlabel('Food Supply Stage')
plt.ylabel('Count')
plt.show()


# In[101]:


# 8. Violin plot for loss_percentage by activity
plt.figure(figsize=(12, 8))
sns.violinplot(x='activity', y='loss_percentage', data=df, palette='Set2')
plt.title('Violin Plot of Loss Percentage by Activity')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Activity')
plt.ylabel('Loss Percentage')
plt.show()


# In[102]:


#Boxplot for loss_percentage by treatment
plt.figure(figsize=(12, 8))
sns.boxplot(x='treatment', y='loss_percentage', data=df, palette='Set3')
plt.title('Boxplot of Loss Percentage by Treatment')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Treatment')
plt.ylabel('Loss Percentage')
plt.show()


# In[103]:


'''
plt.figure(figsize=(12, 8))
sns.swarmplot(x='treatment', y='loss_percentage', data=df, palette='Set3')
plt.title('Swarm Plot of Loss Percentage by Treatment')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Treatment')
plt.ylabel('Loss Percentage')
plt.show()
'''


# In[104]:


# Scatter plot for loss_percentage vs. year colored by region
plt.figure(figsize=(12, 8))
sns.scatterplot(x='year', y='loss_percentage', hue='region', data=df, palette='Set1')
plt.title('Scatter Plot of Loss Percentage vs. Year (Colored by Region)')
plt.xlabel('Year')
plt.ylabel('Loss Percentage')
plt.savefig('my_plot.png')
plt.show()


# In[105]:


plt.figure(figsize=(50,25))
scatter_plot = sns.scatterplot(x='year', y='loss_percentage', hue='region', data=df, palette='Set1')
scatter_plot.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Region')
plt.title('Scatter Plot of Loss Percentage vs. Year (Colored by Region)')
plt.xlabel('Year')
plt.ylabel('Loss Percentage')
plt.savefig('my_plot.png', bbox_inches='tight')  
plt.show()


# In[106]:


plt.figure(figsize=(12, 8))
sns.barplot(x='treatment', y='loss_percentage', data=df, palette='Set3', ci=None)
plt.title('Bar Plot of Mean Loss Percentage by Treatment')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Treatment')
plt.ylabel('Mean Loss Percentage')
plt.show()


# In[107]:


# Getting the top 20 treatments 
top_treatments = df['treatment'].value_counts().head(20).index.tolist()
df_top_treatments = df[df['treatment'].isin(top_treatments)]

plt.figure(figsize=(20, 15))
sns.barplot(x='treatment', y='loss_percentage', data=df_top_treatments, palette='Set3', ci=None)
plt.title('Bar Plot of Mean Loss Percentage for Top 20 Treatments')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Treatment')
plt.ylabel('Mean Loss Percentage')
plt.savefig('my_plot2.png') 
plt.show()


# In[108]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[109]:


# Performing clustering

# Firstly selecting numeric data type for clustering
numeric_data = df.select_dtypes(include=['float64', 'int64'])


# In[110]:


# number of clusters (k)
k = 3


# In[111]:


# Initializing and fitting k-means model
kmeans = KMeans(n_clusters=k)
kmeans.fit(numeric_data)


# In[112]:


cluster_labels = kmeans.labels_


# In[113]:


df['Cluster'] = cluster_labels


# In[114]:


print(df.head())


# In[115]:


# Calculating silhouette score to determine optimal k
silhouette_avg = silhouette_score(numeric_data, cluster_labels)
print("Silhouette Score:", silhouette_avg)


# In[116]:


sns.scatterplot(x='year', y='loss_percentage', hue='Cluster', data=df, palette='tab10')
plt.title('K-means Clustering')
plt.xlabel('Year')
plt.ylabel('Loss Percentage')
plt.savefig('clustering_image.png')
plt.show()


# In[117]:


k_values = [2, 3, 4]


# In[118]:


# Iterate over different k values
for k in k_values:
    # Initialize and fit k-means model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(numeric_data)
    
    # Get cluster labels
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(numeric_data, cluster_labels)
    
    # Print silhouette score for each k
    print(f"Silhouette Score for k={k}: {silhouette_avg}")


# In[119]:


df.info()


# In[120]:



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[121]:


print(df.columns)


# In[122]:


X = df.drop(['country', 'region', 'cpc_code', 'commodity', 'activity', 'food_supply_stage', 'treatment', 'cause_of_loss', 'method_data_collection', 'reference', 'Cluster'], axis=1)
y = df['Cluster']


# In[123]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[124]:


# Initialize the decision tree classifier
clf = DecisionTreeClassifier()


# In[125]:


# Fit the classifier to the training data
clf.fit(X_train, y_train)


# In[126]:


print(clf.tree_.max_depth)


# In[127]:


X_train


# In[128]:


df.info()


# In[129]:


plt.figure(figsize=(10, 6))  # Adjust the size as per your preference
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of DataFrame')
plt.show()


# In[130]:


y_train


# In[131]:


X_test


# In[132]:


y_test


# In[133]:


# Predict on the test data
y_pred = clf.predict(X_test)


# In[134]:


# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[135]:


print(clf.tree_.max_depth)


# In[136]:


#pip install graphviz


# In[137]:


#from sklearn.tree import export_graphviz
#import graphviz


# In[138]:


from sklearn.tree import DecisionTreeClassifier, export_text


clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

# Generate textual representation of the decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)


# In[139]:


# 1. Default Decision Tree
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)


# In[140]:


# 2. Tuned Decision Tree
dt_tuned = DecisionTreeClassifier(max_depth=3, min_samples_split=2, random_state=42)
dt_tuned.fit(X_train, y_train)


# In[141]:


# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# In[ ]:


# Evaluate the models
models = [('Default Decision Tree', dt_default),
          ('Tuned Decision Tree', dt_tuned),
          ('Random Forest', rf)]

for name, model in models:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{name} Confusion Matrix')
    plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


# Predict on the test data
y_pred = clf.predict(X_test)


# In[ ]:


# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:



print("Confusion Matrix:")
print(cm)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# In[ ]:


text_features = ['activity', 'food_supply_stage', 'treatment', 'cause_of_loss', 'method_data_collection', 'reference']
X_text = df[text_features].apply(lambda x: ' '.join(x), axis=1)


# In[ ]:


X = X_text
y = df['Cluster']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


X_train


# In[ ]:


X_test


# In[ ]:


y_train


# In[ ]:


y_test


# In[ ]:


# Vectorizing the text data into feature vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test) 


# In[ ]:


# Initialize the Multinomial Naive Bayes classifier
clf = MultinomialNB()


# In[ ]:


# Fitting the classifier to the training data
clf.fit(X_train_vec, y_train)


# In[ ]:


# Predicting on the test data
y_pred = clf.predict(X_test_vec)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


df


# In[ ]:




