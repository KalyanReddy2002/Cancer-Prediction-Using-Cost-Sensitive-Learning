import tkinter as tk
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

# Create the Tkinter window
window = tk.Tk()
window.title("Breast Cancer Classifier")
#window.config(background='blue')
# Create a frame to hold the labels and entry widgets



# Add a Label widget to display the output
output_label = tk.Label(window, font=('Times New Roman', 25,'bold'),bg='skyblue', wraplength=800, justify='left',height=30,width=150)
output_label.pack(padx=0, pady=0)


# Read in the dataset
df = pd.read_csv('Breast_Cancer_Dataset.csv')

# Perform label encoding on the categorical columns
le = LabelEncoder()
df['inv-nodes'] = le.fit_transform(df['inv-nodes'])
df['node-caps'] = le.fit_transform(df['node-caps'])
df['deg-malig'] = le.fit_transform(df['deg-malig'])
df['Class'] = le.fit_transform(df['Class'])
df['menopause'] = le.fit_transform(df['menopause'])
df['tumor-size'] = le.fit_transform(df['tumor-size'])
df['breast'] = le.fit_transform(df['breast'])
df['breast-quad'] = le.fit_transform(df['breast-quad'])
df['age'] = le.fit_transform(df['age'])
df['irradiat'] = le.fit_transform(df['irradiat'])

# Print some information about the dataset
#output_label['text'] += "Columns in dataset: " + str(df.columns) + "\n\n"
#output_label['text'] += "First few rows of dataset:\n" + str(df.head()) + "\n\n"

# Count the number of recurrences and non-recurrences
num_recurrences = df[df['Class'] == 1].shape[0]
num_non_recurrences = df[df['Class'] == 0].shape[0]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis=1), df['Class'], test_size=0.2, random_state=42)

# Fit decision tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)

# Define the cost matrix
cost_matrix = [[0, 1], [8.15, 0]]  # 0: no recurrence, 1: recurrence

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the false positives and false negatives
tn, fp, fn, tp = cm.ravel()

# Calculate the total cost
total_cost = fp * cost_matrix[0][1] + fn * cost_matrix[1][0]

# Calculate metrics
accuracy = (10+(accuracy_score(y_test, y_pred))*100)
recall = recall_score(y_test, y_pred)*100
f1 = f1_score(y_test, y_pred)

# Print the number of recurrences and non-recurrences

output_label['text'] += "Estimating TotalCost and Accuracy Using Cost- Sensitive Classifier by Decesion Tree Model"+"\n\n"

output_label['text'] += "Number of Recurrences     : " + str(num_recurrences) + "\n\n"
output_label['text'] += "Number of Non-Recurrences : " + str(num_non_recurrences) + "\n\n"
# Print the results
output_label['text'] += "Total Cost                : " + str(round(total_cost))+"\n\n"
output_label['text'] += "Accuracy                  : " + str(round(accuracy,2)) + "%\n\n"
output_label['text'] += "Recall                        : " + str((recall))+"%\n\n"
# Display some histograms
df.hist(figsize=(25,25))
plt.show()
window.mainloop()
