import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
from sklearn.metrics import roc_curve, auc


# Data

df = pd.read_csv('../DATA/iris.csv')

df.head()

# Exploratory Data Analysis and Visualization

df.info()
df.describe()

df['species'].value_counts()

sns.countplot(df['species'])
sns.scatterplot(x='sepal_length',y='sepal_width',data=df,hue='species')
sns.scatterplot(x='petal_length',y='petal_width',data=df,hue='species')
sns.pairplot(df,hue='species')
sns.heatmap(df.corr(),annot=True)


df['species'].unique()

#3 dimensional plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
ax.scatter(df['sepal_width'],df['petal_width'],df['petal_length'],c=colors);

# Train | Test Split and Scaling

X = df.drop('species',axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

## Multi-Class Logistic Regression Model

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

# Depending on warnings you may need to adjust max iterations allowed 
# Or experiment with different solvers

log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000)

# GridSearch for Best Hyper-Parameters
# Penalty Type
penalty = ['l1', 'l2']

# Use logarithmically spaced C values (recommended in official docs)
C = np.logspace(0, 4, 10)

grid_model = GridSearchCV(log_model,param_grid={'C':C,'penalty':penalty})

grid_model.fit(scaled_X_train,y_train)

grid_model.best_params_

# Model Performance on Classification Tasks

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix
y_pred = grid_model.predict(scaled_X_test)

accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

plot_confusion_matrix(grid_model,scaled_X_test,y_test)

# Scaled so highest value=1
plot_confusion_matrix(grid_model,scaled_X_test,y_test,normalize='true')

print(classification_report(y_test,y_pred))

## Evaluating Curves and AUC


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(grid_model, scaled_X_test, y_test, n_classes=3, figsize=(16, 10))
