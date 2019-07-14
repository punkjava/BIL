from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

app.config['SECRET_KEY'] = 'you-will-never-guess'

class LinRegForm(FlaskForm):
    fileName = SelectField('Select one dataset', choices = [('Social_Network_Ads.csv', 'Social Network'), ('Mall_Customers.csv', 'Mall Custiomers Data')])
    submit = SubmitField('Submit')

'''
class ClusForm(FlaskForm):
    clusFile = SelectField('Select one dataset', choices = [('Salary_Data.csv', 'Salary Data'), ('Mall_Customers.csv', 'Mall Custiomers Data')])
    clusSubmit = SubmitField('Submit')
'''

@app.route('/', methods = ['GET', 'POST'])
def dashboard():
   return render_template('index.html')

@app.route('/linReg', methods=['GET', 'POST'])
def linReg():
    fileName = False
    form = LinRegForm()
    if form.validate_on_submit():
        fileName = form.fileName.data

        # Simple Linear Regression

        # Importing the dataset
        dataset = pd.read_csv('datasets/' + fileName)
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() + 1.5, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() + 1.5, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Naive Bayes (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

        # Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Naive Bayes (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

    return render_template('linearRegression.html', form = form, fileName = fileName)
'''
@app.route('/clus', methods = ['GET', 'POST'])
def clus():
    clusFile = False
    clusForm = ClusForm()
    if clusForm.validate_on_submit():
        clusFile = clusForm.clusFile.data

        # Importing the dataset

        # Hierarchical Clustering

    # Importing the dataset
    dataset = pd.read_csv('datasets/' + clusFile)
    dataset = pd.read_csv('Mall_Customers.csv')
    X = dataset.iloc[:, [3, 4]].values
    # y = dataset.iloc[:, 3].values



    # Using the dendrogram to find the optimal number of clusters
    import scipy.cluster.hierarchy as sch
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

    # Fitting Hierarchical Clustering to the dataset
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)

    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
    return render_template('clustering.html', clusForm = clusForm, clusFile = clusFile)
'''

if __name__ == '__main__':
   app.run(debug = False)
