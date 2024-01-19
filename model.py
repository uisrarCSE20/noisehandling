import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class dataClassifier:
    def __init__(self, X, y):
        self.precision = int(0)
        self.recall = int(0)
        self.f_score = int(0)
        self.accuracy = int(0)
        self.matrix = ""
        self.global_y_test = ""
        self.global_y_pred = ""
        self.X = X
        self.y = y
        self.shapiro = ""
        
    def dataAnalysis(self, estimator):
        
        # Create training and test split
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=1)
        
        self.global_y_test = y_test
        
        # creating a RF classifier
        clf = RandomForestClassifier(n_estimators = estimator)  

        # Training the model on the training dataset
        # fit function is used to train the model using the training sets as parameters
        clf.fit(X_train, y_train)

        # performing predictions on the test dataset
        y_pred = clf.predict(X_test)
        
        
        self.global_y_pred = y_pred

        # Calculate the confusion matrix

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        # Print the confusion matrix using Matplotlib

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=16)
        plt.ylabel('Actuals', fontsize=16)
        plt.title('Confusion Matrix', fontsize=16)
        self.matrix = plt
        self.scores()

    def scores(self):
        y_test = self.global_y_test
        y_pred = self.global_y_pred
        
        self.precision = precision_score(y_test, y_pred)
        self.recall = recall_score(y_test, y_pred)
        self.f_score = f1_score(y_test, y_pred)
        self.accuracy = accuracy_score(y_test, y_pred)