import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load digits dataset(8x8 grayscale images)
digits=datasets.load_digits()

# Features and Labels
X=digits.images
y=digits.target

# Flatten images(8x8 -> 64 features)
n_samples=len(X)
X=X.reshape((n_samples,-1))

# Train-Test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,shuffle=False)

# Create and train KNN model(k=5)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

# Predictions
y_pred=knn.predict(X_test)

# Accuracy
print("KNN Accuracy:",metrics.accuracy_score(y_test,y_pred))

#Show a few predictions
images_and_predictions=list(zip(digits.images[n_samples//2:],y_pred))
for index,(image,prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(1,4,index+1)
    plt.axis("off")
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title(f'Pred:{prediction}')
plt.show()