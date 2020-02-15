from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,  accuracy_score
import numpy as np
import matplotlib.pyplot as plt


class kNNClassifier:

  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors
    
  def fit(self, X, y):
    self.X = X
    self.y = y
    
    
  def predict(self, X):
    distance = euclidean_distances(X, self.X)
    sort_distance = np.sort(distance)
    argsort_distance = np.argsort(distance)
    column_index = argsort_distance[:, :self.n_neighbors]
    label = np.array([y,] * len(self.X))
    row_index = np.array([np.arange(len(X)), ] * self.n_neighbors).T
    label = label[row_index, column_index]
    return stats.mode(label, axis = 1)[0].reshape(-1,)
    
  
  
  
 ## A few method to visulaise the output 
  
def true_boundary_voting_pred(wealth, religiousness):
  return religiousness-0.1*((wealth-5)**3-wealth**2+(wealth-6)**2+80)

def generate_data(m, seed=None):
  # if seed is not None, this function will always generate the same data
  np.random.seed(seed) 
  
  X = np.random.uniform(low=0.0, high=10.0, size=(m,2))
  y = np.sign(true_boundary_voting_pred(X[:,0], X[:,1]))
  y[y==0] = 1
  samples_to_flip = np.random.randint(0,m//10)
  flip_ind = np.random.choice(m, samples_to_flip, replace=False)
  y[flip_ind] = -y[flip_ind]
  return X, y

def plot_labeled_data(X, y, no_titles=False):
  republicans = (y==1)
  democrats = (y==-1)
  plt.scatter(X[republicans,0], X[republicans,1], c='r')
  plt.scatter(X[democrats,0], X[democrats,1], c='b')
  if not no_titles:
    plt.xlabel('Wealth')
    plt.ylabel('Religiousness')
    plt.title('Red circles represent Republicans, Blues Democrats')
    
  plt.xlim([0, 10]);
  plt.ylim([0, 10]);
  plt.plot(np.linspace(0,10,1000), -true_boundary_voting_pred(np.linspace(0,10,1000), np.zeros(1000)), linewidth=2, c='k');
    
X, y = generate_data(m=10000, seed=10)
k_list = [1, 3, 5, 11, 21, 51, 99]

acc_list = []

X_train, X_test, X_val  = np.split(X, [int(.7*len(X)), int(.8*len(X))])
y_train, y_test, y_val  = np.split(y, [int(.7*len(y)), int(.8*len(y))])


for k in k_list:
    knnclass = kNNClassifier(k)
    knnclass.fit(X_train,y_train)
    y_pred = knnclass.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("k={}, Accuracy:{}".format(k, accuracy))
    acc_list.append(accuracy)
    plot_labeled_data(X_val, y_pred)
    plt.show()
    
## Plot the elbow for K

plt.plot(k_list, acc_list) 
plt.xlabel('K')
plt.ylabel('accuracy')
plt.title('Elbow Graph')
plt.show()

