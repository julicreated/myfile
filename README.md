'''Fibonacci Using non -recursive way

COUNT=0;
x=int(input("Enter the no of terms :"))
first=0
sec=1 
c=0
if(x<0):
 print("Enter the valid term")
elif(x==0):
 print(0)
elif(x==1):
 print("Fibonacci series upto",x,"is",first)
else:
  while c<x:
      print(first)
      COUNT=COUNT+ 1 
      nth=first+sec
      COUNT=COUNT+ 1 
      first=sec
      COUNT=COUNT+ 1 
      sec=nth
      COUNT=COUNT+ 1 
      c+=1
      COUNT=COUNT+ 1 
       
print("steps required using counter",COUNT)

'''


'''
Fibonacci using recursive way

COUNT=0
def recur_fibo(n): 
    global COUNT
    COUNT=COUNT+1
    if n<=1 :
     return  n 
    else : 
     return recur_fibo(n-1) + recur_fibo(n-2)
     
nterms=int(input("How many terms :"))
if(nterms<=0):
 print("Enter the positive number")
else: 
 
  print("The Fibonacci Sequence Is")
  for i in range(nterms) :
    print(recur_fibo(i))
     
print("The no of steps required",COUNT)
'''
     

''' 
 Huffman coding 

 
string = 'BCAADDDCCACACAC'


# Creating tree nodes
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)


# Main function implementing huffman coding
def huffman_code_tree(node, left=True, binString=''):
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, True, binString + '0'))
    d.update(huffman_code_tree(r, False, binString + '1'))
    return d


# Calculating frequency
freq = {}
for c in string:
    if c in freq:
        freq[c] += 1
    else:
        freq[c] = 1

freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

nodes = freq

while len(nodes) > 1:
    (key1, c1) = nodes[-1]
    (key2, c2) = nodes[-2]
    nodes = nodes[:-2]
    node = NodeTree(key1, key2)
    nodes.append((node, c1 + c2))

    nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

huffmanCode = huffman_code_tree(nodes[0][0])

print(' Char | Huffman code ')
print('----------------------')
for (char, frequency) in freq:
    print(' %-4r |%12s' % (char, huffmanCode[char]))  
 
 '''
 
 ''' 
 3.0/1 knapsack problem - dynamic programming 
 
 #include <iostream>
using namespace std;
// A utility function that returns a maximum of two integers
int max(int a, int b)
{
 return (a > b) ? a: b;
}
// Returns the maximum value that can be put in a knapsack of capacity W
int knapSack(int W, int wt[], int val[], int n)
{
 int i, w;
 int K[n + 1][W + 1];
 // Build table K[][] in bottom up manner
 for (i = 0; i <= n; i++)
 {
 for (w = 0; w <= W; w++)
 {
 if (i == 0 || w == 0)
 K[i][w] = 0;
 else if (wt[i - 1] <= w)
 K[i][w]
 = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w]);
 else
 K[i][w] = K[i - 1][w];
 }
 }
 return K[n][W];
}
int main()
{
 cout << "Enter the number of items in a Knapsack:";
 int n, W;
 cin >> n;
 int val[n], wt[n];
 for (int i = 0; i < n; i++)
 {
 cout << "Enter value and weight for item " << i << ":";
 cin >> val[i];
 cin >> wt[i];
 }
 // int val[] = { 60, 100, 120 };
 // int wt[] = { 10, 20, 30 };
 // int W = 50;
 cout << "Enter the capacity of knapsack";
 cin >> W;
 cout <<"Maximum Profit :"<< knapSack(W, wt, val, n);
 return 0;
}
 
 '''
 
 '''
 4. n queen problem 
 '''
 '''
 import java.util.Arrays;
 
class Main
{
    // Function to check if two queens threaten each other or not
    private static boolean isSafe(char[][] mat, int r, int c)
    {
        // return false if two queens share the same column
        for (int i = 0; i < r; i++)
        {
            if (mat[i][c] == '1') {
                return false;
            }
        }
 
        // return false if two queens share the same `` diagonal
        for (int i = r, j = c; i >= 0 && j >= 0; i--, j--)
        {
            if (mat[i][j] == '1') {
                return false;
            }
        }
 
        // return false if two queens share the same `/` diagonal
        for (int i = r, j = c; i >= 0 && j < mat.length; i--, j++)
        {
            if (mat[i][j] == '1') {
                return false;
            }
        }
 
        return true;
    }
 
    private static void printSolution(char[][] mat)
    {
        for (char[] chars: mat) {
            System.out.println(Arrays.toString(chars).replaceAll(",", ""));
        }
        System.out.println();
    }
 
    private static void nQueen(char[][] mat, int r)
    {
        // if `N` queens are placed successfully, print the solution
        if (r == mat.length)
        {
            printSolution(mat);
            return;
        }
 
        // place queen at every square in the current row `r`
        // and recur for each valid movement
        for (int i = 0; i < mat.length; i++)
        {
            // if no two queens threaten each other
            if (isSafe(mat, r, i))
            {
                // place queen on the current square
                mat[r][i] = '1';
 
                // recur for the next row
                nQueen(mat, r + 1);
 
                // backtrack and remove the queen from the current square
                mat[r][i] = '0';
            }
        }
    }
 
    public static void main(String[] args)
    {
        // `N × N` chessboard
        int N = 8;
 
        // `mat[][]` keeps track of the position of queens in
        // the current configuration
        char[][] mat = new char[N][N];
 
        // initialize `mat[][]` by `-`
        for (int i = 0; i < N; i++) {
            Arrays.fill(mat[i], '0');
        }
 
        nQueen(mat, 0);
    }
}
 
'''

''''
5- qwick sort 

deterministic wayy

#include <iostream>
#include <vector>
// Function to partition the array and return the pivot index
int partition(std::vector<int> &arr, int low, int high) {
 // Choose the middle element as the pivot
 int pivot = arr[(low + high) / 2];
 int i = low - 1;
 int j = high + 1;
 while (true) {
 do {
 i++;
 } while (arr[i] < pivot);
 do {
 j--;
 } while (arr[j] > pivot);
 if (i >= j)
 return j;
 std::swap(arr[i], arr[j]);
 }
}
// Function to perform quicksort
void quicksort(std::vector<int> &arr, int low, int high) {
 if (low < high) {
 int pivotIndex = partition(arr, low, high);
 // Recursively sort the elements on the left and right of the pivot
 quicksort(arr, low, pivotIndex);
 quicksort(arr, pivotIndex + 1, high);
 }
}
int main() {
 std::vector<int> arr = {12, 4, 5, 6, 7, 3, 1, 15, 8, 9, 2, 10};
 int n = arr.size();
 std::cout << "Original array: ";
 for (int num : arr) {
 std::cout << num << " ";
 }
 std::cout << std::endl;
 quicksort(arr, 0, n - 1);
 std::cout << "Sorted array: ";
 for (int num : arr) {
 std::cout << num << " ";
 }
 std::cout << std::endl;
 return 0;
}
'''
'''
randomized wayy


#include <iostream>
#include <cstdlib>
#include <ctime>
// Function to swap two elements
void swap(int arr[], int i, int j) {
 int temp = arr[i];
 arr[i] = arr[j];
 arr[j] = temp;
}
// Function to partition the array
int partitionLeft(int arr[], int low, int high) {
 int pivot = arr[high];
 int i = low;
 for (int j = low; j < high; j++) {
 if (arr[j] <= pivot) {
 swap(arr, i, j);
 i++;
 }
 }
 swap(arr, i, high);
 return i;
}
// Function to perform random partition
int partitionRight(int arr[], int low, int high) {
 srand(time(NULL));
 int r = low + rand() % (high - low);
 swap(arr, r, high);
 return partitionLeft(arr, low, high);
}
// Recursive function for quicksort
void quicksort(int arr[], int low, int high) {
 if (low < high) {
 int p = partitionRight(arr, low, high);
 quicksort(arr, low, p - 1);
 quicksort(arr, p + 1, high);
 }
}
// Function to print the array
void printArray(int arr[], int size) {
 for (int i = 0; i < size; i++)
 std::cout << arr[i] << " ";
 std::cout << std::endl;
}
// Driver code
int main() {
 int arr[] = {6, 4, 12, 8, 15, 16};
 int n = sizeof(arr) / sizeof(arr[0]);
 std::cout << "Original array: ";
 printArray(arr, n);
 quicksort(arr, 0, n - 1);
 std::cout << "Sorted array: ";
 printArray(arr, n);
 return 0;
}


'''

''' Machine learning 
1.uber 
...

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
...
uber = pd.read_csv('uber.csv')
....
uber.head()
...
uber.info()
...
uber.isnull().sum()
...
uber_2 = uber.drop(['Unnamed: 0','key'],axis=1)
uber_2.dropna(axis=0,inplace=True)
....
uber_2.isnull().sum()
...
def haversine (lon_1, lon_2, lat_1, lat_2):

    lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1, lat_2])  #Degrees to Radians


    diff_lon = lon_2 - lon_1
    diff_lat = lat_2 - lat_1


    km = 2 * 6371 * np.arcsin(np.sqrt(np.sin(diff_lat/2.0)**2 +
                                      np.cos(lat_1) * np.cos(lat_2) * np.sin(diff_lon/2.0)**2))

    return km
    
 ...
 
 uber_2['Distance']= haversine(uber_2['pickup_longitude'],uber_2['dropoff_longitude'],
                             uber_2['pickup_latitude'],uber_2['dropoff_latitude'])

uber_2['Distance'] = uber_2['Distance'].astype(float).round(2)    # Round-off Optional
...

uber_2.head()
...

plt.scatter(uber_2['Distance'], uber_2['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
....

uber_2.drop(uber_2[uber_2['Distance'] > 60].index, inplace = True)
uber_2.drop(uber_2[uber_2['Distance'] == 0].index, inplace = True)
uber_2.drop(uber_2[uber_2['fare_amount'] == 0].index, inplace = True)
uber_2.drop(uber_2[uber_2['fare_amount'] < 0].index, inplace = True)
...

uber_2.drop(uber_2[(uber_2['fare_amount']>100) & (uber_2['Distance']<1)].index, inplace = True )
uber_2.drop(uber_2[(uber_2['fare_amount']<100) & (uber_2['Distance']>100)].index, inplace = True )
....

uber_2.info()
....
plt.scatter(uber_2['Distance'], uber_2['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
....
uber_2['pickup_datetime'] = pd.to_datetime(uber_2['pickup_datetime'])

uber_2['Year'] = uber_2['pickup_datetime'].apply(lambda time: time.year)
uber_2['Month'] = uber_2['pickup_datetime'].apply(lambda time: time.month)
uber_2['Day'] = uber_2['pickup_datetime'].apply(lambda time: time.day)
uber_2['Day of Week'] = uber_2['pickup_datetime'].apply(lambda time: time.dayofweek)
uber_2['Day of Week_num'] = uber_2['pickup_datetime'].apply(lambda time: time.dayofweek)
uber_2['Hour'] = uber_2['pickup_datetime'].apply(lambda time: time.hour)

day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
uber_2['Day of Week'] = uber_2['Day of Week'].map(day_map)

uber_2['counter'] = 1
.....

uber_2['pickup'] = uber_2['pickup_latitude'].astype(str) + "," + uber_2['pickup_longitude'].astype(str)
uber_2['drop off'] = uber_2['dropoff_latitude'].astype(str) + "," + uber_2['dropoff_longitude'].astype(str)

.....

uber_2.head()
...

no_of_trips = []
year = [2009, 2010, 2011, 2012, 2013, 2014, 2015]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(2009, 2016):
    x = uber_2.loc[uber_2['Year'] == i, 'counter'].sum()
    no_of_trips.append(x)

print("Average trips a year: ")
print(year, no_of_trips)

plt.bar(year, no_of_trips, color=colors)

...
no_of_trips = []
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(1, 13):
    x = uber_2.loc[uber_2['Month'] == i, 'counter'].sum()
    no_of_trips.append(x)

print("Average trips a Month: ")
print(month, no_of_trips)

plt.bar(month, no_of_trips, color=colors)


....
no_of_trips = []
day = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

for i in range(0, 7):
    x = uber_2.loc[uber_2['Day of Week_num'] == i, 'counter'].sum()
    no_of_trips.append(x)

print("Average trips by Days: ")
print(day, no_of_trips)

plt.bar(day, no_of_trips, color=colors)

.....


year_vs_trips = uber_2.groupby(['Year','Month']).agg(
    no_of_trips = ('counter','count'),
    Average_fair = ('fare_amount','mean'),
    Total_fair = ('fare_amount','sum'),
    Avg_distance = ( 'Distance', 'mean')).reset_index()

year_vs_trips['avg_no_of_trips'] = year_vs_trips['no_of_trips']/30
year_vs_trips['month_year'] = year_vs_trips['Month'].astype(str) +", "+ year_vs_trips['Year'].astype(str)


year_vs_trips = year_vs_trips.reset_index()

year_vs_trips.head()


year_vs_trips.plot(kind='line',x='month_year',y='no_of_trips', xlabel='January, 2009 - June, 2015',
    ylabel='No of Trips', title='No of trips vs Months')
    ......
    
    
import seaborn as sns

df_1 = uber_2[['Distance', 'Day of Week_num', 'Hour']].copy()

df_h = df_1.copy()

df_h = df_h.groupby(['Hour', 'Day of Week_num']).mean()
df_h = df_h.unstack(level=0)

....

fig, ax = plt.subplots(figsize=(24, 7))
sns.heatmap(df_h, cmap="Reds",
           linewidth=0.3, cbar_kws={"shrink": .8})

xticks_labels = ['12 AM', '01 AM', '02 AM ', '03 AM ', '04 AM ', '05 AM ', '06 AM ', '07 AM ',
                 '08 AM ', '09 AM ', '10 AM ', '11 AM ', '12 PM ', '01 PM ', '02 PM ', '03 PM ',
                 '04 PM ', '05 PM ', '06 PM ', '07 PM ', '08 PM ', '09 PM ', '10 PM ', '11 PM ']

yticks_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

plt.xticks(np.arange(24) + .5, labels=xticks_labels)
plt.yticks(np.arange(7) + .5, labels=yticks_labels)

ax.xaxis.tick_top()

title = 'Weekly Uber Rides'.upper()
plt.title(title, fontdict={'fontsize': 25})

plt.show()

....


import statistics as st

print("Mean of fare prices is % s "
         % (st.mean(uber_2['fare_amount'])))

print("Median of fare prices is % s "
         % (st.median(uber_2['fare_amount'])))

print("Standard Deviation of Fare Prices is % s "
                % (st.stdev(uber_2['fare_amount'])))
...

import statistics as st

print("Mean of Distance is % s "
         % (st.mean(uber_2['Distance'])))

print("Median of Distance is % s "
         % (st.median(uber_2['Distance'])))

print("Standard Deviation of Distance is % s "
                % (st.stdev(uber_2['Distance'])))

....

corr = uber_2.corr()

corr.style.background_gradient(cmap='BuGn')

....

X = uber_2['Distance'].values.reshape(-1, 1)        #Independent Variable
y = uber_2['fare_amount'].values.reshape(-1, 1) 

....

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
y_std = std.fit_transform(y)
print(y_std)

x_std = std.fit_transform(X)
print(x_std)

...

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.2, random_state=0)

.....

from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train, y_train)

print("Training set score: {:.2f}".format(l_reg.score(X_train, y_train)))
print("Test set score: {:.7f}".format(l_reg.score(X_test, y_test)))

.....

y_pred = l_reg.predict(X_test)
df = {'Actual': y_test, 'Predicted': y_pred}

from tabulate import tabulate
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
....

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

.....

print(l_reg.intercept_)
print(l_reg.coef_)


.....

plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, l_reg.predict(X_train), color ="blue")
plt.title("Fare vs Distance (Training Set)")
plt.ylabel("fare_amount")
plt.xlabel("Distance")

plt.subplot(2, 2, 2)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, l_reg.predict(X_train), color ="blue")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.title("Fare vs Distance (Test Set)")


plt.tight_layout()
plt.rcParams["figure.figsize"] = (32,22)
plt.show()

...

'''

''' 
2. spam

...
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
....

data=pd.read_csv('spam.csv')
data
...

data.columns
...

data.info()
...

data.isna().sum()
...


data['Spam']=data['Category'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)
...

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data.Message,data.Spam,test_size=0.25)

...

#CounterVectorizer Convert the text into matrics
from sklearn.feature_extraction.text import CountVectorizer

....

from sklearn.naive_bayes import MultinomialNB

...

from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])

.....

clf.fit(X_train,y_train)
...

emails=[
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]

.....

clf.predict(emails)

....

clf.score(X_test,y_test)
'''''

''''
3.Churn_Modelling Bank Customer


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #Importing the libraries
....
df = pd.read_csv("Churn_Modelling.csv")
...
df.head()

....
df.shape
...

df.describe()
....

df.isnull()
....

df.isnull().sum()

....
df.info()
...

df.dtypes

...

df.columns
...
import pandas as pd
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1)
...

df.head()
...
def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()

....

df_churn_exited = df[df['Exited']==1]['Tenure']
df_churn_not_exited = df[df['Exited']==0]['Tenure']
...

visualization(df_churn_exited, df_churn_not_exited, "Tenure")

....

df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']
....

visualization(df_churn_exited2, df_churn_not_exited2, "Age")

...
X = df[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
states = pd.get_dummies(df['Geography'],drop_first = True)
gender = pd.get_dummies(df['Gender'],drop_first = True)

.....
df = pd.concat([df,gender,states], axis = 1)

...

df.head()
....

X = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Male','Germany','Spain']]
....
y = df['Exited']

...

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)

....

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

...

X_train  = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

...

X_train
...
X_test
...

import keras #Keras is the wrapper on the top of tenserflow
#Can use Tenserflow as well but won't be able to understand the errors initially.

....

from keras.models import Sequential #To create sequential neural network
from keras.layers import Dense #To create hidden layers

....
classifier = Sequential()
...

classifier.add(Dense(activation = "relu",input_dim = 11,units = 6,kernel_initializer = "uniform"))

...
classifier.add(Dense(activation = "relu",units = 6,kernel_initializer = "uniform"))   #Adding second hidden layers
...
classifier.add(Dense(activation = "sigmoid",units = 1,kernel_initializer = "uniform")) #Final neuron will be having siigmoid function
...
classifier.compile(optimizer="adam",loss = 'binary_crossentropy',metrics = ['accuracy'])
..

classifier.summary() #3 layers created. 6 neurons in 1st,6neurons in 2nd layer and 1 neuron in last
...

classifier.fit(X_train,y_train,batch_size=10,epochs=50) #Fitting the ANN to training dataset
..
y_pred =classifier.predict(X_test)
y_pred = (y_pred > 0.5) #Predicting the result
...
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

...
cm = confusion_matrix(y_test,y_pred)
...

cm
...
accuracy = accuracy_score(y_test,y_pred)
...
accuracy
...

plt.figure(figsize = (10,7))
sns.heatmap(cm,annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
...

print(classification_report(y_test,y_pred))

'''

'''
4.KNN diabetes

from mlxtend.plotting import plot_decision_regions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#plt.style.use('ggplot')
#ggplot is R based visualisation package that provides better graphics with higher level of abstraction

.....

#Loading the dataset
diabetes_data = pd.read_csv('diabetes.csv')

#Print the first 5 rows of the dataframe.
diabetes_data.head()

.....
## gives information about the data types,columns, null value counts, memory usage etc
## function reference : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html
diabetes_data.info(verbose=True)
....
diabetes_data.describe()

....
diabetes_data.describe().T

...

diabetes_data_copy = diabetes_data.copy(deep = True)
diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = diabetes_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

## showing the count of Nans
print(diabetes_data_copy.isnull().sum())

.....

p = diabetes_data.hist(figsize = (20,20))
...
diabetes_data_copy['Glucose'].fillna(diabetes_data_copy['Glucose'].mean(), inplace = True)
diabetes_data_copy['BloodPressure'].fillna(diabetes_data_copy['BloodPressure'].mean(), inplace = True)
diabetes_data_copy['SkinThickness'].fillna(diabetes_data_copy['SkinThickness'].median(), inplace = True)
diabetes_data_copy['Insulin'].fillna(diabetes_data_copy['Insulin'].median(), inplace = True)
diabetes_data_copy['BMI'].fillna(diabetes_data_copy['BMI'].median(), inplace = True)

.....
p = diabetes_data_copy.hist(figsize = (20,20))

....

## observing the shape of the data
diabetes_data.shape
...

## data type analysis
#plt.figure(figsize=(5,5))
#sns.set(font_scale=2)
sns.countplot(y=diabetes_data.dtypes ,data=diabetes_data)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()
...
## null count analysis
import missingno as msno
p=msno.bar(diabetes_data)

...
## checking the balance of the data by plotting the count of outcomes by their value
color_wheel = {1: "#0392cf",
               2: "#7bc043"}
colors = diabetes_data["Outcome"].map(lambda x: color_wheel.get(x + 1))
print(diabetes_data.Outcome.value_counts())
p=diabetes_data.Outcome.value_counts().plot(kind="bar")
.....

from pandas.plotting import scatter_matrix
p=scatter_matrix(diabetes_data,figsize=(25, 25))

...
p=sns.pairplot(diabetes_data_copy, hue = 'Outcome')

....
plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes_data.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap

.....

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X =  pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
       
......

X.head()

....
#X = diabetes_data.drop("Outcome",axis = 1)
y = diabetes_data_copy.Outcome

.....
#importing train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

....

from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)

    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
    
.....

## score that comes from testing on the same datapoints that were used for training
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

..

## score that comes from testing on the datapoints that were split in the beginning to be used for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
.....

plt.figure(figsize=(12,5))
p = sns.lineplot(data=(range(1,15),train_scores),marker='*',label='Train Score')
p = sns.lineplot(data=(range(1,15),test_scores),marker='o',label='Test Score')
....

#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(11)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
...
value = 20000
width = 20000
plot_decision_regions(X.values, y.values, clf=knn, legend=2,
                      filler_feature_values={2: value, 3: value, 4: value, 5: value, 6: value, 7: value},
                      filler_feature_ranges={2: width, 3: width, 4: width, 5: width, 6: width, 7: width},
                      X_highlight=X_test.values)

# Adding axes annotations
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
plt.title('KNN with Diabetes Data')
plt.show()

....

#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
confusion_matrix(y_test,y_pred)
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
....

y_pred = knn.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

......

'''
'''
5. K means -sales_data_sample

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Importing the required libraries.

.....
from sklearn.cluster import KMeans, k_means #For clustering
from sklearn.decomposition import PCA #Linear Dimensionality reduction.
....
df = pd.read_csv("sales_data_sample.csv", sep=",", encoding='Latin-1') #Loading the dataset.

.....

df.head()
...
df.shape
...
df.describe()
...

df.info()
....
df.isnull().sum()
...
df.dtypes
...
df_drop  = ['ADDRESSLINE1', 'ADDRESSLINE2', 'STATUS','POSTALCODE', 'CITY', 'TERRITORY', 'PHONE', 'STATE', 'CONTACTFIRSTNAME', 'CONTACTLASTNAME', 'CUSTOMERNAME', 'ORDERNUMBER']
df = df.drop(df_drop, axis=1) #Dropping the categorical uneccessary columns along with columns having null values. Can't fill the null values are there are alot of null values.

.....
df.isnull().sum()

...

df.dtypes
...
df['COUNTRY'].unique()

....
df['PRODUCTLINE'].unique()
....
df['DEALSIZE'].unique()
.....
productline = pd.get_dummies(df['PRODUCTLINE']) #Converting the categorical columns.
Dealsize = pd.get_dummies(df['DEALSIZE'])
.....

df = pd.concat([df,productline,Dealsize], axis = 1)
.....
df_drop  = ['COUNTRY','PRODUCTLINE','DEALSIZE'] #Dropping Country too as there are alot of countries.
df = df.drop(df_drop, axis=1)
...
df['PRODUCTCODE'] = pd.Categorical(df['PRODUCTCODE']).codes #Converting the datatype.

....
df.drop('ORDERDATE', axis=1, inplace=True) #Dropping the Orderdate as Month is already included.
....
df.dtypes #All the datatypes are converted into numeric

....
distortions = [] # Within Cluster Sum of Squares from the centroid
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)   #Appeding the intertia to the Distortions
.....

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

.....
X_train = df.values #Returns a numpy array.
....
X_train.shape
...
model = KMeans(n_clusters=3,random_state=2) #Number of cluster = 3
model = model.fit(X_train) #Fitting the values to create a model.
predictions = model.predict(X_train) #Predicting the cluster values (0,1,or 2)

....
unique,counts = np.unique(predictions,return_counts=True)
....

counts = counts.reshape(1,3)
....
counts_df = pd.DataFrame(counts,columns=['Cluster1','Cluster2','Cluster3'])
.....
counts_df.head()
....
pca = PCA(n_components=2) #Converting all the features into 2 columns to make it easy to visualize using Principal COmponent Analysis.
....
reduced_X = pd.DataFrame(pca.fit_transform(X_train),columns=['PCA1','PCA2']) #Creating a DataFrame.

.....
reduced_X.head()

.....
#Plotting the normal Scatter Plot
plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])
.....
model.cluster_centers_ #Finding the centriods. (3 Centriods in total. Each Array contains a centroids for particular feature )

.....
reduced_centers = pca.transform(model.cluster_centers_) #Transforming the centroids into 3 in x and y coordinates

....
reduced_centers
....
plt.figure(figsize=(14,10))
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'])
plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300) #Plotting the centriods

.....
reduced_X['Clusters'] = predictions #Adding the Clusters to the reduced dataframe.
....
reduced_X.head()

....
#Plotting the clusters
plt.figure(figsize=(14,10))
#                     taking the cluster number and first column           taking the same cluster number and second column      Assigning the color
plt.scatter(reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 0].loc[:,'PCA2'],color='slateblue')
plt.scatter(reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 1].loc[:,'PCA2'],color='springgreen')
plt.scatter(reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA1'],reduced_X[reduced_X['Clusters'] == 2].loc[:,'PCA2'],color='indigo')


plt.scatter(reduced_centers[:,0],reduced_centers[:,1],color='black',marker='x',s=300)
....
'''

'''' 
Blockchain Technology

3- Bank account of a customer 

//SPDX-License-Identifier: MIT 

pragma solidity ^0.8.0;

//Created a smart contract that allows a user to deposit, withdraw and save ETH!!

contract bank{
    //we mapped the address of the caller balance in the contract
    mapping(address => uint) public balances;

// whatever the user deposit is added to msg.value of the sender address we mapped above
    function deposit() public payable{
        balances[msg.sender] += msg.value; 
    }
    
//we create the fucntion of witdraw 
    function withdraw(uint _amount) public{
        //we create a require arg to make sure the balance of the sender is >= _amount if not ERR
        require(balances[msg.sender]>= _amount, "Not enough ether");
        //if the amount is availabe we subtract it from the sender 
        balances[msg.sender] -= _amount*1000000000000000000;
        //True bool is called to confirm the amount
        (bool sent,) = msg.sender.call{value: _amount*1000000000000000000}("Sent");
        require(sent, "failed to send ETH");

        
    }

    function getBal() public view returns(uint){
        return address(this).balance/1000000000000000000;
    }

}

'''
''' 
4-student data

// SPDX-License-Identifier: MIT   
pragma solidity >= 0.8.7;

// Build the Contract
contract MarksManagmtSys
{
	// Create a structure for
	// student details
	struct Student
	{
		int ID;
		string fName;
		string lName;
		int marks;
	}

	address owner;
	int public stdCount = 0;
	mapping(int => Student) public stdRecords;

	modifier onlyOwner
	{
		require(owner == msg.sender);
		_;
	}
	constructor()
	{
		owner=msg.sender;
	}

	// Create a function to add
	// the new records
	function addNewRecords(int _ID,
						string memory _fName,
						string memory _lName,
						int _marks) public onlyOwner
	{
		// Increase the count by 1
		stdCount = stdCount + 1;

		// Fetch the student details
		// with the help of stdCount
		stdRecords[stdCount] = Student(_ID, _fName,
									_lName, _marks);
	}

	// Create a function to add bonus marks
	function bonusMarks(int _bonus) public onlyOwner
	{
		stdRecords[stdCount].marks =
					stdRecords[stdCount].marks + _bonus;
	}
}
''' 
''' 
miniproject -voting 


// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Ballot {
    // VARIBLES
    struct vote {    
        //struct is a special datatype in solidity which enables us to bundle various variables of different datatypes together
        address voterAddresss;
        bool choice;
    }
    struct voter {
        string voterName;
        bool voted;
    }
    //created counters which will keep track of various events
    uint private countResult = 0;
    uint public finalResult = 0;
    uint public totalVoter = 0;
    uint public totalVote = 0;

    address public ballotOfficialAddress;
    string public ballotOfficalName;
    string public proposal;

    mapping(uint => vote) private votes;
    mapping(address => voter) public voterRegister;

    enum State { Created, Voting, Ended }
    State public state;


    // MODIFIER
    modifier condition(bool _condition) {
        require(_condition);
        _;
    }

    modifier onlyOfficial() {
        require(msg.sender == ballotOfficialAddress);
        _;
    }

    modifier inState(State _state) {
        require(state == _state);
        _;
    }


    // FUNCTION
    constructor(
        string memory _ballotofficalName,
        string memory _proposal
    )  {
        ballotOfficialAddress = msg.sender;
        ballotOfficalName = _ballotofficalName;
        proposal = _proposal;
        state = State.Created;
    }

    
    function addVoter(
        address _voterAdress,
        string memory _voterName
    ) public
        inState(State.Created)
        onlyOfficial    
    {
        voter memory v;
        v.voterName = _voterName;
        v.voted = false;
        voterRegister[_voterAdress] = v;
        totalVoter++;
    }


    function startVote() 
        public 
        inState(State.Created) 
        onlyOfficial 
    {
        state = State.Voting;
    }



    function doVote(bool _choice)
        public
        inState(State.Voting)
        returns (bool voted) 
    {
        bool isFound = false;
        if(bytes(voterRegister[msg.sender].voterName).length != 0 
            && voterRegister[msg.sender].voted == false ) 
        {
            voterRegister[msg.sender].voted = true;
            vote memory v;
            v.voterAddresss = msg.sender;
            v.choice = _choice;
            if(_choice) {
                countResult++;
            }
            votes[totalVote] = v;
            totalVote++;
            isFound = true;
        }
        return isFound;
    }
    function endVote() 
        public
        inState(State.Voting)
        onlyOfficial
    {
        state = State.Ended;
        finalResult = countResult;
    }

}

'''





