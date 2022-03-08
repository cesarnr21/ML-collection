
import numpy
import urllib.request
import sklearn

from sklearn.model_selection import train_test_split

url = "https://www.ece.villanova.edu/~xjiao/course/ECE5400/dataset/beer_50000.json"

def parse_data(fname):
    print("Acquiring Data...")
    for l in urllib.request.urlopen(fname):
        yield eval(l)
    print("Data Acquired")

print("Parsing Data...")
data = list(parse_data(url))
print("Data Parsed")

# still need to add the other features here
data2 = [d for d in data if 'user/ageInSeconds'  in d and 'review/palate' in d \
    and 'beer/ABV' in d and 'review/taste' in d and 'review/timeUnix' in d and 'review/aroma' in d]

def feature(datum):
    feat = [1]
    feat.append(datum['user/ageInSeconds'])
    feat.append(datum['review/palate'])
    feat.append(datum['beer/ABV'])
    feat.append(datum['review/taste'])
    feat.append(datum['review/timeUnix'])
    feat.append(datum['review/aroma'])
    return feat

print("Adding Features...")
x = [feature(d) for d in data2]
print("Features Added")
y = [d['review/overall'] for d in data2]

print("Splitting Data...")
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, \
    test_size = 0.2, train_size = 0.8, random_state = None, shuffle = True, stratify = None)
theta, residuals, rank, s = numpy.linalg.lstsq(X_train, y_train)
print("Data Split")

print("Making Predictions...")
prediction = [sum(i * theta) for i in X_test]
print("Predictions Made")

# Mean Average Percentage Error
def MAPE(actual, prediction):
    result = 0
    for i in range(len(actual)):
        val = abs((actual[i] -  prediction[i]) / actual[i])
        result = result + val

    result = result / len(actual)
    return result * 100

print('Mean Average Percentage Error = %2.2f' %(MAPE(y_test, prediction)) + '%')

