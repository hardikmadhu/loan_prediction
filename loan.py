import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


csvFile = open('train.csv')
trainData = csv.reader(csvFile, delimiter = ',')

trainDict = {}

counter = 0

for row in trainData:

	skip = False
	if counter == 0:
		counter = counter + 1
		continue

	for r in row:
		if len(r) == 0:
			skip = True
			break

	if skip == True:
		continue

	#print row

	tmpDict = {}
	tmpDict[row[0]] = []
	
	if row[1] == 'Male':
		tmpDict[row[0]].append(0)
	else:
		tmpDict[row[0]].append(1)

	if row[2] == 'Yes':
		tmpDict[row[0]].append(0)
	else:
		tmpDict[row[0]].append(1)

	if row[3] == '3+':
		tmpDict[row[0]].append(3)
	else:
		tmpDict[row[0]].append(int(row[3]))
	
	if row[4] == 'Graduate':
		tmpDict[row[0]].append(0)
	else:
		tmpDict[row[0]].append(1)
	
	if row[5] == 'No':
		tmpDict[row[0]].append(0)
	else:
		tmpDict[row[0]].append(1)

	tmpDict[row[0]].append(float(row[6]))
	tmpDict[row[0]].append(float(row[7]))
	tmpDict[row[0]].append(float(row[8]))
	tmpDict[row[0]].append(float(row[9]))
	tmpDict[row[0]].append(float(row[10]))

	if row[11] == 'Urban':
		tmpDict[row[0]].append(0)
	elif row[11] == 'Semiurban':
		tmpDict[row[0]].append(0.5)
	else:
		tmpDict[row[0]].append(1)

	if row[12] == 'Y':
		tmpDict[row[0]].append(1)
	else:
		tmpDict[row[0]].append(0)
		

	trainDict.update(tmpDict)

#print trainDict.keys(), len(trainDict.keys())

indexList = [2,5,6,7,8,9]

for idx in indexList:
	tmpList = []
	for lid in trainDict.keys():
		tmpList.append(trainDict[lid][idx])

	maximum = max(tmpList)
	minimum = min(tmpList)

	data_range = (maximum - minimum)*1.0

	tmpList = [(x - minimum)/data_range for x in tmpList]

	for i in range(len(trainDict.keys())):
		lid = trainDict.keys()[i]
		trainDict[lid][idx] = tmpList[i]


trainX = []
trainY = []

for lid in trainDict.keys():
	trainX.append(trainDict[lid][0:11])
	trainY.append(trainDict[lid][11])


trainX = np.array(trainX)
trainY = np.array(trainY)

cList = [10]
for i in range(1500):
	cList.append(cList[-1]+10)

gammaList = list(np.arange(0.01,0.1,0.01))


for c in cList:
    for g in gammaList:
	clf = SVC(C=c,gamma = g)

	clf.fit(trainX[0:400],trainY[0:400])

	predictY = clf.predict(trainX[400:])

	conf_mat = confusion_matrix(trainY[400:], predictY, labels=[0,1])

	sens_0 = (conf_mat[0][0] * 1.0)/sum(conf_mat[0])
	sens_1 = (conf_mat[1][1] * 1.0)/sum(conf_mat[1])

	avgSens = (sens_0 + sens_1)/2.0

	print c,'\t',g,'\t',avgSens
