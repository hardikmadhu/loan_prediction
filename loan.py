import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg


csvFile = open('train.csv')
trainData = csv.reader(csvFile, delimiter = ',')

trainDict = {}

counter = 0

for row in trainData:

	skip = False
	if counter == 0:
		counter = counter + 1
		continue
	'''
	for r in row:
		if len(r) == 0:
			skip = True
			break

	if skip == True:
		continue
	'''
	#print row

	tmpDict = {}
	tmpDict[row[0]] = []
	
	if len(row[1]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   if row[1] == 'Male':
		tmpDict[row[0]].append(0)
	   else:
		tmpDict[row[0]].append(1)

	if len(row[2]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   if row[2] == 'Yes':
		tmpDict[row[0]].append(0)
	   else:
		tmpDict[row[0]].append(1)

	if len(row[3]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   if row[3] == '3+':
		tmpDict[row[0]].append(3)
	   else:
		tmpDict[row[0]].append(int(row[3]))
	
	if len(row[4]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   if row[4] == 'Graduate':
		tmpDict[row[0]].append(0)
	   else:
		tmpDict[row[0]].append(1)
	
	if len(row[5]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   if row[5] == 'No':
		tmpDict[row[0]].append(0)
	   else:
		tmpDict[row[0]].append(1)

	if len(row[6]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   tmpDict[row[0]].append(float(row[6]))

	if len(row[7]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   tmpDict[row[0]].append(float(row[7]))

	if len(row[8]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   tmpDict[row[0]].append(float(row[8]))

	if len(row[9]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   tmpDict[row[0]].append(float(row[9]))

	if len(row[10]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
	   tmpDict[row[0]].append(float(row[10]))

	if len(row[11]) == 0:
	   tmpDict[row[0]].append(-1)
	else:
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

print len(trainDict.keys())

indexList = [2,5,6,7,8,9]

for idx in indexList:
	tmpList = []
	for lid in trainDict.keys():
	   if trainDict[lid][idx] != -1:
		tmpList.append(trainDict[lid][idx])
	   else:
		tmpList.append(-1)

	maximum = max(tmpList)
	minimum = min(tmpList)

	data_range = (maximum - minimum)*1.0

	for i in range(len(tmpList)):
	    if tmpList[i] != -1:
		tmpList[i]= (tmpList[i] - minimum)/data_range

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
for i in range(100):
	cList.append(cList[-1]+10)

gammaList = list(np.arange(0.01,0.1,0.01))

cList = [760]
gammaList = [0.04]

trainErrorList = []
testErrorList = []
mList = []

for c in cList:
  for g in gammaList:
    for i in range(10,400):
	clf = SVC(C=c,gamma = g,degree =3)

	clf.fit(trainX[0:i],trainY[0:i])

	predictY = clf.predict(trainX[400:])

	conf_mat = confusion_matrix(trainY[400:], predictY, labels=[0,1])

	sens_0 = (conf_mat[0][0] * 1.0)/sum(conf_mat[0])
	sens_1 = (conf_mat[1][1] * 1.0)/sum(conf_mat[1])

	avgSens = (2 - (sens_0 + sens_1))/2.0

	print c,'\t',g,'\t',avgSens, i

	testErrorList.append(avgSens)
	mList.append(i)

	predictY = clf.predict(trainX[0:i])

	conf_mat = confusion_matrix(trainY[0:i], predictY, labels=[0,1])

	sens_0 = (conf_mat[0][0] * 1.0)/sum(conf_mat[0])
	sens_1 = (conf_mat[1][1] * 1.0)/sum(conf_mat[1])

	avgSens = (2 - (sens_0 + sens_1))/2.0
	trainErrorList.append(avgSens)

pg.plot(trainErrorList, color = 'r')
pg.plot(testErrorList, color = 'y')
s = raw_input()
