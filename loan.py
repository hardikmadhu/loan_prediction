import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pyqtgraph as pg

M_TRAIN = 500
csvFile = open('train.csv')
trainData = csv.reader(csvFile, delimiter = ',')

trainDict = {}

counter = 0

def getF(conf_mat):
	precision = conf_mat[1][1]*1.0/(conf_mat[1][1] + conf_mat[0][1])
	recall = conf_mat[1][1] * 1.0/sum(conf_mat[1]) 
	f = (recall * precision)/ (recall+precision)
	return f


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
for i in range(200):
	cList.append(cList[-1]+10)

gammaList = list(np.arange(0.01,0.1,0.01))
gammaList.extend(np.arange(0.1,0.5,0.1))

cList = [90]
gammaList = [0.2]

trainErrorList = []
testErrorList = []
mList = []

for c in cList:
  for g in gammaList:
    for i in range(10,M_TRAIN):
	clf = SVC(C=c,gamma = g,degree =3)

	clf.fit(trainX[0:i],trainY[0:i])

	predictY = clf.predict(trainX[M_TRAIN:])

	conf_mat = confusion_matrix(trainY[M_TRAIN:], predictY, labels=[0,1])

	sens_0 = 1 - ((conf_mat[0][0] * 1.0)/sum(conf_mat[0]))
	sens_1 = 1 - ((conf_mat[1][1] * 1.0)/sum(conf_mat[1]))

	avgSens = (sens_0 + sens_1)/2.0
	f = getF(conf_mat)
	#print c,'\t',g,'\t',avgSens

	testErrorList.append(f)
	mList.append(i)

	predictY = clf.predict(trainX[0:i])

	conf_mat = confusion_matrix(trainY[0:i], predictY, labels=[0,1])

	sens_0 = 1 - ((conf_mat[0][0] * 1.0)/sum(conf_mat[0]))
	sens_1 = 1 - ((conf_mat[1][1] * 1.0)/sum(conf_mat[1]))

	avgSens = (sens_0 + sens_1)/2.0
	f = getF(conf_mat)
	trainErrorList.append(f)

win = pg.GraphicsWindow()
pl1 = win.addPlot()
pl1.plot(trainErrorList, pen = 'r')
pl1.plot(testErrorList, pen = 'y')
pl1.show()
s = raw_input()
print M_TRAIN,":",trainErrorList[-1], testErrorList[-1]
