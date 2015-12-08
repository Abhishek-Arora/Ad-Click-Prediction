from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, NaiveBayes, NaiveBayesModel,SVMWithSGD, SVMModel
from pyspark.sql import SQLContext
import numpy
import math
from operator import add
import sys, os

def generateOHD(featureSet):   
    return featureSet.flatMap(lambda x : x).distinct().zipWithIndex().collectAsMap()

def featureParser(adRow):

	##Remove the label column first.
	splittedColumns = adRow.split(',')
	listOfFeatureTuples = splittedColumns[1:]
	
	##Generate a feature id for the each feature of the ad.
	##Convert the list of features into a list of tuples,
	##where each tuple is of the type (feature_id, feature_value).	
	
	return [(i,x) for i,x in enumerate(listOfFeatureTuples)]
	
def createSparseVectorOfFeatures(features, oneHotDict, numberOfFeatures):
    
	##Filter the features. Basically keep only the ones for which a key exists
	##in the oneHotDict.
    filteredFeatures = filter(lambda item : oneHotDict.has_key(item) , features)
           
    sparseVector = SparseVector(numberOfFeatures, sorted(map(lambda x : oneHotDict[x], sorted(filteredFeatures))), numpy.ones(len(filteredFeatures)))
    # print '*****************************'
	# print sparseVector
    return sparseVector
	
def createLabeledSparseVector(ad, oneHotDict, numberOfFeatures):
	# Create LabeledPoint in the form of (label, sparse vector of features)
	
	features = featureParser(ad)
	sparseVector = createSparseVectorOfFeatures(features, oneHotDict, numberOfFeatures)

	return LabeledPoint(ad[0], sparseVector)
	
	
def calculateTestAccuracy(predictionOnTest, totalTestRecords):
	correctlyClassified = predictionOnTest.filter(lambda x: x[0] == x[1]).count()
	testAccuracy = float(correctlyClassified)/totalTestRecords
	
	return testAccuracy
	

	
def modelWithSVM(trainingData, validationData):
	##Train the model using Support Vector Machines with different values of iterations.
	##Return the SVM model with best accuracy rate
	
	#eta = [0.1, 0.3, 0.5, 1.0, 5.0]
	regularizationParamater = [.0000001, 1., 5000., 10000., 200000.]
	bestSVMModel = None
	bestAccuracy = 0
	numOfIterations = 100
	visualizationData = []
	
	
		
	for regularizer in regularizationParamater:

		model = SVMWithSGD.train(trainingData, numOfIterations, 1.0, regParam=regularizer)
		predict = validationData.map(lambda ad: (ad.label, model.predict(ad.features)))
		totalValidationAds = validationData.count()
		correctlyPredicted = predict.filter(lambda x: x[0] == x[1]).count()
		accuracy = float(correctlyPredicted)/totalValidationAds
		
		visualizationData += [(regularizer, accuracy)]
		
		if accuracy > bestAccuracy:
			bestAccuracy = accuracy
			bestSVMModel = model
				
			
				
	return bestSVMModel, visualizationData
			
def modelWithNaiveBayes(trainingData, validationData):
	##Train the model using Naive Bayes with different values for the regularization parameter lambda.
	##Return the Naive Bayes model with best accuracy rate

	regularizationParamater = [.000000001, .0005, 1., 100000., 2000000.]
	bestNaiveBayesModel = None
	bestAccuracy = 0
	visualizationData = []
	
	for regularizer in regularizationParamater:
		model = NaiveBayes.train(trainingData, regularizer)
		predict = validationData.map(lambda ad: (ad.label, model.predict(ad.features)))
		totalValidationAds = validationData.count()
		correctlyPredicted = predict.filter(lambda x: x[0] == x[1]).count()
		accuracy = float(correctlyPredicted)/totalValidationAds
		
		##Record the accuracy of this model for different values of lambda (the regularization parameter)
		visualizationData += [(regularizer, accuracy)]
		
		if accuracy > bestAccuracy:
			bestAccuracy = accuracy
			bestNaiveBayesModel = model
			
	return bestNaiveBayesModel, visualizationData
	
	
def modelWithLogisticRegression(trainingData, validationData):

	##Train the model using Logistic Regression that employs Stochastic Gradient Descent
	##with different sets of parameters (i.e the value of lambda and the learning step size.
	##Return the LR model with best accuracy rate
	
	#eta = [0.1, 0.3, 0.5, 1.0, 5.0]
	regularizationParamater = [.00000001, .0000005, 1., 1000., 100000.]
	bestLRModel = None
	bestAccuracy = 0
	numOfIterations = 200
	visualizationData = []
	
	
	for regularizer in regularizationParamater:

		model = LogisticRegressionWithSGD.train(trainingData, numOfIterations, 1.0, regParam=regularizer)
		predict = validationData.map(lambda ad: (ad.label, model.predict(ad.features)))
		totalValidationAds = validationData.count()
		correctlyPredicted = predict.filter(lambda x: x[0] == x[1]).count()
		accuracy = float(correctlyPredicted)/totalValidationAds
		
		visualizationData += [(regularizer, accuracy)]
		
		if accuracy > bestAccuracy:
			bestAccuracy = accuracy
			bestLRModel = model
				
	return bestLRModel, visualizationData
	
	
	
	
			
			
	
	# print '***************** total validation ads*************' + str(totalValidationAds)
	# print '***************** total correctly predicted ****************' + str(correctlyPredicted)
	# print '***************** accuracy ****************' + str(accuracy)
		
	
	
	


def main():
	
	input = sys.argv[1] #Takes in the training data file as the input. The file contains the 
						#data where each row represents an ad. The columns represent various
						#features of an ad and the first column represents whether the ad was
						#clicked(represented as 1) or not(represented as 0).
						
	outputPath = sys.argv[2]
	NBPath = sys.argv[3]
	SVMPath = sys.argv[4]
	LRPath = sys.argv[5]
						
	conf = SparkConf().setAppName('Click Prediction')
	conf.set("spark.storage.memoryFraction", "0.40")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)
	
	##Read the text file and preprocess the data after creating the RDD:
	##Remove the new line character from the end and replace the tabs with ',' 
	##as the columns are tab separated in the file.

	
	adsRDD  =  sc.textFile(input).map(lambda x : unicode(x.replace('\n', '').replace('\t', ','))).cache()
	##totalads = adsRDD.count()
	
	##Split the ad data into training set, validation set, and test set.
	##As the data instances are big enough we don't need to perform cross
	##validation. So we can simply split the data into 70-15-15 proportions.
	
	trainingSet, validationSet, testSet = adsRDD.randomSplit([.7 ,.15, .15], 25)
	
	# ##Let's cache the above 3 RDDs as we will be using them during the models traning.
	# ##I already lost marks in the assignment for this reason. Don't wanna repeat it.
	
	# trainingSet.cache()
	# validationSet.cache()
	# testSet.cache()

	
	
	##Parse the feature of each ad and turn them into the form [featureId, (features)]	
	trainingSetFeatures =  trainingSet.map(featureParser)
	
	
	##Create one hot encoding dictionary for the features of the ad.
	oneHotDictForTheAd = generateOHD(trainingSetFeatures)
	##print '****************** OHD*************' + oneHotDictForTheAd.collect()
	numberOfFeatures = len(oneHotDictForTheAd.keys())
	
	
	##Generate labelled Sparse vector for the training and validation set
	##This will be used as an input to our Machine Learning Classification Algorithms
	
	readyTrainingData = trainingSet.map(lambda ad: createLabeledSparseVector(ad, oneHotDictForTheAd, numberOfFeatures))
	readyValidationData = validationSet.map(lambda ad: createLabeledSparseVector(ad, oneHotDictForTheAd, numberOfFeatures))
	readyTestData = testSet.map(lambda ad: createLabeledSparseVector(ad, oneHotDictForTheAd, numberOfFeatures))
	
	
	##Train the model using Logistic Regression, Naive Bayes, and Support Vector Machines
	
	LRModel, visualizationDataForLR  = modelWithLogisticRegression(readyTrainingData, readyValidationData)
	LRpredictionOnTestData = readyTestData.map(lambda ad: (ad.label, LRModel.predict(ad.features)))
	LRAccuracy = calculateTestAccuracy(LRpredictionOnTestData, LRpredictionOnTestData.count())
	
	SVMmodel, visualizationDataForSVM  = modelWithSVM(readyTrainingData, readyValidationData)
	SVMpredictionOnTestData = readyTestData.map(lambda ad: (ad.label, SVMmodel.predict(ad.features)))
	SVMAccuracy = calculateTestAccuracy(SVMpredictionOnTestData, SVMpredictionOnTestData.count())
	
	NBModel, visualizationDataForNaiveBayes = modelWithNaiveBayes(readyTrainingData, readyValidationData)
	NBpredictionOnTestData = readyTestData.map(lambda ad: (ad.label, NBModel.predict(ad.features)))
	NBAccuracy = calculateTestAccuracy(NBpredictionOnTestData, NBpredictionOnTestData.count())
	
	
	
	finalAccuracies = [('Logistic Regression With SGD', LRAccuracy), ('SVM With SGD', SVMAccuracy), ('Naive Bayes', NBAccuracy)]
	finalAccuraciesDF = sqlContext.createDataFrame(finalAccuracies, ['Algorithm', 'Accuracy'])
	
	
	DFvisualizationDataForLR = sqlContext.createDataFrame(visualizationDataForLR, ['Regularization Value', 'Validation Accuracy'])
	DFvisualizationDataForSVM = sqlContext.createDataFrame(visualizationDataForSVM, ['Regularization Value', 'Validation Accuracy'])
	DFvisualizationDataForNaiveBayes = sqlContext.createDataFrame(visualizationDataForNaiveBayes, ['Regularization Value', 'Validation Accuracy'])
	
	print '*************************************************************************************************************************************'
	

	print finalAccuracies
	
	results = sc.parallelize(finalAccuracies).coalesce(1)
	NB = sc.parallelize(visualizationDataForNaiveBayes).coalesce(1)
	SVM = sc.parallelize(visualizationDataForSVM).coalesce(1)
	LR = sc.parallelize(visualizationDataForLR).coalesce(1)
	#output = finalAccuraciesDF.rdd.coalesce(1)
	results.saveAsTextFile(outputPath)
	NB.saveAsTextFile(NBPath)
	SVM.saveAsTextFile(SVMPath)
	LR.saveAsTextFile(LRPath)
	


			
			
			
	
	
	
if __name__ == "__main__":
	main()

