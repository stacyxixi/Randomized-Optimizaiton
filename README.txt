## Spring 2018 CS7641 Assignment2: Randomized Optimization



### Description


The code takes csv files of data set "btrain.csv" and "btest.csv" under the directory "/data".
It outputs csv files, for different algorithms and hyperparameter settings, which contains the results(training accuracy, testing accuracy and training time for each iteration) to the directory “/output”.


### Software Requirements
Download ABAGAIL package from: https://github.com/pushkar/ABAGAIL. To use ABAGAIL with Eclipse, follow the steps described in:
https://github.com/pushkar/ABAGAIL/blob/master/faq.md#how-to-use-abagail-with-eclipse

### Code
“/src/TrainNeuralNetworks.java”: neural network optimization with RHC, SA and GA
“/src/ContinouesPeaks.java”: analyzing the continous peak problem with RHC, SA, GA and MIMIC
“/src/Knapsack.java”: analyzing the knapsack problem with RHC, SA, GA and MIMIC
“/src/TravelingSalesman.java”: analyzing the traveling salesman problem with RHC, SA, GA and MIMIC

### Other Files
"README.txt": instructions for running the code
"xwang738-analysis.pdf": report file

"final_results_analysis.xlsx": excel file for making all the figures in the report 

“/data/btrain.csv”: training data for the Breast Cancer Wisconsin (Diagnostic) Data Set
“/data/btest.csv”: testing data for the Breast Cancer Wisconsin (Diagnostic) Data Set
"/output": directory for exporting csv files


