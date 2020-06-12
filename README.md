Santander Customer Transaction Prediction

This project is about solving a particular problem posted by Kaggle which involves helping Santander Bank in identifying 
and predicting customers who could make future transactions with the bank. For this, Decision tree classifier and 
Gradient boosted tree have been implemented and the results are compared to the better-performing algorithm.

Steps to run:

1)Import the project in IntellijIDEA and generate the .jar file.
2)Create and start a cluster in EMR (AWS).
3)Create a bucket in S3. 
4)Upload the input files (train.csv and test.csv) and the .jar file in the S3 bucket.
5)Add step in EMR:
	->Select the spark application option and specify a name.
	->Specify the class name in --class
	->Input the path of .jar file, input and output files.
6)Run the step.
