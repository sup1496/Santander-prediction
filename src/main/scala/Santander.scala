import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{Encoder, Encoders}

 object santander {
   def main(args: Array[String]) = {
   	if (args.length != 2) {
       println("Insufficient number of args")
     }
   	val conf = new SparkConf().setAppName("santander")
     val sc = new SparkContext(conf)
     val spark: SparkSession = SparkSession.builder.master("local").appName("santander").getOrCreate()
     import spark.implicits._

//Inputing Data
val input_data = spark.read.format("csv")
  .option("inferSchema","true")
  .option("hea	er","true")
  .load(args(0))

//Removing null values
val filtered_data = input_data.na.drop()

//Creating training and testing datasets
val Array(training_data, testing_data)=filtered_data.randomSplit(Array(0.6, 0.4), seed = 12345)

//Feature Vector
val feature_vector = new VectorAssembler().setOutputCol("attributes").setInputCols(filtered_data.columns.slice(2, filtered_data.columns.size))

//Scaling the attributes
val scaled_data = new MinMaxScaler().setOutputCol("ScaledAttributes").setInputCol("attributes")

//Reducing dimensions of data
val reduced_data = new PCA()
  .setInputCol("ScaledAttributes")
  .setOutputCol("reduced_attributes")
  .setK(180)

//Decision tree model creation
val decisiontree = new DecisionTreeClassifier().setFeaturesCol("reduced_attributes").setLabelCol("target")

//Developing pipeline
val pipeline_dt = new Pipeline().setStages(Array(feature_vector, scaled_data, reduced_data, decisiontree))

//Fitting model
val model_dt = pipeline_dt.fit(training_data)

//Tree
val tree = model_dt.stages.last.asInstanceOf[DecisionTreeClassificationModel]

//Model Validation
val result_dt = model_dt.transform(testing_data)

//Displaying model result
val result = result_dt.select("target","prediction")
val result_df = result.toDF("label","prediction")

//Coverting to RDD
val result_rdd=result_df.select("label","prediction").as[(Double, Double)].rdd

//Evaluation metrics
val evalmetrics_dt = new MulticlassMetrics(result_rdd)
var fileoutput = "Evaluation metrics for Decision Tree\n"

// Finding model accuracy o  f decision tree
val evalaccuracy_dt = evalmetrics_dt.accuracy.toString
fileoutput = fileoutput.concat("\nAccuracy = ")
fileoutput = fileoutput.concat(evalaccuracy_dt)

 //Finding precision of decision tree
 val labels_metrics = evalmetrics_dt.labels
 labels_metrics.foreach { l => fileoutput = fileoutput.concat("\nPrecision(" + l + ") = " )
 fileoutput = fileoutput.concat(evalmetrics_dt.precision(l).toString)}

 // //Finding recall of decision tree
 labels_metrics.foreach { l => fileoutput = fileoutput.concat("\nRecall(" + l  + ") = " )
 fileoutput =fileoutput.concat(evalmetrics_dt.recall(l).toString)}

 // //Finding FPR of decision tree
 labels_metrics.foreach { l => fileoutput = fileoutput.concat("\nFalse Positive Rate(" + l + ") = " )
 fileoutput = fileoutput.concat(evalmetrics_dt.falsePositiveRate(l).toString)}

 // //Finding F1 Score of decision tree
 labels_metrics.foreach { l => fileoutput =fileoutput.concat("\nF1 Score(" + l + ") = " )
 fileoutput =fileoutput.concat(evalmetrics_dt.fMeasure(l).toString)}

///Test Data
val data_test = spark.read.format("csv").option("inferSchema","true").option("header","true").load(args(1))

//Getting the result for test data
val result_testdata = model_dt.transform(data_test)

//Displaying the customer transcation prediction
val final_result=result_testdata.select("ID_code","prediction").take(10)
fileoutput = fileoutput.concat("\nCustomer Transaction Prediction using Decision Tree\n")

final_result.foreach{row =>
row.toSeq.foreach{col => fileoutput=fileoutput.concat(col.toString)
fileoutput = fileoutput.concat("\t")}
fileoutput = fileoutput.concat("\n")}


//Gradient Boosted Tree Classifier
val gbt = new GBTClassifier().setFeaturesCol("reduced_attributes").setLabelCol("target").setFeatureSubsetStrategy("auto").setMaxIter(10)

val pipeline_gbt = new Pipeline().setStages(Array(feature_vector, scaled_data, reduced_data, gbt))

// Parameter Grid creation
val parameter_grid = new ParamGridBuilder().build()

//Binary classification evaluator
val bin_eval = new BinaryClassificationEvaluator()
bin_eval.setLabelCol("target")

//Cross Validation
val cross_val = new CrossValidator().setEstimator(pipeline_gbt).setEstimatorParamMaps(parameter_grid).setEvaluator(bin_eval).setNumFolds(5)

//Fitting the model
val model_cv =cross_val.fit(training_data)

//Model Validation
val predicted_gbt = model_cv.transform(testing_data)

//Displaying the model result
val result_gbt=predicted_gbt.select("target","prediction")
val result_gbt_df=result_gbt.toDF("label","prediction")
//display(result_gbt_df)

//Converting to RDD
val rdd_gbt=result_gbt_df.select("label","prediction").as[(Double, Double)].rdd

//Evaluation Metrics
val evalmetrics_gbt = new MulticlassMetrics(rdd_gbt)
//println(evalmetrics_gbt.confusionMatrix)

fileoutput.concat("\nEvaluation Metrics for GBTCLassifier\n")
//Evaluation Metrics - Accuracy of GBT Classifier
val evalmetrics_acc = evalmetrics_gbt.accuracy.toString
fileoutput = fileoutput.concat("\nAccuracy = ")
fileoutput = fileoutput.concat(evalmetrics_acc)

val labelmetric_gbt = evalmetrics_gbt.labels
labelmetric_gbt.foreach { l => fileoutput = fileoutput.concat("\nPrecision" + l + ") = " )
fileoutput.concat(evalmetrics_gbt.precision(l).toString)}

labelmetric_gbt.foreach { l => fileoutput = fileoutput.concat("\nRecall" + l + ") = ")
fileoutput.concat(evalmetrics_gbt.recall(l).toString)}

labelmetric_gbt.foreach { l => fileoutput = fileoutput.concat("False Positive Rate" + l + ") = ")
fileoutput.concat(evalmetrics_gbt.falsePositiveRate(l).toString)}

labelmetric_gbt.foreach { l => fileoutput = fileoutput.concat("F1 Score" + l + ") = ")
fileoutput.concat(evalmetrics_gbt.fMeasure(l).toString)}

//Getting the result for test data
val result_testdata_gbt = model_cv.transform(data_test)

//Displaying the customer transcation prediction
val final_result_gbt = result_testdata_gbt.select("ID_code","prediction")

fileoutput = fileoutput.concat("\nCustomer Transaction Prediction using Gradient Boosted Tree\n")
final_result.foreach { row =>
  row.toSeq.foreach{col => fileoutput=fileoutput.concat(col.toString)
    fileoutput = fileoutput.concat("\t")
  }
  fileoutput= fileoutput.concat("\n")
}
//display(final_result_gbt)
val rdd = sc.parallelize(fileoutput.split("\n"))
rdd.saveAsTextFile(args(2))
sc.stop()
}}