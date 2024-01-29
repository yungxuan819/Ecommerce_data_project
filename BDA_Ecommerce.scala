// Databricks notebook source
import org.apache.spark.sql.Encoders



case class eCommerce(Email: String,

                    Avatar: String,

                    Avg_Session_Length: Double,

                    Time_on_App: Double,

                    Time_on_Website: Double,

                    Length_of_Membership: Double,

                    Yearly_Amount_Spent: Double)

// we create a class definition above to store the e-commerce object in the memory
// we create the e-commerce schema object that will be used to locate the dataset
// in the DBFS and load the files into the e-commerce data frame
// then we will display the contents of the data frame

val eCommerceSchema = Encoders.product[eCommerce].schema



val eCommerceDF = spark.read.schema(eCommerceSchema).option("header", "true").csv("/FileStore/tables/ecommerce.csv")



display(eCommerceDF)

// import dataset into dataframe and display the contents of the dataframe

// COMMAND ----------

eCommerceDF.printSchema()

// display the columns details of the ecommerce dataframe such as name and data types

// COMMAND ----------

eCommerceDF.show()

// COMMAND ----------

eCommerceDF.select("Avg_Session_Length","Time_on_App", "Time_on_Website", "Length_of_Membership", "Yearly_Amount_Spent").describe().show()

// use the describe() function to display the count, mean, std dev, min and max value
// for all the numeric double data type column in the ecommerce dataframe

// COMMAND ----------

eCommerceDF.createOrReplaceTempView("EcommerceData")

// create an ecommerce data table for us to run sql statements on this table later

// COMMAND ----------

// MAGIC %sql
// MAGIC select * from EcommerceData
// MAGIC
// MAGIC -- run the sql statements on the new ecommerce data table, this is a spark sql statements

// COMMAND ----------

// MAGIC %sql
// MAGIC select Yearly_Amount_Spent from EcommerceData
// MAGIC
// MAGIC -- what are the customers' spending trends? descriptive analytics, we perform exploratory data analytics
// MAGIC -- EDA here

// COMMAND ----------

// MAGIC %sql
// MAGIC select Avatar as Fashion, count(Avatar) from EcommerceData group by Avatar
// MAGIC -- what is the most popular avatar selected by customers?
// MAGIC -- how this information can be useful for business decision making?

// COMMAND ----------

// MAGIC %sql
// MAGIC select Email, Avatar, Avg_Session_Length, Time_on_App, Time_on_Website, Length_of_Membership, Yearly_Amount_Spent from EcommerceData
// MAGIC
// MAGIC -- heatmap 

// COMMAND ----------

// MAGIC %sql
// MAGIC select Yearly_Amount_Spent, Time_on_App from EcommerceData
// MAGIC
// MAGIC -- what is the correlation between the yearly amount spend vs time spend on the app? 
// MAGIC -- positively correlated
// MAGIC -- the more time spent on the app, the more they spend yearly

// COMMAND ----------

// MAGIC %sql
// MAGIC select Yearly_Amount_Spent, Avg_Session_Length from EcommerceData
// MAGIC
// MAGIC -- what is the correlation between yearly amount spend vs average session length (website)?
// MAGIC -- positively correlated
// MAGIC -- the greater the avg session length, the greater the yearly amount spent

// COMMAND ----------

// MAGIC %sql
// MAGIC select Yearly_Amount_Spent, Time_on_Website from EcommerceData
// MAGIC
// MAGIC -- what is the correlation between yearly amount spent and the time spent on the website?
// MAGIC -- positively correlated
// MAGIC -- the more time they spend on the website, the more money they spend on the website
// MAGIC -- *storytelling* they are more attracted to the website the longer they stay on the website, causing them to spend more moeny on it

// COMMAND ----------

// MAGIC %md
// MAGIC so far we already performed EDA (exploratory data analytics includes descriptive analytics and diagnositic analytics) the above sql statements, look at the business questions above 
// MAGIC and you need to provide the details explaination as the answers (what is happening) for the above business questions and your answers must include the details explaination why this happens?
// MAGIC next we will perform predictive analytics with the machine learning model that uses the historical dataset to predict the future, we will predict the yearly amount spent by each customers using regression models

// COMMAND ----------

//linear regression model, for prediction

import org.apache.spark.sql.functions._

import org.apache.spark.sql.Row

import org.apache.spark.sql.types._



import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

var StringfeatureCol = Array("Email", "Avatar")
// create an array that contains two string columns for email and avatar

// COMMAND ----------

import org.apache.spark.ml.attribute.Attribute

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}

import org.apache.spark.ml.{Pipeline, PipelineModel}



val indexers = StringfeatureCol.map { colName =>

 new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")

}



val pipeline = new Pipeline()

                   .setStages(indexers)     



val FinalecommerceDF = pipeline.fit(eCommerceDF).transform(eCommerceDF)

// COMMAND ----------

FinalecommerceDF.printSchema()
// display the final ecommerce data frame schema 
// as you can see below the email and avatar columns in a string datatyoe
// what we do here is we take these two columns data and perform an indexing operation
// what we do here is we take these two column data to double data type
// and create two more new columns email indexed and avatar indexed columns
// this is becase the linear regression model cannot process string data types
// we need to prepare our dataset in a proper format first before we can process the data
// this is known as data preparation

// COMMAND ----------

// MAGIC %md
// MAGIC Define the Pipeline
// MAGIC
// MAGIC A predictive model often requires multiple stages of feature preparation.
// MAGIC
// MAGIC A pipeline consists of a series of transformer and estimator stages that typically prepare a DataFrame for modeling and then train a predictive model.
// MAGIC
// MAGIC In this case, you will create a pipeline with stages:
// MAGIC
// MAGIC - A StringIndexer estimator that converts string values to indexes for categorical features
// MAGIC
// MAGIC - A VectorAssembler that combines categorical features into a single vector
// MAGIC
// MAGIC we convert the email and avatar string values to indexes later we will combine these categorical columns into a single vector aka arrays
// MAGIC before we can apply linear regression model on the dataset we need to prepare the dataset first

// COMMAND ----------

FinalecommerceDF.show()

// COMMAND ----------

val splits = FinalecommerceDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// we need to split the dataset into two parts 70% as training data and 30% as testing data
// computers learns from the traning dataset to predict the outcome on the testing dataset
// linear regression model is a supervised learning models thus we need to provide both
// 500 * 70% = 385 (training data) and 500 * 30% = 185 (testing data)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler



val assembler = new VectorAssembler().setInputCols(Array("Email_indexed", "Avatar_indexed", "Avg_Session_Length", "Time_on_App", "Time_on_Website", "Length_of_Membership")).setOutputCol("features")



val training = assembler.transform(train).select($"features", $"Yearly_Amount_Spent".alias("label"))



training.show()

// we combine all the columns above into a single column
// we set a label data for our linear regression model, yearly amount spent (Y axis)
// supervised learning model need to have a label data
// if we know the X then we can know the Y
// if we know the customer email (x1), avatar (x2), avg session length (x3), time on website and apps (x4, x5), 
// duration of the membership (x6) then we can predict the yearly amount spent (Y axis) 

// COMMAND ----------

// MAGIC %md
// MAGIC VectorAssembler(): is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees.
// MAGIC
// MAGIC VectorAssembler accepts the following input column types: all numeric types, boolean type, and vector type.
// MAGIC
// MAGIC In each row, the values of the input columns will be concatenated into a vector in the specified order.
// MAGIC
// MAGIC as you can see from the above codes we combine all the categorical columns data into a single vector something like merge multiple cells in MS excel into a single cell
// MAGIC
// MAGIC before we can encode the string data type for email and avatar columns with indexing techniques to convert string to double data type

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression



val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)

val model = lr.fit(training)

println("Model Trained!")

// send the training dataset into the linear regresision model for learning purposes
// computer learns from the training dataset to predict the future yearly amount spent 
// 70% of the 500 records will be sent as training data and 30% of 500 records will be
// used to predict the future yearly amount spent

// COMMAND ----------

val testing = assembler.transform(test).select($"features", $"Yearly_Amount_Spent".alias("trueLabel"))



testing.show()

// display the testing data with the actual value of yearly amount spent (true label)
// we will apply what we have learned on the training data on this testing dataset
// then we will compare the predicted value with the actual value to measure the accuracy of the prediction

// COMMAND ----------

val prediction = model.transform(testing)

val predicted = prediction.select("features", "prediction", "trueLabel")

predicted.show()

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

val evaluator = new RegressionEvaluator().setLabelCol("trueLabel").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(prediction)

println("Root Mean Square Error (RMSE):"+(rmse))

// COMMAND ----------

// MAGIC %md
// MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average number spending of Customer(ie Yearly_Amount_Spent ) values. You can use the RegressionEvaluator class to retrieve the RMSE.

// COMMAND ----------

predicted.createOrReplaceTempView("eCommerceData")

// COMMAND ----------

// MAGIC %sql
// MAGIC select prediction, trueLabel from eCommerceData

// COMMAND ----------


