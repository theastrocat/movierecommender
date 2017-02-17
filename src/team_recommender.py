import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np


class MovieRecommender():
    def __init__(self, rank = 100):
        """
        Initialize
        """
        self.ratings_spark = None
        self.train = None
        self.test = None
        self.als_model = None
        self.prediction = None

        self.rank = rank
        self.spark = SparkSession.builder.getOrCreate()

        self.ratings_df = pd.read_csv('data/training.csv')
        self.get_split(self.ratings_df)

        self.requests = self.get_requests()
        self.fitting = False

    def fit(self, full=False):
        self.als_model = ALS(
            itemCol='user',
            userCol='movie',
            ratingCol='rating',
            nonnegative=True,
            regParam=0.1,
            rank=self.rank,
            maxIter=10
            )
        if full == False:
            self.als_model.fit(self.train)
            self.fitting = True
        elif full == True:
            self.als_model.fit(self.ratings_spark)
            self.fitting = True


    def predict(self, dat = 'train'):
        """Dat string representing which dataframe should get predictions: train, test, requests
        """
        if self.fitting:
            if dat == 'train':
                preds = self.transform(self.train_spark_df).collect()
                self.prediction = pd.DataFrame(preds, columns=['user', 'movie', 'rating'])
            if dat == 'test':
                preds = self.transform(self.test_spark_df).collect()
                self.prediction = pd.DataFrame(preds, columns=['user', 'movie', 'rating'])
            if dat == 'requests':
                preds = self.transform(self.requests).collect()
                self.prediction = pd.DataFrame(preds, columns=['user', 'movie', 'rating'])
        else:
            print "Not yet Fit"
            pass

    def get_split(self, data):
        '''
        In init, creat split and rdd
        '''
        self.train = data.sort_values('timestamp', ascending=True)[:-100000]
        self.test = data.sort_values('timestamp', ascending=True)[-100000:]

        self.train_spark_df = self.spark.createDataFrame(self.train).drop('timestamp')
        self.test_spark_df = self.spark.createDataFrame(self.test).drop('timestamp')

        self.ratings_spark = self.spark.createDataFrame(data).drop('timestamp')


    def get_requests(self):
        schema = StructType( [
            StructField('user', IntegerType(), True),
            StructField('movie', IntegerType(), True)])
        requests = self.spark.read.csv('data/requests.csv', header=True, schema = schema, inferSchema=False)
        requests = requests.rdd.map(self.casting_function)
        requests = self.spark.createDataFrame(requests, schema)
        return requests

    def casting_function(self, row):
        id, movie = row
        return int(id), int(movie)

    def write_out(self):
        #Writes the predictions out to file.
        pass
    def show_prediction(self):
        self.prediction.show()

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.critical('you should use run.py instead')
