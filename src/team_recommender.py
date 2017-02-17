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
        self.recommender = None

        self.rank = rank
        self.spark = SparkSession.builder.getOrCreate()

        self.fitting = False

    def fit(self, data, full=False):
        self.als_model = ALS(
            itemCol='user',
            userCol='movie',
            ratingCol='rating',
            nonnegative=True,
            regParam=0.1,
            rank=self.rank,
            maxIter=10
            )
        self.recommender = self.als_model.fit(data)

    def predict(self, data, ratings_df):
        """Dat string representing which dataframe should get predictions: train, test, requests
        """
        preds = self.recommender.transform(data).collect()
        self.prediction = pd.DataFrame(preds, columns=['user', 'movie', 'prediction'])
        avg_ratings = ratings_df.groupby('movie').mean()
        avg_ratings = avg_ratings.drop('user', axis=1)
        preds_df_nonan = self.prediction.join(avg_ratings, on='movie', how='left', rsuffix='movie')
        preds_df_nonan['prediction'] = preds_df_nonan['prediction'].fillna(preds_df_nonan.ratingmovie)
        self.prediction = preds_df_nonan.drop('ratingmovie', axis=1)
        return self.prediction

    def get_split(self):
        '''
        In init, creat split and rdd
        '''
        self.ratings_df = pd.read_csv('data/training.csv')

        self.train = self.ratings_df.sort_values('timestamp', ascending=True)[:-100000]
        self.test = self.ratings_df.sort_values('timestamp', ascending=True)[-100000:]

        self.train_spark_df = self.spark.createDataFrame(self.train).drop('timestamp')
        self.test_spark_df = self.spark.createDataFrame(self.test).drop('timestamp')

        self.ratings_spark = self.spark.createDataFrame(self.ratings_df).drop('timestamp')


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
