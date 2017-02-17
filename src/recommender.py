import logging
import numpy as np

class MovieRecommender():
    def __init__(self):
        """
        blabla
        """
        # if you want to use this...
        self.logger = logging.getLogger()


    def fit(self, ratings):
        """Fits the recommender on the given ratings
        Returns a model (an object that has a predict function)
        """
        self.logger.debug("starting fit")

        # ...

        self.logger.debug("finishing fit")
        return(self)  # we return self to use .predict() on it


    def predict(self, requests):
        """On a given dataframe, predict the ratings.
        Returns the ratings df with an added 'rating' column. (???)
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        # just relying on luck for now...
        output_ratings = np.random.choice(range(1, 5), requests.shape[0])

        self.logger.debug("finishing predict")
        return(output_ratings)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.critical('you should use run.py instead')
