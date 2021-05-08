import unittest

from evaluation import metrics


class TestMetrics(unittest.TestCase):
    def test_mcu_metric(self):
        """ 
        Key:
        1 <- 2 <- 3 <- 4 <- 5
        6 <- 7
        8 <- 9 <- 10 <- 11 <- 12
        Response:
        1 <- 2 <- 3 <- 4 <- 5
        6 <- 7 <- 8 <- 9 <- 10 <- 11 <- 12
        """

        # Create the sets of equivalence (The sets are disjunct!)
        key = [{1, 2, 3, 4, 5}, {6, 7}, {8, 9, 10, 11, 12}]
        response = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11, 12}]

        prec, recall, f1 = metrics.muc(key, response)

        self.assertAlmostEqual(prec, 0.9)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.947, 3)

    def test_bcubed_metric(self):

        # The same task as above
        key = [{1, 2, 3, 4, 5}, {6, 7}, {8, 9, 10, 11, 12}]
        response = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11, 12}]

        prec, recall, f1 = metrics.b_cubed(key, response)

        self.assertAlmostEqual(prec, 0.762, 3)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 0.865, 3)
