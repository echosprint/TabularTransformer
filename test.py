import unittest
from tokenizer import Tokenizer
from util import CATEGORICAL_UNK, SCALAR_UNK, SCALAR_NUMERIC, FeatureType, TaskType
import numpy as np
import pandas as pd
from tabular_transformer import Transformer, ModelArgs
from preprocessor import data_stats, generate_feature_vocab


class TestTokenizer(unittest.TestCase):

    def test_encode(self):
        feature_vocab = {
            f"Name_{CATEGORICAL_UNK}":    0,
            "Name_Alice":                 1,
            "Name_Bob":                   2,
            "Name_Charlie":               3,
            f"Age_{SCALAR_UNK}":          4,
            f"Age_{SCALAR_NUMERIC}":      5,
            f"City_{CATEGORICAL_UNK}":    6,
            "City_San Francisco":         7,
            "City_Los Angeles":           8,
            f"Income_{SCALAR_UNK}":       9,
            f"Income_{SCALAR_NUMERIC}":   10,
        }

        feature_type = {
            "Name": FeatureType.CATEGORICAL,
            "Age": FeatureType.NUMERICAL,
            "City": FeatureType.CATEGORICAL,
            "Income": FeatureType.NUMERICAL,
        }

        data = {
            'Name': ['Alice', 'Bob', CATEGORICAL_UNK, 'Charlie'],
            'Age': [-1.1547, np.nan, 0.5774, 0.5774],
            'City': [CATEGORICAL_UNK, 'San Francisco', 'Los Angeles', 'Los Angeles'],
            'Income': [-0.1650, 1.0722, -0.9073, np.nan]
        }

        tokenizer = Tokenizer(feature_type=feature_type,
                              feature_vocab=feature_vocab)
        print(tokenizer.feature_vocab_item)
        self.assertEqual(tokenizer.feature_vocab_size, 11)
        df = pd.DataFrame(data)
        print(df)
        enc = tokenizer.encode(df)
        print(enc[0], '\n', enc[1])

        model_args = ModelArgs()
        transformer = Transformer(model_args)
        result = transformer(enc[0], enc[1], TaskType.BINCLASS)
        # transformer.configure_optimizers(0.98,0.02,[0.98, 0.97],'cuda')
        print(result)


class TestPreprocessor(unittest.TestCase):

    def test_static(self):
        data = {
            'Name': ['Alice', 'Alice', 'Cda', 'Charlie'],
            'Age': [-1.1547, np.nan, 0.5774, 0.5774],
            'City': ['lssa', 'San Francisco', 'Los Angeles', 'Los Angeles'],
            'Income': [-0.1650, 1.0722, -0.9073, np.nan]
        }
        stats, _ = data_stats(pd.DataFrame(data), 2)
        print(generate_feature_vocab(stats))


if __name__ == '__main__':
    unittest.main()
