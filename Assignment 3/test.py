import unittest
import pickle
import pandas as pd
import random
import numpy as np
import requests
import subprocess
import time
from score import *

class TestScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(r"D:/cmi/sem 4/AppliedML/assi3/finetunedlogistic.pkl", 'rb') as model_file:
            cls.loaded_model = pickle.load(model_file)
        cls.test_df = pd.read_csv(r"D:/cmi/sem 4/AppliedML/assi3/test (1).csv")

    def smoke_test(self): # Smoke test
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        result = score(text, self.loaded_model, 0.5)
        self.assertIsNotNone(result)

    def format_test(self):# Format test
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        prediction, propensity = score(text, self.loaded_model, 0.5)
        self.assertIsInstance(prediction, (bool, np.bool_))
        self.assertIsInstance(propensity, float)

    def test_prediction(self):# Check if prediction values are 0 or 1
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertIn(prediction, [True, False])

    def test_propensity(self):# Check if propensity score is between 0 and 1
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        _, propensity = score(text, self.loaded_model, 0.5)
        self.assertTrue(0 <= propensity <= 1)

    def threshold_zero(self):# Test if setting the threshold to 0 always produces prediction as True 
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        prediction, _ = score(text, self.loaded_model, 0.0)
        self.assertTrue(prediction)

    def threshold_one(self):# Test if setting the threshold to 1 always produces prediction as False 
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        prediction, _ = score(text, self.loaded_model, 1.0)
        self.assertFalse(prediction)

    def spam_input(self):# Test if an obvious spam input text results in prediction as True
        text = self.test_df.iat[1, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertTrue(prediction)

    def ham_input(self):# Test if an obvious non-spam input text results in prediction as False  
        text = self.test_df.iat[6, 0]
        prediction, _ = score(text, self.loaded_model, 0.5)
        self.assertFalse(prediction)

class TestFlaskIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flask_process = subprocess.Popen(['python', 'app.py']) 
        time.sleep(10)  
        cls.test_df = pd.read_csv(r"D:/cmi/sem 4/AppliedML/assi3/test (1).csv")      
        
    def lask_test(self):
        k = random.randint(0, len(self.test_df) - 1)
        text = self.test_df.iat[k, 0]
        data = {'text': text}
        response = requests.post('http://127.0.0.1:5000/score', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.json())
        self.assertIn('propensity', response.json())
        
    @classmethod
    def closeflask(cls):# Close Flask app
        cls.flask_process.terminate()

if __name__ == '__main__':
    unittest.main()