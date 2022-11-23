import unittest
import torch as t
import pandas as pd
import numpy as np
import os
import tabulate
import argparse
import re
import stat

ID = 4  # identifier for dispatcher


class TestAccess(unittest.TestCase):
    def test_access(self):
        try:
            pattern = '/tmp/pycharm_project_[0-9]{1,3}/'
            cwd = os.getcwd()

            # check whether cwd is a PyCharm remote interpreter directory
            match = re.search(pattern, cwd)
            if match:
                path = match.group()

                # check whether group or others have any rights
                if bool(os.stat(path).st_mode & (stat.S_IRWXO | stat.S_IRWXG)):

                    # set the permissions using the users current permissions, but removing group's and other's permissions
                    os.chmod(path, os.stat(path).st_mode & stat.S_IRWXU)

                    # fail with a message
                    self.fail('I fixed the permissions of your PyCharm remote interpreter directory for you.')
        except:  # intentionally broad exception clause. If anything goes wrong, this test should just be ignored.
            pass


class TestDataset(unittest.TestCase):

    def setUp(self):
        # locate the csv file in file system and read it
        csv_path = ''
        for root, _, files in os.walk('.'):
            for name in files:
                if name == 'data.csv':
                    csv_path = os.path.join(root, name)
        self.assertNotEqual(csv_path, '', 'Could not locate the data.csv file')
        self.tab = pd.read_csv(csv_path, sep=';')

    def test_shape(self):
        from data import ChallengeDataset

        val_dl = t.utils.data.DataLoader(ChallengeDataset(self.tab, 'val'), batch_size=1)
        for x, y in val_dl:
            x = x[0].cpu().numpy()
            self.assertEqual(x.shape[0], 3, 'Make sure that your images are converted to RGB')
            self.assertEqual(x.shape[1], 300, 'Your samples are not correctly shaped')
            self.assertEqual(x.shape[2], 300, 'Your samples are not correctly shaped')

            y = y[0].cpu().numpy()
            self.assertEqual(y.size, 2)

            break

    def test_normalization(self):
        from data import ChallengeDataset

        val_dl = t.utils.data.DataLoader(ChallengeDataset(self.tab, 'val'), batch_size=1)
        a = 0.0
        s = np.zeros(3)
        s2 = np.zeros(3)
        for x, _ in val_dl:
            x = x[0].cpu().numpy()
            a += np.prod(x.shape[1:])
            s += np.sum(x, axis=(1, 2))
            s2 += np.sum(x ** 2, axis=(1, 2))

        for i in range(3):
            self.assertTrue(-a * 0.09 < s[i] < a * 0.09, 'Your normalization seems wrong')
            self.assertTrue(a * 0.91 < s2[i] < a * 1.09, 'Your normalization seems wrong')


class TestModel(unittest.TestCase):
    def setUp(self):
        from trainer import Trainer
        from model import ResNet

        self.model = ResNet()
        crit = t.nn.BCELoss()
        trainer = Trainer(self.model, crit, cuda=False)
        trainer.save_onnx('checkpoint_test.onnx')

    def test_prediction(self):
        pred = self.model(t.rand((50, 3, 300, 300)))
        pred = pred.cpu().detach().numpy()

        self.assertEqual(pred.shape[0], 50)
        self.assertEqual(pred.shape[1], 2)
        self.assertFalse(np.isnan(pred).any(), 'Your prediction contains NaN values')
        self.assertFalse(np.isinf(pred).any(), 'Your prediction contains inf values')
        self.assertTrue(np.all([0 <= pred, pred <= 1]), 'Make sure your predictions are sigmoided')

    def test_prediction_after_save_and_load(self):
        import onnxruntime

        ort_session = onnxruntime.InferenceSession('checkpoint_test.onnx')
        ort_inputs = {ort_session.get_inputs()[0].name: t.rand((50, 3, 300, 300)).numpy()}
        pred = ort_session.run(None, ort_inputs)[0]

        self.assertEqual(pred.shape[0], 50)
        self.assertEqual(pred.shape[1], 2)
        self.assertFalse(np.isnan(pred).any(), 'Your prediction contains NaN values')
        self.assertFalse(np.isinf(pred).any(), 'Your prediction contains inf values')
        self.assertTrue(np.all([0 <= pred, pred <= 1]), 'Make sure your predictions are sigmoided')


if __name__ == '__main__':

    import sys
    if sys.argv[-1] == "Bonus":
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestDataset, TestModel]
        percentages = [50, 50]
        total_points = 0
        for test, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(test)).wasSuccessful():
                bonus_points.update({test.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({test.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 1.5
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p),
                          "{:.3f} / 10 (%)".format(p / 100 * exam_percentage)])
        table.append([])
        table.append(["Ex4", "Total Achieved", "", "{} / 100 (%)".format(total_points),
                      "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])

        print(tabulate.tabulate(table, headers=['Pos', 'Test', "Result", 'Percent', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()