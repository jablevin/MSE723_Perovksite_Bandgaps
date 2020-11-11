import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, f_regression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
import os
import seaborn as sns

class Classifier(object):
    def __init__(self, datafile, n_range, estimators=100, nsteps=5, ftest=True, mutual=True,
                 rfe=True, seed=1, loss_='deviance', testsize=0.15):
        self.data = pd.read_csv(datafile)
        self.x, y = self.data.iloc[:, :-1], self.data.iloc[:, -1]
        self.feature_names=list(self.x.columns)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, y, test_size=.15, random_state=0)

        self.acc = {'f_regression':[], 'mutual':[], 'rfe':[]}
        for n in n_range:
            np.random.seed(seed)
            #########################################################################
            # Train and test on GradientBoostingClassifier with f_regression
            #########################################################################

            if ftest:
                Ftest = SelectKBest(score_func=f_regression, k=n)
                boost_clf = Pipeline((("Feature_select", Ftest),
                                    ("Regression", GradientBoostingClassifier(loss=loss_, n_estimators=estimators))))

                self.y_pred_f = boost_clf.fit(self.x_train, self.y_train).predict(self.x_test)
                self.acc['f_regression'].append(accuracy_score(self.y_test, self.y_pred_f))
                mask_Ftest  = Ftest.get_support()

            #########################################################################
            # Train and test on GradientBoostingClassifier with mutual info regression
            #########################################################################

            if mutual:
                mutual = SelectKBest(score_func=mutual_info_regression, k=n)
                boost_m_clf = Pipeline((("Feature_select", mutual),
                                    ("Regression", GradientBoostingClassifier(loss=loss_, n_estimators=estimators))))

                self.y_pred_m = boost_m_clf.fit(self.x_train, self.y_train).predict(self.x_test)
                self.acc['mutual'].append(accuracy_score(self.y_test, self.y_pred_m))
                mask_mutual = mutual.get_support()

            #########################################################################
            # Recursive Feature Elimination with GradientBoostingClassifier
            #########################################################################

            if rfe:
                rfe = RFE(estimator=GradientBoostingClassifier(loss=loss_, n_estimators=estimators), step=nsteps, n_features_to_select=n)
                self.selector = rfe.fit(self.x_train, self.y_train)
                self.y_pred_r = self.selector.predict(self.x_test)

                self.acc['rfe'].append(rfe.score(self.x_test, self.y_test))
                self.r2 = r2_score(self.y_test, self.y_pred_r)
                mask_rfe    = self.selector.support_

            #########################################################################
            # Test for correlation between features
            #########################################################################

            if ftest:
                self.Ftest_features  = [feature for bool, feature in zip(mask_Ftest, self.feature_names) if bool]
            if mutual:
                self.mutual_features = [feature for bool, feature in zip(mask_mutual, self.feature_names) if bool]
            if rfe:
                self.rfe_features = [feature for bool, feature in zip(mask_rfe, self.feature_names) if bool]
                self.features_rank = self.selector.ranking_
