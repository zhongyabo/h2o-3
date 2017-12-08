import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
import numpy as np
import warnings

# HEXDEV-700: GBM reproducibility issue.
def gbm_reproducibility():

# first run with true_reproducibility default to false.  Should not get the correct AUC though and get a warning
  auc_13node = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/jira/threshold_n8_20t_2d_s8568380275437571355.csv"))
  auc_n13 = np.transpose(auc_13node[1].as_data_frame().values)[0]  # grab it as a list
  # grab data and train GBM model for h2o
  cars = h2o.import_file(path=pyunit_utils.locate("bigdata/laptop/jira/reproducibility_issue.csv.zip"))
  gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=20, seed= 8568380275437571355, max_depth = 2,
                                     min_rows = 7)
  gbm.train(x=list(range(2,365)), y="response", training_frame=cars)
  auc_h2o = pyunit_utils.extract_from_twoDimTable(gbm._model_json['output']['training_metrics']._metric_json['thresholds_and_metric_scores'], 'threshold', takeFirst=False)
  assert not(pyunit_utils.equal_two_arrays(auc_n13, auc_h2o, 1e-10, False)), "parameter true_reproducibility is not working."

# set the true_reproducibility to true, should get the correct AUC in this case
  gbm2 = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=20, seed= 8568380275437571355, max_depth = 2,
                                   min_rows = 7, true_reproducibility=True)
  gbm2.train(x=list(range(2,365)), y="response", training_frame=cars)
  auc_h2o2 = pyunit_utils.extract_from_twoDimTable(gbm2._model_json['output']['training_metrics']._metric_json['thresholds_and_metric_scores'], 'threshold', takeFirst=False)
  pyunit_utils.equal_two_arrays(auc_n13, auc_h2o2, 1e-10, True)  # should be equal in this case


if __name__ == "__main__":
  pyunit_utils.standalone_test(gbm_reproducibility)
else:
  gbm_reproducibility()
