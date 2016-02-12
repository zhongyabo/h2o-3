import sys
from feature.feature_space import *

def main(argv):

  f = open("/Users/ece/0xdata/h2o-3/h2o-test-feature/featureTestCases.csv", "wb")
  f.write('~'.join(["id", "feature", "feature_params", "data_set_ids", "validation_method", "validation_data_set_id",
                      "description"]) + '\n')

  CosFeatureSpace().sample().make_R_tests(f)
  ACosFeatureSpace().sample().make_R_tests(f)
  ACoshFeatureSpace().sample().make_R_tests(f)
  CoshFeatureSpace().sample().make_R_tests(f)
  SinFeatureSpace().sample().make_R_tests(f)
  SinhFeatureSpace().sample().make_R_tests(f)
  ASinFeatureSpace().sample().make_R_tests(f)
  ASinhFeatureSpace().sample().make_R_tests(f)
  TanFeatureSpace().sample().make_R_tests(f)
  TanhFeatureSpace().sample().make_R_tests(f)
  ATanFeatureSpace().sample().make_R_tests(f)
  ATanhFeatureSpace().sample().make_R_tests(f)
  GammaFeatureSpace().sample().make_R_tests(f)
  DigammaFeatureSpace().sample().make_R_tests(f)
  TrigammaFeatureSpace().sample().make_R_tests(f)
  AbsFeatureSpace().sample().make_R_tests(f)
  CeilingFeatureSpace().sample().make_R_tests(f)
  FloorFeatureSpace().sample().make_R_tests(f)
  ExpFeatureSpace().sample().make_R_tests(f)
  Expm1FeatureSpace().sample().make_R_tests(f)
  TruncFeatureSpace().sample().make_R_tests(f)
  IsCharFeatureSpace().sample().make_R_tests(f)
  IsNaFeatureSpace().sample().make_R_tests(f)
  IsNumericFeatureSpace().sample().make_R_tests(f)
  Log10FeatureSpace().sample().make_R_tests(f)
  Log1pFeatureSpace().sample().make_R_tests(f)
  Log2FeatureSpace().sample().make_R_tests(f)
  LogFeatureSpace().sample().make_R_tests(f)
  LGammaFeatureSpace().sample().make_R_tests(f)
  LevelsFeatureSpace().sample().make_R_tests(f)
  NLevelsFeatureSpace().sample().make_R_tests(f)
  NcolFeatureSpace().sample().make_R_tests(f)
  NrowFeatureSpace().sample().make_R_tests(f)
  NotFeatureSpace().sample().make_R_tests(f)
  SignFeatureSpace().sample().make_R_tests(f)
  SqrtFeatureSpace().sample().make_R_tests(f)
  RoundFeatureSpace().sample().make_R_tests(f)
  SignifFeatureSpace().sample().make_R_tests(f)
  AndFeatureSpace().sample().make_R_tests(f)
  OrFeatureSpace().sample().make_R_tests(f)
  DivFeatureSpace().sample().make_R_tests(f)
  ModFeatureSpace().sample().make_R_tests(f)
  MultFeatureSpace().sample().make_R_tests(f)
  SubtFeatureSpace().sample().make_R_tests(f)
  IntDivFeatureSpace().sample().make_R_tests(f)
  ScaleFeatureSpace().sample().make_R_tests(f)
  PowFeatureSpace().sample().make_R_tests(f)
  PlusFeatureSpace().sample().make_R_tests(f)
  GEFeatureSpace().sample().make_R_tests(f)
  GTFeatureSpace().sample().make_R_tests(f)
  LEFeatureSpace().sample().make_R_tests(f)
  LTFeatureSpace().sample().make_R_tests(f)
  EQFeatureSpace().sample().make_R_tests(f)
  NEFeatureSpace().sample().make_R_tests(f)
  AllFeatureSpace().sample().make_R_tests(f)
  CbindFeatureSpace().sample().make_R_tests(f)
  ColnamesFeatureSpace().sample().make_R_tests(f)
  SliceFeatureSpace().sample().make_R_tests(f)
  TableFeatureSpace().sample().make_R_tests(f)
  TableFeatureSpace(two_col=True).sample().make_R_tests(f)
  QuantileFeatureSpace().sample().make_R_tests(f)
  CutFeatureSpace().sample().make_R_tests(f)
  MatchFeatureSpace().sample().make_R_tests(f)
  WhichFeatureSpace().sample().make_R_tests(f)
  RepLenFeatureSpace().sample().make_R_tests(f)
  StrSplitFeatureSpace().sample().make_R_tests(f)
  ToUpperFeatureSpace().sample().make_R_tests(f)
  TransposeFeatureSpace().sample().make_R_tests(f)
  MMFeatureSpace().sample().make_R_tests(f)
  VarFeatureSpace().sample().make_R_tests(f)
  VarFeatureSpace(na=False).sample().make_R_tests(f)

  f.close()

if __name__ == "__main__":
  main(sys.argv)