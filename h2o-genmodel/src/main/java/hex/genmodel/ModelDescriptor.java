package hex.genmodel;

import hex.ModelCategory;

import java.util.Map;

public interface ModelDescriptor {

  String[][] scoringDomains();

  String projectVersion();

  String algoName();

  String algoFullName();

  ModelCategory getModelCategory();

  boolean isSupervised();

  int nfeatures();

  int nclasses();

  String[] columnNames();

  boolean balanceClasses();

  double defaultThreshold();

  double[] priorClassDist();
  
  Map<String, Map<String, int[]>> targetEncodingMap();

  double[] modelClassDist();

  String uuid();

  String timestamp();

}
