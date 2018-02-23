package hex.genmodel.algos.glm;

import java.util.Arrays;

public class GlmOrdinalMojoModel extends GlmMojoModelBase {

  private int P;
  private int noff;
  private int lastClass;
  private int secondLastClass;
  private int[] icptIndices;

  GlmOrdinalMojoModel(String[] columns, String[][] domains, String responseColumn) {
    super(columns, domains, responseColumn);
  }

  @Override
  void init() {
    P = _beta.length / _nclasses;
    int firstIcpt = P-1;

    lastClass = _nclasses-1;
    secondLastClass = lastClass-1;
    icptIndices = new int[lastClass];
    for (int c = 0; c < lastClass; c++) {
      icptIndices[c] = P-1+c*P;
    }
    if (P * _nclasses != _beta.length)
      throw new IllegalStateException("Incorrect coding of Beta.");
    noff = _catOffsets[_cats];
  }

  @Override
  double[] glmScore0(double[] data, double[] preds) {
    Arrays.fill(preds, 0.0);

    double etaNoIcpt = 0;
    if (_cats > 0) {
      if (!_useAllFactorLevels) { // skip level 0 of all factors
        for (int i = 0; i < _catOffsets.length - 1; ++i)
          if (data[i] != 0) {
            int ival = (int) data[i] - 1;
            if (ival != data[i] - 1) throw new IllegalArgumentException("categorical value out of range");
            ival += _catOffsets[i];
            if (ival < _catOffsets[i + 1])
              etaNoIcpt += _beta[ival];
          }
      } else { // do not skip any levels
        for (int i = 0; i < _catOffsets.length - 1; ++i) {
          int ival = (int) data[i];
          if (ival != data[i]) throw new IllegalArgumentException("categorical value out of range");
          ival += _catOffsets[i];
          if (ival < _catOffsets[i + 1])
            etaNoIcpt += _beta[ival];
        }
      }
    }

    for (int i = 0; i < _nums; ++i) // add contribution of numeric columns
      etaNoIcpt += _beta[noff + i] * data[i];

    if (etaNoIcpt < _beta[icptIndices[0]]) { // class 0
      preds[0] = 0;
    } else if (etaNoIcpt > _beta[icptIndices[secondLastClass]])
      preds[0] = lastClass;
    else {  // row belongs to class 1 to nclass-2
      for (int c=1; c < lastClass; c++) {
        if (etaNoIcpt >= _beta[icptIndices[c-1]] && etaNoIcpt <_beta[icptIndices[c]]) {
          preds[0] = c;
          break;
        }
      }
    }
    // calculate cdf for each class
    double expEta = Math.exp(etaNoIcpt+_beta[icptIndices[0]]);
    preds[1]  = expEta/(1+expEta);
    double previousCDF = expEta;
    expEta = Math.exp(etaNoIcpt+_beta[icptIndices[secondLastClass]]);
    preds[lastClass] = 1-(expEta/(1+expEta));

    for (int c = 1; c < lastClass; c++) {
      expEta = Math.exp(etaNoIcpt+_beta[icptIndices[c]]);
      double currCDF = expEta/(1+expEta);
      if (currCDF > previousCDF) {
        preds[c+1] = currCDF-previousCDF;
        previousCDF = currCDF;
      } else
        break;
    }
    return preds;
  }
}
