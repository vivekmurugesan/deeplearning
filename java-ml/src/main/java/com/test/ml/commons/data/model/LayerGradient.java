package com.test.ml.commons.data.model;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class LayerGradient {
	
	private int layerNumber;
	
	private DoubleMatrix dWeights;
	private double[] dBias;
	private DoubleMatrix dZ;
	private DoubleMatrix dA;
	
	public LayerGradient(int layerNumber) {
		super();
		this.layerNumber = layerNumber;
	}
	
	public DoubleMatrix getdWeights() {
		return dWeights;
	}
	public void setdWeights(DoubleMatrix dWeights) {
		this.dWeights = dWeights;
	}
	public double[] getdBias() {
		return dBias;
	}
	public void setdBias(double[] dBias) {
		this.dBias = dBias;
	}
	public DoubleMatrix getdZ() {
		return dZ;
	}
	public void setdZ(DoubleMatrix dZ) {
		this.dZ = dZ;
	}
	public DoubleMatrix getdA() {
		return dA;
	}
	public void setdA(DoubleMatrix dA) {
		this.dA = dA;
	}

	public int getLayerNumber() {
		return layerNumber;
	}

	public void setLayerNumber(int layerNumber) {
		this.layerNumber = layerNumber;
	}
	
}
