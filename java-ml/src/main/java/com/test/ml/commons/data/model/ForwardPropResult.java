package com.test.ml.commons.data.model;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class ForwardPropResult {

	private double derivBias;
	private DoubleMatrix derivWeights;
	private double cost;
	
	public ForwardPropResult(double derivBias, DoubleMatrix derivWeights, double cost) {
		super();
		this.derivBias = derivBias;
		this.derivWeights = derivWeights;
		this.cost = cost;
	}
	
	public double getDerivBias() {
		return derivBias;
	}
	public void setDerivBias(double derivBias) {
		this.derivBias = derivBias;
	}
	public DoubleMatrix getDerivWeights() {
		return derivWeights;
	}
	public void setDerivWeights(DoubleMatrix derivWeights) {
		this.derivWeights = derivWeights;
	}
	public double getCost() {
		return cost;
	}
	public void setCost(double cost) {
		this.cost = cost;
	}
	
	
}
