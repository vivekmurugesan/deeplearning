package com.test.ml.commons.data.model;

import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class ModelParams {

	private DoubleMatrix weights;
	private double bias;
	
	private double cost;
	
	public ModelParams(DoubleMatrix weights, double bias, double cost) {
		super();
		this.weights = weights;
		this.bias = bias;
		this.cost = cost;
	}

	public DoubleMatrix getWeights() {
		return weights;
	}

	public void setWeights(DoubleMatrix weights) {
		this.weights = weights;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public double getCost() {
		return cost;
	}

	public void setCost(double cost) {
		this.cost = cost;
	}
	
}
