package com.test.ml.commons.data.model;

import com.test.ml.commons.MathUtil;

/**
 * 
 * @author vivek
 *
 */
public class Activation {

	private ActivationType type;
	
	public Activation(ActivationType type) {
		this.type = type;
	}
	
	public double apply(double z) {
		double result = -1;
		switch(type) {
		case sigmoid:
			result = MathUtil.sigmoid(z);
			break;
		case tanh:
			throw new UnsupportedOperationException("tanh not yet implemented..");
		case relu:
			throw new UnsupportedOperationException("relu not yet implemented..");
		}
		
		return result;
	}

	public ActivationType getType() {
		return type;
	}

	public void setType(ActivationType type) {
		this.type = type;
	}
	
	
}
