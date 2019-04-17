package com.test.ml.commons.dtype;

/**
 * 
 * @author vivek
 *
 */
public class Dataset {
	
	public enum Type {train, test};

	private DoubleMatrix features;
	private DoubleMatrix label;
	
	private Type type; 
	
	public Dataset(DoubleMatrix features, DoubleMatrix label, Type type) {
		super();
		this.features = features;
		this.label = label;
		this.type = type;
	}

	public DoubleMatrix getFeatures() {
		return features;
	}

	public void setFeatures(DoubleMatrix features) {
		this.features = features;
	}

	public DoubleMatrix getLabel() {
		return label;
	}

	public void setLabel(DoubleMatrix label) {
		this.label = label;
	}

	public Type getType() {
		return type;
	}

	public void setType(Type type) {
		this.type = type;
	}
	
}
