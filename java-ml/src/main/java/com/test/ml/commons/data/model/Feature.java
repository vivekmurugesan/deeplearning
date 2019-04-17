package com.test.ml.commons.data.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 
 * @author vivek
 *
 */
public abstract class Feature {
	public enum Type {numerical, categorical};
	
	protected String featureName;
	protected Type featureType;
	
	protected List<String> data;
	
	public Feature(String featureName, Type featureType, List<String> data) {
		this.featureName = featureName;
		this.featureType = featureType;
		this.data = data;
	}
	
	public Feature(String featureName, Type featureType) {
		this.featureName = featureName;
		this.featureType = featureType;
		this.data = new ArrayList<>();
	}

	public String getFeatureName() {
		return featureName;
	}

	public void setFeatureName(String featureName) {
		this.featureName = featureName;
	}

	public Type getFeatureType() {
		return featureType;
	}

	public void setFeatureType(Type featureType) {
		this.featureType = featureType;
	}

	public List<String> getData() {
		return data;
	}

	public void setData(List<String> data) {
		this.data = data;
	}
	
	public abstract double[] generateData(boolean applyNormalization);
	
	public abstract double[][] generateEncodedData();
}
