package com.test.ml.commons.data.model;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 
 * @author vivek
 *
 */
public class NumericalFeature extends Feature {
	

	public NumericalFeature(String featureName, List<String> data) {
		super(featureName,Feature.Type.numerical, data);
	}
	
	public NumericalFeature(String featureName) {
		super(featureName, Feature.Type.numerical);
	}
	
	public double[] generateData(boolean applyNormalization){
		
		if(applyNormalization)
			return normalize();
		
		double[] dataArr = new double[data.size()];
		int index=0;
		for(String x : data)
			dataArr[index++] = Double.parseDouble(x);
		return dataArr;
	}
	
	public double[] normalize() {
		List<Double> dataList = new ArrayList<>();
		
		for(String x : data)
			dataList.add(Double.parseDouble(x));
		
		double mean = 
				dataList.parallelStream().collect(Collectors.averagingDouble(x -> x));
		double ex2 = 
				dataList.parallelStream().collect(Collectors.averagingDouble(x -> x*x));
		double sd = Math.sqrt(ex2 - mean*mean);
		
		double[] result = new double[data.size()];
		for(int i=0;i<result.length;i++) {
			result[i] = (dataList.get(i)-mean)/sd;
		}
		
		return result;
	}

	@Override
	public double[][] generateEncodedData() {
		return null;
	}
	
	
}
