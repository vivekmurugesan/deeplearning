package com.test.ml.commons.data.model;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 
 * @author vivek
 *
 */
public class CategoricalFeature extends Feature {
	
	//private List<String> classes;
	private Map<String, Integer> encoding;
	
	
	
	public CategoricalFeature(String featureName, List<String> data) {
		super(featureName, Feature.Type.categorical, data);
		this.encoding = new HashMap<>();
	}
	
	public CategoricalFeature(String featureName) {
		super(featureName, Feature.Type.categorical);
		this.encoding = new HashMap<>();
	}
	
	public double[][] generateEncodedData(){
		extractEncoding();
		double[][] result = new double[encoding.keySet().size()][data.size()];
		for(double[] row : result)
			Arrays.fill(row, 0.0);
		
		int cIndex = 0;
		for(String x : data) {
			int rIndex = encoding.containsKey(x)?encoding.get(x):-1;
			if(rIndex>=0)
				result[rIndex][cIndex] = 1.0;
			cIndex++;
		}
		
		return result;
	}
	
	private void extractEncoding() {
		
		List<String> uniqueVals = 
				data.parallelStream().distinct().collect(Collectors.toList());
		Collections.sort(uniqueVals);
		int index = 0;
		for(String x : uniqueVals) 
			encoding.put(x, index++);
		
		/*
		 * int index = 0; for(String x : data) { if(!encoding.containsKey(x)) {
		 * encoding.put(x, index); index++; } }
		 */
	}

	@Override
	public double[] generateData(boolean applyNormalization) {
		return null;
	}
	
}
