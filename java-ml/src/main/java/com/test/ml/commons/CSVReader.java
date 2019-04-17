package com.test.ml.commons;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.test.ml.commons.data.model.CategoricalFeature;
import com.test.ml.commons.data.model.Feature;
import com.test.ml.commons.data.model.NumericalFeature;
import com.test.ml.commons.dtype.Dataset;
import com.test.ml.commons.dtype.DoubleMatrix;

/**
 * 
 * @author vivek
 *
 */
public class CSVReader {

	private List<String> headers;
	
	private Map<String, Feature> featureMap;
	
	private int sampleCount;
	
	private String path;
	private String targetLabel;
	private String delim;
	private List<String> catFeatures;
	
	public CSVReader(String path, String targetLabel, String delim, 
			List<String> catFeatures) {
		this.path = path;
		this.targetLabel = targetLabel;
		this.delim = delim;
		this.catFeatures = catFeatures;
		this.headers = new ArrayList<>();
		this.sampleCount = 0;
		
		this.featureMap = new HashMap<>();
	}
	
	private void parseCsv() throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(this.path)));
		
		String header = br.readLine();
		String[] tokens = header.split(delim);
		for(String t : tokens) {
			headers.add(t);
			if(this.catFeatures.contains(t))
				featureMap.put(t, new CategoricalFeature(t));
			else
				featureMap.put(t, new NumericalFeature(t));
		}
		
		System.out.println("Headers.. ::" + this.headers);
		
		String line = br.readLine();
		sampleCount++;
		
		while(line != null && !line.isEmpty()) {
			tokens = line.split(delim);
			for(int i=0;i<tokens.length;i++)
				featureMap.get(headers.get(i)).getData().add(tokens[i]);
			sampleCount++;
			line = br.readLine();
		}
		sampleCount--;
		
		br.close();
	}
	
	public Map<Dataset.Type, Dataset> buildDataSet(boolean trainTestSplit){
		
		Map<Dataset.Type, Dataset> result = new HashMap<>();
		
		try {
			parseCsv();
			
			if(!trainTestSplit) {
				
				DoubleMatrixBuilder featuresBuilder = 
						new DoubleMatrixBuilder(sampleCount);
				DoubleMatrix label = new DoubleMatrix(sampleCount, 1);
				
				for(String k : this.headers) {
					System.out.println(".. Processing data for column..::" + k);
					if(k.equalsIgnoreCase(this.targetLabel))
						populateLabels(label, featureMap.get(k));
					else
						populateCol(featuresBuilder, featureMap.get(k));
				}
				DoubleMatrix features = featuresBuilder.build();
				result.put(Dataset.Type.train, new Dataset(features, label, Dataset.Type.train));
				
				System.out.println("Train.. features..");
				features.printDimension();
				System.out.println("Train.. labels..");
				label.printDimension();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return result;
	}
	
	public void populateLabels(DoubleMatrix label, Feature labelObj) {
		double[] data = labelObj.generateData(false);
		
		for(int i=0;i<sampleCount;i++)
			label.set(i, 0, data[i]);
	}
	
	public void populateCol(DoubleMatrixBuilder builder, Feature featureObj) {
		if(Feature.Type.categorical.equals(featureObj.getFeatureType())) {
			double[][] data = featureObj.generateEncodedData();
			for(double[] colData : data )
				builder.addColData(colData);
		}else 
			builder.addColData(featureObj.generateData(true));
	}
	
	
}
