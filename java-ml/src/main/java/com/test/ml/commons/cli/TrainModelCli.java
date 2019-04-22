package com.test.ml.commons.cli;

import java.io.FileNotFoundException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.IParameterSplitter;
import com.beust.jcommander.converters.IntegerConverter;
import com.test.ml.commons.CSVReader;
import com.test.ml.commons.ModelUtil;
import com.test.ml.commons.NNModelUtil;
import com.test.ml.commons.data.model.Activation;
import com.test.ml.commons.data.model.ActivationType;
import com.test.ml.commons.data.model.nn.NetworkLayout;
import com.test.ml.commons.dtype.Dataset;

/**
 * 
 * @author vivek
 *
 */
public class TrainModelCli {

	@Parameter(names= {"--activation-fn"})
	private String activationFn;
	
	@Parameter(names= {"--data-file"})
	private String dataFile;
	
	@Parameter(names={"--text"})
	private boolean text;
	
	@Parameter(names={"--delim"})
	private String delim;
	
	@Parameter(names= {"--cat-vars"}, splitter = CommaSplitter.class)
	private List<String> catVars = new ArrayList<>();
	
	@Parameter(names= {"--target-label"})
	private String targetLabel;
	
	@Parameter(names= {"--test-ratio"})
	private double testRatio;
	
	@Parameter(names= {"--epoch-count"})
	private int epochCount;

	@Parameter(names= {"--learning-rate"})
	private double learningRate;
	
	@Parameter(names= {"--model-type"}, converter=ModelTypeConvertor.class)
	private ModelType modelType;
	
	@Parameter(names= {"--layer-count"})
	private int layerCount;
	
	@Parameter(names= {"--layer-node-count"}, splitter = CommaSplitter.class, converter = IntegerConverter.class )
	private List<Integer> layerNodeCount = new ArrayList<>();
	
	@Parameter(names= {"--class-count"})
	private int classCount;
	
	@Parameter(names= {"--layer-activations"}, splitter = CommaSplitter.class, converter = ActivationConverter.class )
	private List<ActivationType> activationTypeList = new ArrayList<>();
	
	@Parameter(names= {"--rand-seed"})
	private long randSeed;
	
	
	/*
	 * @Parameter (names= {"--model-type"}) private String modelType;
	 */
	
	public static void main(String[] args) throws IllegalArgumentException, IllegalAccessException, FileNotFoundException {
		TrainModelCli cli = new TrainModelCli();
		
		JCommander.newBuilder()
			.addObject(cli)
			.build()
			.parse(args);
		
		cli.printArgs();
		cli.triggerModel();
	}
	
	private void triggerModel() throws FileNotFoundException {
		CSVReader reader = new CSVReader(dataFile, targetLabel, delim, catVars);
		Map<Dataset.Type, Dataset> datasets = reader.buildDataSet(false);
		Dataset train = datasets.get(Dataset.Type.train);
		
		System.out.println("... ModelType.." + modelType);
		
		switch(modelType) {
		case logit:
			ModelUtil util = new ModelUtil();
			util.fitModel(train.getFeatures(), train.getLabel(), 
					null, null, epochCount, learningRate);
			break;
		case neural_network:
			NNModelUtil nnUtil = new NNModelUtil();
			int featureCount = train.getFeatures().getCols();
			NetworkLayout network = 
					nnUtil.initNetwork(layerCount, featureCount, classCount, layerNodeCount, randSeed, activationTypeList);
			nnUtil.fitModel(train.getFeatures(), train.getLabel(), 
					null, null, epochCount, learningRate, network);
			break;
		default:
			throw new UnsupportedOperationException("ModelType:" + modelType);
		}
	}



	public void printArgs() throws IllegalArgumentException, IllegalAccessException {
		Field[] fields = 
				this.getClass().getDeclaredFields();
		for(Field f : fields) {
			System.out.print(".. " + f.getName());
			System.out.println("\t::" + f.get(this));
		}
	}
	
	public static class CommaSplitter implements IParameterSplitter {

		@Override
		public List<String> split(String value) {
			return Arrays.asList(value.split(","));
		}
		
	}
	
	public static class ModelTypeConvertor implements IStringConverter<ModelType> {

		@Override
		public ModelType convert(String value) {
			return ModelType.fromString(value);
		}
	}
	
	public static class ActivationConverter implements IStringConverter<ActivationType>{

		@Override
		public ActivationType convert(String value) {
			return ActivationType.fromString(value);
		}
		
	}
	
	public static enum ModelType{
		logit, neural_network;
		
		public static ModelType fromString(String t) {
			ModelType result = null;
			System.out.println("t.. " + t);
			if(t.equalsIgnoreCase(logit.name()))
				result = logit;
			else if(t.equalsIgnoreCase(neural_network.toString()))
				result = neural_network;
			else 
				throw new IllegalArgumentException(t);
			
			System.out.println("ModelType.." + result);
			
			return result;
		}
	}
	
}
