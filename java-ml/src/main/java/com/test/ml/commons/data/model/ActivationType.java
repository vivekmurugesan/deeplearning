package com.test.ml.commons.data.model;

public enum ActivationType {sigmoid, tanh, relu;
	public static ActivationType fromString(String input) {
		ActivationType result=sigmoid;
		if(input.equalsIgnoreCase(sigmoid.toString()))
			result = sigmoid;
		else if(input.equalsIgnoreCase(tanh.toString()))
			result = tanh;
		else if(input.equalsIgnoreCase(relu.toString()))
			result = relu;
		
		return result;
	};
}