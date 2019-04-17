package com.test.ml.commons.simulate;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 
 * @author vivek
 *
 */
public class CircularDecisionBoundary {

	private String fileName = "test_1.csv";
	
	public static void main(String[] args) throws FileNotFoundException {
		CircularDecisionBoundary util = new CircularDecisionBoundary();
		util.simulateData(10*1000, 12345);
	}
	
	public void simulateData(int recordCount, long seed) throws FileNotFoundException {
		Random rand = new Random(seed);
		int offset = (rand.nextInt(100)+1);
		int radius1 = offset*1000;
		int radius2 = offset*700;
		int origX = offset*500;
		int origY = offset*500;
		
		int xOffset = 100;
		int yOffset = 100;
		
		
		System.out.printf("radius1:%d, radius2:%d, origX:%d, origY:%d\n",
				radius1, radius2, origX, origY);
		
		List<DataPoint2D> data = new ArrayList<>();
		
		for(int i=0;i<recordCount;i++) {
			double theta = rand.nextDouble()*2.0*(22.0/7.0);
			double offsetX = rand.nextDouble()*xOffset;
			//double offsetY = rand.nextDouble()*yOffset;
			
			DataPoint2D point = new DataPoint2D();
			if(rand.nextBoolean()) {
				double x = origX + (radius1 - offsetX) * Math.cos(theta) ;
				double y = origY + (radius1 - offsetX) * Math.sin(theta);
				point.setX(x);
				point.setY(y);
				point.setLabel(0);
			}else {
				double x = origX + (radius2-offsetX) * Math.cos(theta);
				double y = origY + (radius2-offsetX) * Math.sin(theta);
				point.setX(x);
				point.setY(y);
				point.setLabel(1.0);
			}
			
			data.add(point);
			System.out.println(point);
		}
		
		PrintStream ps = new PrintStream(new FileOutputStream(this.fileName));
		
		
		for(DataPoint2D point : data) {
			ps.println(point.toPrint());
		}
		
		ps.close();
		
		/*
		 * for(int theta = 0;theta<360;theta++)
		 * System.out.println(Math.cos(theta/(2.0*22/7)) + "\t" +
		 * Math.sin(theta/(2.0*22/7)));
		 */
	}
	
	public static class DataPoint2D {
		private double x;
		private double y;
		private double label;
		
		public DataPoint2D(double x, double y) {
			this.x = x;
			this.y = y;
		}
		public DataPoint2D() {
		}
		public double getX() {
			return x;
		}
		public void setX(double x) {
			this.x = x;
		}
		public double getY() {
			return y;
		}
		public void setY(double y) {
			this.y = y;
		}
		
		public double getLabel() {
			return label;
		}
		public void setLabel(double label) {
			this.label = label;
		}
		@Override
		public String toString() {
			return "DataPoint2D [x=" + x + ", y=" + y + "]";
		}
		public String toPrint() {
			return x+","+y+","+ (x*x) +","+(y*y)+","+label;
		}
	}

}
