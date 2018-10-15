import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.*;

/**
 * Modified from AbaloneTest
 * @author Hannah Lau
 * @version 1.0
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to find optimal weights
 * to a neural network that is classifying the Breast Cancer Wisconsin dataset as being benign or malignant
 * 
 */


public class TrainNeuralNetworks {
	
	private static Instance[] instances_train = initializeInstances(478, "data/btrain.csv");
	private static Instance[] instances_test = initializeInstances(205, "data/btest.csv");

    private static int inputLayer = 9, hiddenLayer = 20, outputLayer = 1, trainingIterations = 5000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances_train);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    

    private static DecimalFormat df = new DecimalFormat("0.000");
    private static DecimalFormat df2 = new DecimalFormat("0");
    
    //SA specifications
    private static double startTemp = 1E9;
    private static double coolRate = 1.0;

    //GA specifications
    private static int population =50;
    private static int mate = 30;
    private static int mutate = 30;
    
    private static String output_dir = "output";

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(startTemp, coolRate, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(population, mate, mutate, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
        	
        	String results ="Iteration" + "," + "Error" + ","+ "Training_accuracy" + ","+ "Test_accuracy"+ "\n";
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            results += train(oa[i], networks[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);
            
            results += "\nTraining time: " + df.format(trainingTime) + " seconds";
            String para = "";
            if (i == 1) {
            	para = df2.format(startTemp) + "_" +  df.format(coolRate) + "_";            	
            }
            
            if (i == 2) {
            	para = new Integer(population).toString() + "_" + new Integer(mate).toString()+ "_" + new Integer(mutate).toString() + "_";
            }
            
            String file_path = oaNames[i] + "_" + para + "iter_" + trainingIterations + ".csv";
            write_output_to_file(output_dir, file_path, results);
            //write_output_to_file(output_dir, file_path, "," + df.format(trainingTime), true);
            System.out.println("\n" + file_path);
            System.out.println(results);
            System.out.println("End of results for " + file_path);  
            //System.out.println(inputLayer);
          
        } 
    }

    private static String train(OptimizationAlgorithm oa, BackPropagationNetwork network) {
        
    	
    	String results_training = "";

        for(int i = 0; i < trainingIterations; i++) {
        	double error = 1/oa.train();
            Instance optimalInstance = oa.getOptimal();
            String training_accuracy = df.format(calculate_accuracy(instances_train, optimalInstance, network));
            String test_accuracy = df.format(calculate_accuracy(instances_test, optimalInstance, network));
            //results += df.format(error) + "\n";
            results_training += "Iteration"+ i + "," + df.format(error) + "," +training_accuracy + "," + test_accuracy + "\n";
            }
        return results_training;
        
        
    }

    private static Instance[] initializeInstances(int number, String filename) {

        int n_instances = number;
    	double[][][] attributes = new double[n_instances][][];
    	int n_attributes = 9;
    	

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File(filename)));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[n_attributes]; // # attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < n_attributes; j++) {
                	attributes[i][0][j] = Double.parseDouble(scan.next());
                	//System.out.println(i+" "+j);
                    //System.out.println(attributes[i][0][j]);                  
                }                   

                attributes[i][1][0] = Double.parseDouble(scan.next());
                //System.out.println("label"+ i);
                //System.out.println(attributes[i][1][0]);             
       
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications 0 and 1
            instances[i].setLabel(new Instance(attributes[i][1][0] < 0.5 ? 0 : 1));
            //System.out.println(i);
            //System.out.println(instances[i]);
            //System.out.println(instances[i].getLabel());
        }

        return instances;
    }
    
    private static double calculate_accuracy(Instance[] instances, Instance optimalInstance, BackPropagationNetwork network) {
        int correct = 0, incorrect = 0;
        
        network.setWeights(optimalInstance.getData());
        for(int j = 0; j < instances.length; j++) {
            network.setInputValues(instances[j].getData());
            network.run();

            double predicted = Double.parseDouble(instances[j].getLabel().toString());
            double actual = Double.parseDouble(network.getOutputValues().toString());
            
            //System.out.println("instance"+ j +"predicted:"+predicted+", actual:" + actual);

            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        
        //System.out.println("correct: "+ correct + "; incorrect: " + incorrect);
        return correct * 100.0 / (correct + incorrect);
        
    }
    
    public static void write_output_to_file(String output_dir, String file_path, String result) {
    	try { 
                    String full_path = output_dir + "/" + file_path;
                    PrintWriter pwtr = new PrintWriter(new BufferedWriter(new FileWriter(full_path, true)));
                    synchronized (pwtr) {
                        pwtr.println(result);
                        pwtr.close();
                    }
                
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    
    

}
