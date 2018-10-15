import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class Knapsack {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
    
    private static String output_dir = "output/knapsack";
    private static String filepath = "knapsack_compare4.csv";
    private static DecimalFormat df1 = new DecimalFormat("0.0000");       
    
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        
        
        
        for (int i = 0; i < 5; i++) {
        	
        	
			String results = "Iteration" + "," + "RHC_optimal" + "," + "RHC_time" + "," + "SA_optimal" + "," + "SA_time"
					+ "," + "GA_optimal" + "," + "GA_time" + "," + "MIMIC_optimal" + "," + "MIMIC_time" + "," + "\n";
			for (int iter = 0; iter < 26; iter++) {
				System.out.println("-------------------------Run" + i + "iter" + iter + "-------------------------");
				
				results += 200 * iter + ",";

				double start = System.nanoTime(), end, trainingTime;
				double optimal_value;

				RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
				FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter * 200);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(rhc.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(optimal_value);

				start = System.nanoTime();
				SimulatedAnnealing sa = new SimulatedAnnealing(1E10, .95, hcp);
				fit = new FixedIterationTrainer(sa, iter * 200);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(sa.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(optimal_value);

				start = System.nanoTime();
				StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
				fit = new FixedIterationTrainer(ga, iter * 200);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(ga.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(ef.value(ga.getOptimal()));

				start = System.nanoTime();
				MIMIC mimic = new MIMIC(200, 100, pop);
				fit = new FixedIterationTrainer(mimic, iter * 200);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(mimic.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + "\n";
				System.out.println(trainingTime);
				System.out.println(optimal_value);
			}
			write_output_to_file(output_dir, filepath, results);
		}
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
