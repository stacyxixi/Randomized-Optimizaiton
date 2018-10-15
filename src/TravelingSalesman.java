import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanEvaluationFunction;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
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
 * modified from Abagail
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesman {
    /** The n value */
    private static final int N = 50;
    private static String output_dir = "output/travelingSalesman";
    private static String filepath = "travelingSalesman_compare4.csv";
    private static DecimalFormat df1 = new DecimalFormat("0.0000");
    
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();            
        }
        
        
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        
        for (int i = 0; i < 5; i++) {
			String results = "Iteration" + "," + "RHC_optimal" + "," + "RHC_time" + "," + "SA_optimal" + "," + "SA_time"
					+ "," + "GA_optimal" + "," + "GA_time" + "," + "MIMIC_optimal" + "," + "MIMIC_time" + "," + "\n";
			for (int iter = 0; iter < 26; iter++) {

				results += 100 * iter + ",";

				double start = System.nanoTime(), end, trainingTime;
				double optimal_value;
				RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
				FixedIterationTrainer fit = new FixedIterationTrainer(rhc, iter * 100);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(rhc.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(optimal_value);

				start = System.nanoTime();
				SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
				fit = new FixedIterationTrainer(sa, iter * 100);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(sa.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(optimal_value);

				start = System.nanoTime();
				StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
				fit = new FixedIterationTrainer(ga, iter * 100);
				fit.train();
				end = System.nanoTime();
				trainingTime = end - start;
				trainingTime /= Math.pow(10, 9);
				optimal_value = ef.value(ga.getOptimal());
				results += optimal_value + "," + df1.format(trainingTime) + ",";
				System.out.println(trainingTime);
				System.out.println(optimal_value);

				// for mimic we use a sort encoding
				start = System.nanoTime();
				int[] ranges = new int[N];
				Arrays.fill(ranges, N);
				odd = new DiscreteUniformDistribution(ranges);
				Distribution df = new DiscreteDependencyTree(.1, ranges);
				ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

				MIMIC mimic = new MIMIC(200, 100, pop);
				fit = new FixedIterationTrainer(mimic, iter * 100);
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
        System.out.println("End of Run");
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
