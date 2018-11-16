package org.apache.hadoop.examples;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class PageRank {

  public static int NODECOUNT = 10876;  // Number of nodes in the graph
  public static int MAXITER = 20;  // Maximum iterations of PageRank calculating
  public static double BETA = 0.8;  // A constant for PageRank calculating

  /* Mapper for initializing the key-value pairs for later calculating */
  /* Input: <in-node>  <out-node> */
  /* Output: <in-node>  <out-node> */
  /*         <in-node>  # */
  public static class InitMapper extends Mapper<Object, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String t1 = itr.nextToken();
        String t2 = itr.nextToken();

        // Collect all nodes (both in-nodes and out-nodes)
        keyText.set(t1);
        valueText.set(t2);
        context.write(keyText, valueText);

        keyText.set(t2);
        valueText.set("#");
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for initializing the key-value pairs for later calculating */
  /* Input: <in-node> <out-node> */
  /*        <in-node> # */
  /* Output: <in-node>  <in-rank>  <list of out-nodes> */
  /*         <in-node>  <in-rank>  X */
  public static class InitReducer extends Reducer<Text, Text, Text, Text> {

    private Text valueText = new Text();
    private double rank;

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      rank = 1.0 / (double)(NODECOUNT);
      String links = new String("");

      // If the node has out-links
      boolean hasOutlinks = false;
      for (Text out : values) {
        String val = out.toString();
        if (val.indexOf("#") == -1) {
          if (hasOutlinks) {
            links += ",";
          }
          hasOutlinks = true;
          links += out.toString();
        }
      }

      if (hasOutlinks) {
        valueText.set(String.valueOf(rank) + "\t" + links);
      }
      else {
        valueText.set(String.valueOf(rank) + "\t" + "X");
      }
      context.write(key, valueText);
    }
  }

  /* Mapper for calculating the PageRank for each node */
  /* Input: <in-node>  <in-rank>  <list of out-nodes> */
  /*        <in-node>  <in-rank>  X */
  /* Output: <in-node>  #<list of out-nodes> */
  /*         <in-node>  #X */
  /*         <out-node> <rank-to-be-added> */
  public static class CalRankMapper extends Mapper<Object, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String in = itr.nextToken();
        String rank = itr.nextToken();
        String linklist = itr.nextToken();

        if (linklist.indexOf("X") == -1) {
          String[] outlinks = linklist.split(",");
          for (String out : outlinks) {
            int numOfNodes = outlinks.length;

            // The PageRank value contributed to the out-node
            double rankAdded = Double.parseDouble(rank) / (double)(numOfNodes);
            keyText.set(out);
            valueText.set(String.valueOf(rankAdded));
            context.write(keyText, valueText);
          }
        }

        keyText.set(in);
        valueText.set("#" + linklist);
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for calculating the PageRank for each node */
  /* Input: <in-node> #<list of out-nodes> */
  /*        <in-node> #X */
  /*        <out-node>  <rank-to-be-added> */
  /* Output: <in-node>  <in-rank> <list of out-nodes> */
  /*         <in-node>  <in-rank> X */
  public static class CalRankReducer extends Reducer<Text, Text, Text, Text> {

    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String linklist = new String("");
      double rankSum = 0.0;

      for (Text value : values) {
        String val = value.toString();

        if (val.indexOf("#") != -1) {
          linklist = val.substring(val.indexOf("#") + 1);
        }
        else {
          // Sum up all PageRank values that come from all in-links
          double rank = Double.parseDouble(val);
          rankSum += rank;
        }
      }
      // PageRank formula
      double newRank = BETA * rankSum + (1.0 - BETA) / (double)(NODECOUNT);

      valueText.set(String.valueOf(newRank) + "\t" + linklist);
      context.write(key, valueText);
    }
  }

  /* Mapper for calculating sum of all the PageRanks for each node */
  /* Input: <in-node>  <in-rank>  <list of out-nodes> */
  /*        <in-node>  <in-rank>  X */
  /* Output: <in-node>  <in-rank> <list of out-nodes> */
  /*         <in-node>  <in-rank> X */
  /*         S  <in-rank> <in-node> */
  public static class SumRankMapper extends Mapper<Object, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());

      while (itr.hasMoreTokens()) {
        String in = itr.nextToken();
        String rank = itr.nextToken();
        String linklist = itr.nextToken();

        keyText.set(in);
        valueText.set(rank + "\t" + linklist);
        context.write(keyText, valueText);

        // Make a key-value pair for summing up the PageRanks
        double rankNum = Double.parseDouble(rank);

        keyText.set("S");
        valueText.set(String.valueOf(rankNum) + "\t" + in);
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for calculating sum of all the PageRanks for each node */
  /* Input: <in-node>  <in-rank> <list of out-nodes> */
  /*        <in-node>  <in-rank> X */
  /*        S  <in-rank>  <in-node> */
  /* Output: <in-node>  <in-rank> <list of out-nodes> */
  /*         <in-node>  <in-rank> X */
  /*         S  <rank-sum>  <list of nodes> */
  public static class SumRankReducer extends Reducer<Text, Text, Text, Text> {

    private Text valueText = new Text();
    private double rankSum = 0.0;

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String keyStr = key.toString();

      // Sum up all the ranks
      if (keyStr.indexOf("S") != -1) {
        double rankSum = 0.0;
        boolean first = true;
        String nodeList = new String("");

        for (Text value : values) {
          StringTokenizer itr = new StringTokenizer(value.toString());
          String rank = itr.nextToken();
          String node = itr.nextToken();

          rankSum += Double.parseDouble(rank);
          if (!first) {
            nodeList += ",";
          }
          nodeList += node;
          first = false;
        }
        valueText.set(String.valueOf(rankSum) + "\t" + nodeList);
        context.write(key, valueText);
      }

      else {
        for (Text value : values) {
          valueText.set(value.toString());
        }
        context.write(key, valueText);
      }
    }
  }

  /* Mapper for renormalizing all the PageRanks for each node */
  /* Input: <in-node>  <in-rank> <list of out-nodes> */
  /*        <in-node>  <in-rank> X */
  /*        S  <rank-sum> <list of nodes> */
  /* Output: <in-node>  <in-rank> <list of out-nodes> */
  /*         <in-node>  <in-rank> X */
  /*         <in-node>  <rank-sum> S */
  public static class NormalizeMapper extends Mapper<Object, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());

      while (itr.hasMoreTokens()) {
        String in = itr.nextToken();

        // Put sum of ranks in all key-value pairs
        if (in.indexOf("S") != -1) {
          String rankSum = itr.nextToken();
          String nodeList = itr.nextToken();

          String[] nodes = nodeList.split(",");
          for (String node : nodes) {
            keyText.set(node);
            valueText.set(rankSum + "\t" + "S");
            context.write(keyText, valueText);
          }
        }
        else {
          String rank = itr.nextToken();
          String linklist = itr.nextToken();

          keyText.set(in);
          valueText.set(rank + "\t" + linklist);
          context.write(keyText, valueText);
        }
      }
    }
  }

  /* Reducer for renormalizing all the PageRanks for each node */
  /* Input: <in-node>  <in-rank> <list of out-nodes> */
  /*        <in-node>  <in-rank> X */
  /*        <in-node>  <rank-sum> S */
  /* Output: <in-node>  <in-rank>  <list of out-nodes> */
  /*         <in-node>  <in-rank>  X */
  public static class NormalizeReducer extends Reducer<Text, Text, Text, Text> {

    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      String rank = new String("");
      String linklist = new String("");
      double rankSum = 1.0;

      for (Text value : values) {
        String val = value.toString();

        // Get the sum of ranks
        if (val.indexOf("S") != -1) {
          StringTokenizer itr = new StringTokenizer(val.toString());
          String sRank = itr.nextToken();
          rankSum = Double.parseDouble(sRank);
        }
        // Get the information of the node
        else {
          StringTokenizer itr = new StringTokenizer(val.toString());
          rank = itr.nextToken();
          linklist = itr.nextToken();
        }
      }

      // Calculate the renormalized rank
      double newRank = Double.parseDouble(rank);
      newRank += ((1.0 - rankSum) / (double)(NODECOUNT));
      valueText.set(String.valueOf(newRank) + "\t" + linklist);
      context.write(key, valueText);
    }
  }

  /* Mapper for sorting all key-value pairs by their PageRanks */
  /* Input: <in-node>  <in-rank>  <list of out-nodes> */
  /*        <in-node>  <in-rank>  X */
  /* Output: <in-rank>  <in-node> */
  public static class SortMapper extends Mapper<Object, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
      StringTokenizer itr = new StringTokenizer(value.toString());
      while (itr.hasMoreTokens()) {
        String node = itr.nextToken();
        String rank = itr.nextToken();
        String linklist = itr.nextToken();

        // Get the inverse of all ranks to get a descending order of ranking
        double nodeRank = Double.parseDouble(rank);
        nodeRank = 1.0 / nodeRank;
        String negRank = String.valueOf(nodeRank);

        // Keep only the node and its rank
        // Swap the node and its rank, in order to sort
        keyText.set(negRank);
        valueText.set(node);
        context.write(keyText, valueText);
      }
    }
  }

  /* Reducer for sorting all key-value pairs by their PageRanks */
  /* Input: <in-rank>  <in-node> */
  /* Output: <in-node>  <in-rank> */
  public static class SortReducer extends Reducer<Text, Text, Text, Text> {

    private Text keyText = new Text();
    private Text valueText = new Text();

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
      for (Text value : values) {
        String node = value.toString();
        String rank = key.toString();

        // Calculate the correct rank back
        double nodeRank = Double.parseDouble(rank);
        nodeRank = 1.0 / nodeRank;
        String finalRank = String.valueOf(nodeRank);

        keyText.set(node);
        valueText.set(finalRank);
        context.write(keyText, valueText);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
      System.err.println("Usage: pagerank <in> <out>");
      System.exit(2);
    }

    // Initialize key-value pairs
    Job job1 = new Job(conf, "Init Graph");
    job1.setJarByClass(PageRank.class);
    job1.setMapperClass(InitMapper.class);
    // job1.setCombinerClass(InitReducer.class);
    job1.setReducerClass(InitReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job1, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job1, new Path(otherArgs[1] + "_0"));
    job1.waitForCompletion(true);

    for (int iter = 0; iter < MAXITER; iter++) {
      // Calculate PageRank for all nodes
      Job job2 = new Job(conf, "Calculate Rank");
      job2.setJarByClass(PageRank.class);
      job2.setMapperClass(CalRankMapper.class);
      job2.setReducerClass(CalRankReducer.class);
      job2.setOutputKeyClass(Text.class);
      job2.setOutputValueClass(Text.class);
      FileInputFormat.addInputPath(job2, new Path(otherArgs[1] + "_" + String.valueOf(iter)));
      FileOutputFormat.setOutputPath(job2, new Path(otherArgs[1] + "_" + String.valueOf(iter) + "a"));
      job2.waitForCompletion(true);

      // Calculate sum of all PageRanks
      Job job3 = new Job(conf, "Sum up Ranks");
      job3.setJarByClass(PageRank.class);
      job3.setMapperClass(SumRankMapper.class);
      job3.setReducerClass(SumRankReducer.class);
      job3.setOutputKeyClass(Text.class);
      job3.setOutputValueClass(Text.class);
      FileInputFormat.addInputPath(job3, new Path(otherArgs[1] + "_" + String.valueOf(iter) + "a"));
      FileOutputFormat.setOutputPath(job3, new Path(otherArgs[1] + "_" + String.valueOf(iter) + "b"));
      job3.waitForCompletion(true);

      // Renormalize all PageRanks
      Job job4 = new Job(conf, "Normalize Ranks");
      job4.setJarByClass(PageRank.class);
      job4.setMapperClass(NormalizeMapper.class);
      job4.setReducerClass(NormalizeReducer.class);
      job4.setOutputKeyClass(Text.class);
      job4.setOutputValueClass(Text.class);
      FileInputFormat.addInputPath(job4, new Path(otherArgs[1] + "_" + String.valueOf(iter) + "b"));
      FileOutputFormat.setOutputPath(job4, new Path(otherArgs[1] + "_" + String.valueOf(iter + 1)));
      job4.waitForCompletion(true);
    }

    // Sort the nodes by their PageRanks
    Job job5 = new Job(conf, "Sort Rank");
    job5.setJarByClass(PageRank.class);
    job5.setMapperClass(SortMapper.class);
    job5.setReducerClass(SortReducer.class);
    job5.setOutputKeyClass(Text.class);
    job5.setOutputValueClass(Text.class);
    FileInputFormat.addInputPath(job5, new Path(otherArgs[1] + "_" + String.valueOf(MAXITER)));
    FileOutputFormat.setOutputPath(job5, new Path(otherArgs[1]));
    job5.waitForCompletion(true);

  }
}
