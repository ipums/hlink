# Word Count Tutorial - Verifying Your Spark Installation

This guide walks you through running a classic word count job on your newly installed Spark cluster to verify everything is working correctly.

## Prerequisites

1. You have successfully run `setup-spark.py` with your desired role (single, master, or worker)
2. Services are running:
   - HDFS: Check with `jps` - you should see `NameNode` and `DataNode`
   - Spark: Check with `jps` - you should see `Master` and `Worker`
3. You have sourced your environment: `source ~/.bashrc`

## Step 1: Create Sample Data

First, create a text file with some sample content to process:

```bash
cat > sample_text.txt << 'EOF'
Apache Spark is a unified analytics engine for large-scale data processing.
Spark provides high-level APIs in Java, Scala, Python and R.
Spark supports SQL queries, streaming data, machine learning and graph processing.
Apache Spark runs much faster than traditional MapReduce by using in-memory computing.
Spark can run on Hadoop, Apache Mesos, Kubernetes, standalone, or in the cloud.
Spark is designed to be fast and general purpose for big data processing.
The Spark framework includes Spark SQL, Spark Streaming, MLlib and GraphX.
Spark SQL allows querying structured data inside Spark programs.
Spark Streaming enables processing of live data streams.
Machine learning in Spark is powered by the MLlib library.
EOF
```

This creates a text file in your current directory with 10 lines about Apache Spark.

## Step 2: Upload Data to HDFS

Spark reads data from HDFS (Hadoop Distributed File System) by default. You need to upload your sample file:

### 2.1 Create your user directory in HDFS

```bash
~/hadoop-3.3.6/bin/hdfs dfs -mkdir -p /user/$USER
```

This creates `/user/ccd` (or whatever your username is) in HDFS.

### 2.2 Upload the sample file

```bash
~/hadoop-3.3.6/bin/hdfs dfs -put sample_text.txt /user/$USER/sample_text.txt
```

### 2.3 Verify the upload

```bash
~/hadoop-3.3.6/bin/hdfs dfs -ls /user/$USER/
```

Expected output:
```
Found 1 items
-rw-r--r--   1 ccd supergroup        717 2025-11-13 12:31 /user/ccd/sample_text.txt
```

## Step 3: Create the Word Count Script

Create a PySpark script that will count word occurrences:

```bash
cat > word_count.py << 'EOF'
#!/usr/bin/env python3
"""
Classic word count example using PySpark.
This script counts the occurrences of each word in a text file.
"""

from pyspark.sql import SparkSession
import sys

def main():
    # Create Spark session
    spark = SparkSession.builder \
        .appName("WordCount") \
        .getOrCreate()

    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")

    # Read the input file
    input_file = "sample_text.txt"

    print(f"\n{'='*60}")
    print(f"Starting Word Count on: {input_file}")
    print(f"{'='*60}\n")

    # Read text file as RDD
    text_rdd = spark.sparkContext.textFile(input_file)

    # Perform word count:
    # 1. Split lines into words
    # 2. Map each word to (word, 1)
    # 3. Reduce by key to sum counts
    # 4. Sort by count descending
    word_counts = text_rdd \
        .flatMap(lambda line: line.lower().split()) \
        .map(lambda word: (word.strip('.,'), 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda x: x[1], ascending=False)

    # Collect results
    results = word_counts.collect()

    # Display results
    print(f"Total unique words: {len(results)}\n")
    print(f"{'Word':<20} {'Count':>10}")
    print(f"{'-'*20} {'-'*10}")

    for word, count in results:
        print(f"{word:<20} {count:>10}")

    print(f"\n{'='*60}")
    print(f"Word Count Complete!")
    print(f"{'='*60}\n")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
EOF
```

## Step 4: Run the Spark Job

Submit your word count job to Spark:

```bash
~/spark-3.5.7-bin-hadoop3/bin/spark-submit word_count.py
```

### What happens:
1. Spark initializes (you'll see INFO messages about SparkContext, executors, etc.)
2. The job reads data from HDFS
3. Spark performs distributed word counting
4. Results are displayed to your terminal

### Expected Output:

You should see output like this at the end:

```
============================================================
Starting Word Count on: sample_text.txt
============================================================

Total unique words: 69

Word                      Count
-------------------- ----------
spark                        13
data                          5
and                           4
processing                    4
apache                        3
streaming                     3
is                            3
in                            3
sql                           3
the                           3
...

============================================================
Word Count Complete!
============================================================
```

## Step 5: Access Results from HDFS (Optional)

If you want to save results to HDFS for later use, modify the script to write output:

### 5.1 Save results to HDFS

Add this before `spark.stop()` in your script:

```python
# Save results to HDFS
output_path = "word_count_output"
word_counts.saveAsTextFile(output_path)
print(f"\nResults saved to HDFS at: {output_path}\n")
```

### 5.2 View results from HDFS

```bash
# List output files
~/hadoop-3.3.6/bin/hdfs dfs -ls word_count_output/

# View the results
~/hadoop-3.3.6/bin/hdfs dfs -cat word_count_output/part-* | head -20
```

### 5.3 Download results from HDFS

```bash
# Download to local filesystem
~/hadoop-3.3.6/bin/hdfs dfs -get word_count_output ./local_output

# View locally
cat local_output/part-* | head -20
```

## Useful HDFS Commands

Here are common HDFS commands you'll need:

### Listing files
```bash
# List files in a directory
~/hadoop-3.3.6/bin/hdfs dfs -ls /user/$USER/

# Recursive listing
~/hadoop-3.3.6/bin/hdfs dfs -ls -R /user/$USER/
```

### Viewing file contents
```bash
# View entire file
~/hadoop-3.3.6/bin/hdfs dfs -cat /user/$USER/sample_text.txt

# View first 10 lines
~/hadoop-3.3.6/bin/hdfs dfs -cat /user/$USER/sample_text.txt | head -10

# View last 1KB
~/hadoop-3.3.6/bin/hdfs dfs -tail /user/$USER/sample_text.txt
```

### Uploading files
```bash
# Upload single file
~/hadoop-3.3.6/bin/hdfs dfs -put local_file.txt /user/$USER/

# Upload directory
~/hadoop-3.3.6/bin/hdfs dfs -put local_directory/ /user/$USER/

# Upload and overwrite existing file
~/hadoop-3.3.6/bin/hdfs dfs -put -f local_file.txt /user/$USER/
```

### Downloading files
```bash
# Download file
~/hadoop-3.3.6/bin/hdfs dfs -get /user/$USER/sample_text.txt ./

# Download directory
~/hadoop-3.3.6/bin/hdfs dfs -get /user/$USER/my_directory ./
```

### Deleting files
```bash
# Delete file
~/hadoop-3.3.6/bin/hdfs dfs -rm /user/$USER/sample_text.txt

# Delete directory (recursive)
~/hadoop-3.3.6/bin/hdfs dfs -rm -r /user/$USER/my_directory

# Empty trash
~/hadoop-3.3.6/bin/hdfs dfs -expunge
```

### Creating directories
```bash
# Create directory
~/hadoop-3.3.6/bin/hdfs dfs -mkdir /user/$USER/new_directory

# Create nested directories
~/hadoop-3.3.6/bin/hdfs dfs -mkdir -p /user/$USER/path/to/directory
```

### Checking disk usage
```bash
# Show disk usage for directory
~/hadoop-3.3.6/bin/hdfs dfs -du -h /user/$USER/

# Show disk usage summary
~/hadoop-3.3.6/bin/hdfs dfs -du -s -h /user/$USER/
```

### File permissions
```bash
# Change permissions
~/hadoop-3.3.6/bin/hdfs dfs -chmod 755 /user/$USER/sample_text.txt

# Change owner
~/hadoop-3.3.6/bin/hdfs dfs -chown newuser:newgroup /user/$USER/sample_text.txt
```

## Monitoring and Web UIs

### Spark Master UI
View cluster status and running applications:
```
http://localhost:8080
```

Shows:
- Active workers
- Running applications
- Completed applications
- Resource usage

### HDFS NameNode UI
View HDFS status and browse files:
```
http://localhost:9870
```

Shows:
- HDFS capacity and usage
- DataNode status
- File browser
- Logs and metrics

### Spark Application UI
When a Spark job is running, view real-time job details:
```
http://localhost:4040
```

Shows:
- Job stages and tasks
- Storage usage
- Environment configuration
- Executors

## Troubleshooting

### Issue: "command not found: spark-submit"

**Solution**: Source your environment
```bash
source ~/.bashrc
```

Or use full path:
```bash
~/spark-3.5.7-bin-hadoop3/bin/spark-submit word_count.py
```

### Issue: "Input path does not exist: hdfs://localhost:9000/user/..."

**Solution**: File not uploaded to HDFS or wrong path
```bash
# Check if file exists
~/hadoop-3.3.6/bin/hdfs dfs -ls /user/$USER/

# Upload if missing
~/hadoop-3.3.6/bin/hdfs dfs -put sample_text.txt /user/$USER/
```

### Issue: "Connection refused" when accessing HDFS

**Solution**: HDFS services not running
```bash
# Check if NameNode and DataNode are running
jps

# If not, start HDFS
~/hadoop-3.3.6/sbin/start-dfs.sh
```

### Issue: "Connection refused" when accessing Spark

**Solution**: Spark services not running
```bash
# Check if Master and Worker are running
jps

# If not, start Spark
~/spark-3.5.7-bin-hadoop3/sbin/start-all.sh
```

### Issue: Job hangs or runs very slowly

**Possible causes**:
1. Insufficient memory - check Spark UI at http://localhost:4040
2. Too many concurrent jobs - wait for other jobs to complete
3. Network issues in multi-node setup - check worker connectivity

**Check logs**:
```bash
# Spark logs
tail -f ~/spark-3.5.7-bin-hadoop3/logs/spark-*-master-*.out
tail -f ~/spark-3.5.7-bin-hadoop3/logs/spark-*-worker-*.out

# HDFS logs
tail -f ~/hadoop-3.3.6/logs/hadoop-*-namenode-*.log
tail -f ~/hadoop-3.3.6/logs/hadoop-*-datanode-*.log
```

### Issue: "No space left on device"

**Solution**: Clean up HDFS or local disk
```bash
# Check HDFS usage
~/hadoop-3.3.6/bin/hdfs dfs -df -h

# Clean up old files in HDFS
~/hadoop-3.3.6/bin/hdfs dfs -rm -r /user/$USER/old_data

# Empty HDFS trash
~/hadoop-3.3.6/bin/hdfs dfs -expunge
```

## Next Steps

Now that your Spark installation is verified, you can:

1. **Try more complex Spark jobs**: Process larger datasets, use DataFrames/Datasets
2. **Use Spark SQL**: Query structured data with SQL syntax
3. **Explore Spark Streaming**: Process real-time data streams
4. **Try MLlib**: Build machine learning models
5. **Scale up**: Add more worker nodes to your cluster

### Sample commands to explore:

```bash
# Run Spark shell (interactive)
~/spark-3.5.7-bin-hadoop3/bin/pyspark

# Run Spark SQL shell
~/spark-3.5.7-bin-hadoop3/bin/spark-sql

# Run Scala Spark shell
~/spark-3.5.7-bin-hadoop3/bin/spark-shell
```

## Clean Up

When you're done testing, you can clean up the example files:

```bash
# Remove from HDFS
~/hadoop-3.3.6/bin/hdfs dfs -rm /user/$USER/sample_text.txt
~/hadoop-3.3.6/bin/hdfs dfs -rm -r /user/$USER/word_count_output

# Remove local files
rm sample_text.txt word_count.py
```

## Summary

You've successfully:
- Created sample data
- Uploaded files to HDFS
- Run a distributed word count job on Spark
- Learned essential HDFS commands
- Verified your Spark + Hadoop installation is working correctly

Your cluster is now ready for production workloads!
