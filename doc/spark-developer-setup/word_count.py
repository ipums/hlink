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
