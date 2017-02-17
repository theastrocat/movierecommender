#!/bin/bash
${SPARK_HOME}/bin/spark-submit \
--master local[4] \
--executor-memory 3G \
--driver-memory 3G \
--conf spark.sql.warehouse.dir="file:///tmp/spark-warehouse" \
--conf "spark.executor.extraJavaOptions=-Dlog4j.rootCategory=ERROR,console" \
--packages com.databricks:spark-csv_2.11:1.5.0 \
--packages com.amazonaws:aws-java-sdk-pom:1.10.34 \
--packages org.apache.hadoop:hadoop-aws:2.7.3 \
$@
