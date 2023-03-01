def createSparkSession (appName: str ="Default" , master: str = "local[*]"):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName(appName) \
        .master(master) \
        .getOrCreate()
    return spark

def loadSklearnDataset(datasetName: str):
    from sklearn import datasets
    sklearnDataset = getattr(datasets,"load_"+datasetName)
    return sklearnDataset

def convertSklearnDatasetToSparkDataFrame(dataset, spark):
    from pyspark.sql.types import StructField, StructType , DoubleType

    fields = [StructField(dataset.feature_names[i],DoubleType(),True) for i in range(dataset.data.shape[1])]
    fields.append(StructField("Target",DoubleType(),True))
    schema = StructType(fields)
    data_list = dataset.data.tolist()
    target_list = dataset.target.tolist()
    data_target_list = [(data_list[i] + [target_list[i]]) for i in range(len(target_list))]
    data = spark.createDataFrame(data_target_list,schema=schema)
    return data

def writeDataFrameToLocal(data, path :str, format: str = "parquet", mode: str = "overwrite",options: dict = {}):
    data.write.format(format).mode(mode).options(**options).save(path)

def readDataFrameFromLocal(path :str, format: str = "parquet", options: dict = {},spark=None):
    if spark is None:
        spark = createSparkSession()
    data = spark.read.format(format).options(**options).load(path)
    return data

def assembleFeatures(data, featureCols: list, targetCol: str = "Target"):
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    data = assembler.transform(data)
    data = data.select("features", targetCol)
    return data

def exploreData(data):
    data.printSchema()
    data.show(5)

def splitData(data, trainRatio: float = 0.8, seed: int = 42):
    train, test = data.randomSplit([trainRatio, 1 - trainRatio], seed=seed)
    return train, test
