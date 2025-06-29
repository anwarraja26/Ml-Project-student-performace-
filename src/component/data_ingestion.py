import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# It is imported after creating the data_transformation file
from src.component.data_transformation import DataTransformation
from src.component.data_transformation import DataTransformationConfig

#It is imported after creating the model_trainer file
from src.component.model_trainer import ModelTrainer
from src.component.model_trainer import ModelTrainderConfig

# the DataIngestionConfig class is used to store the configuration for data ingestion like creating the path to store the splitted data files.
 
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

# In the below class we are using the DataIngestionConfig class by creating the object for it and access it components
class DataIngestion:
    #The below is the constructor to the DataIngestionConfig class 
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    # The below method is used to initiate the data ingestion process, which includes reading the data from a CSV file, splitting it into train and test sets, and saving them to specified paths.
    # It also handles exceptions and logs the process.
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\stud.csv") # here if we want we can connect out mongo db server or any other database
            if df.empty:
                raise CustomException("No data found in the database")
            
            logging.info('Read the dataset as dataframe using pandas')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test split initiated')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
# initializing the data ingestion component and executing the method
################# The below is to check the data_ingestion.py file 
################## python -m src.component.data_ingestion

# if __name__=="__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()

#################### The below is to check the data_transformation.py file 
#################### python -m src.component.data_ingestion because we call every thing from the data_ingestion.py file

# if __name__=="__main__":
#     obj=DataIngestion()
#     train_data,test_data=obj.initiate_data_ingestion()
#     data_trainsformation=DataTransformation()
#     data_trainsformation.initiate_data_transformation(train_data,test_data)
#     logging.info("Data ingestion completed successfully")
#     logging.info("Data transformation completed successfully")


#################### The below is to check the model_trainer.py file
#################### python -m src.component.data_ingestion

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_trainsformation=DataTransformation()
    train_arr,test_arr,_=data_trainsformation.initiate_data_transformation(train_data,test_data)
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

