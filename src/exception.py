import sys
import logging
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message  # <-- This was missing

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
# the below is to check if the module file is working it is run by python src/exception.py

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     try:
#         a = 1 / 0
#     except Exception as e:
#         logging.info("Divided by zero error")
#         raise CustomException(e, sys)
