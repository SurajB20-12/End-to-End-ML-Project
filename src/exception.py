import sys
import logging
"""
The sys module in Python provides functions and variables that interact with the Python runtime system. It is commonly used in exception handling to retrieve detailed error information.
"""

def error_detail_message(error,error_deatil:sys):
  _,_,exc_tb=error_deatil.exc_info()
  file_name=exc_tb.tb_frame.f_code.co_filename
  error_message="Error occered in a python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))

  return error_message


class Coustom_exception(Exception):
  def __init__(self,error_message,error_detail:sys):
    super().__init__(error_message)
    self.error_message=error_detail_message(error_message,error_deatil=error_detail)

  def __str__(self):
    return self.error_message
  
