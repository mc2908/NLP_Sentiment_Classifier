from csvreader import CsvReader, read_csv
from table import Table

# IMPORT CSV
# creating an instance of CsvReader which analyzes the file
csv_reader = CsvReader('car-reviews.csv', verbose=1)
# If no header is detected automatically. force the first row in the file to be the header
if csv_reader.header is None:
    csv_reader.header = csv_reader.raw_data[0]
#parse the file and return a Table object
tbl = csv_reader.parse()
#set the maximum row to be printed on screen
tbl.max_row_print = 10
print(tbl)



# SHUFFLE THE RECORDS


# SPLIT INTO TRAINING AND TEST DATA


# REMOVE PUNCTUATION AND STOP WORDS


# USE STEMMING


# TRAINING DIFFERENT NAIVE BAYES MODELS (MODEL SELECTION)


# USE K-FOLD VALIDATION TO DETERMINE THE BEST




