
import codecs
import datetime
from dateutil.parser import parse as dtu_parse
import re
import copy
from collections import Counter
from table import Table

class CsvReader:
    def __init__(self, file_name, verbose=1):
        self.file_name = file_name
        self.delimiter = ","
        self.encoding = 'utf_8'
        self.raw_data = [] #
        self.data = []
        self.field_types = []
        self.data_types = []
        self.verbose = verbose
        self.header = None
        self.n_records = 0
        self.n_columns = 0
        #self.compatible_data_type = {"int": ["int","none"], "float": ["float", "int", "none"], "complex": ["complex", "float", "int", "none"],
        #                             "string": ["string", "datetime", "int", "float", "complex","none"], "datetime": ["datetime", "none"]}
        self.compatible_data_type = {int: [int, bool, None], float: [float, int, bool, None],
                                     complex: [complex, float, int, bool, None],
                                     str: [str, datetime.datetime, int, float, complex, bool, None],
                                     datetime.datetime: [datetime.datetime, None], None: [None], bool: [bool, None]}
        self.sniff()


    def sniff(self):
        self.autodetect_encoding()
        self.autodectect_delimiter()
        self.read_raw_data()
        self.autodetect_columns_and_records_number()
        self.remove_long_short_records()
        self.autodetect_field_types()
        self.autodetect_header()
        self.autodetect_column_types()

    def parse(self):
        self.remove_records_with_wrong_types()
        self.cast()
        self.data = list(map(list, zip(*self.data)))
        table = Table(self.header, self.data, self.data_types)
        return table

    def read_raw_data(self):
        # this methods reads the csv file char by char and creates a 2D list filled with the data contained in the csv
        # file in string form
        table = []
        _, charList = self.decode_file(self.encoding)
        if charList == "":
            self.raw_data = table
            return
        in_string = False
        escape = False
        n = 0
        table.append([""])
        n_row = 0
        n_field = 0
        while n < len(charList):
            char = charList[n]
            if n+2 >= len(charList):
                next_next_char = ""
            else:
                next_next_char = charList[n + 2]
            if n+1 >= len(charList):
                next_char = ""
            else:
                next_char = charList[n + 1]
            row = table[n_row]
            in_string, new_state = self.set_in_string(char, in_string)
            if new_state:  # first double quote indicating the start of a string. just continue
                n += 1
                continue
            if in_string:
                if self.is_escape(char,next_char,in_string,escape):
                    escape = True
                else:
                    escape = False
                    if self.is_out_string(char,next_char,in_string,escape):
                        in_string = False
                    else:
                        row[n_field] += char
            else:
                if self.is_new_field(char,in_string):
                    row[n_field] = row[n_field].rstrip()
                    n_field += 1
                    if next_char == " ":# discard 1 white spaces after the delimiter
                        n += 1
                        if next_next_char == " ":  # discard 2 white spaces after the delimiter
                            n += 1
                    row.append("")
                else:
                    if self.is_new_line(char):
                        n_row += 1
                        n_field = 0
                        table.append([""])
                    else:
                        row[n_field] += char
            n += 1
        self.raw_data = table


    def is_escape(self, char, next_char, in_string, escape):
        return in_string and char == "\"" and next_char == "\"" and not escape

    def set_in_string(self, char, in_string):
         if not in_string and char == "\"":
             return True, True
         return in_string, False

    def is_out_string(self, char, next_char, in_string, escape):
        return char == "\"" and (next_char == self.delimiter or next_char == "\n") and in_string and not escape

    def is_new_line(self,char):
        return char == "\n" or char == "\r"

    def is_new_field(self,char, in_string):
        return not in_string and char == self.delimiter

    def autodetect_header(self):
    # This methods estimate whether or not the first record is an header or not
        raw_data = self.raw_data
        if self.n_records == 0:
            self.header = None
            return
        first_record = raw_data[0]
        first_row = self.field_types[0]
        for element in first_row:
            if not (element == str or element == None):
                # if there are other types rather then strings in the first record. there can not be an header.
                self.header = None
                if self.verbose >= 1:
                    print("No header detected")
                return
        # the first record contains only string
        header_dict = Counter(first_record)
        most_frequent_header = max(header_dict, key=header_dict.get)
        if not most_frequent_header == "":
            if header_dict[most_frequent_header] > 1:
                # if the string is repeated twice or more, then most likely it's not a header as they tend to be unique
                # in a given table
                self.header = None
                if self.verbose >= 1:
                    print("No header detected")
                return
        # the first record contains only unique strings
        # If the majority of fields in  each column are also string (or none) then I cannot be sure the first record is a header
        has_not_header = [False for _ in range(self.n_columns)]
        types_t = list(map(list, zip(*self.field_types)))
        for idx, col in enumerate(types_t):
            column_counter = Counter(col)
            col_type = max(column_counter, key=column_counter.get)
            if (col_type == str or col_type == None):
                has_not_header[idx] = True
        if all(has_not_header):
            self.header = None
            if self.verbose >= 1:
                print("No header detected")
            return
        # all test passed, the table has the header
        self.header = first_record
        if self.verbose >= 1:
            print(f"Header detected = {self.header}")
        return

    def autodetect_column_types(self):
    # this methods estimates the type of each column by looking at the most frequent type in the column fields
        column_types = [str for _ in range(self.n_columns)]
        types_t = list(map(list, zip(*self.field_types)))
        k = 0
        if self.header is not None:
            k = 1
        for idx,col in enumerate(types_t):
            column_counter = Counter(col[k:])
            col_type = max(column_counter, key=column_counter.get)
            all_col_types = list(column_counter.keys())
            if len(all_col_types) > 1:
                if self.verbose >= 2:
                    print(f"Multiple data types detected in column {idx}. ({all_col_types})")
            if col_type is int or col_type is float or col_type is complex or col_type is bool or col_type is None:
                if col_type is None and str in all_col_types:
                    col_type = str
                elif col_type is not None and str in all_col_types and column_counter[str]/(self.n_records-k) > 0.1:
                    col_type = str
                elif complex in all_col_types:
                    col_type = complex
                elif float in all_col_types:
                    col_type = float
                elif None in all_col_types:
                    col_type = float
                elif int in all_col_types:
                    col_type = int
                elif bool in all_col_types:
                    col_type = bool
            # if col_type == "datetime":
            #     if "string" in all_col_types:
            #         col_type = "string"
            column_types[idx] = col_type
            type_occ = column_counter[col_type]
            confidence = type_occ / (self.n_records-k)
            if self.verbose >= 2:
                print(f"Column {idx} auto-detected data type = {col_type}. Confidence level = {confidence * 100}%")
        if self.verbose >= 1:
            print(f"Auto-detected column data types = {column_types}")
        self.data_types = column_types

    def remove_records_with_wrong_types(self):
        #  this methods removes all records that have at least one filed of a type that does not match its column expected type
        k = 0
        if self.header is not None:
            k = 1
        for col_idx, col_type in enumerate(self.data_types):
            for n_rec in reversed(range(k, self.n_records)):
                if self.field_types[n_rec][col_idx] not in self.compatible_data_type[col_type]:
                    print(
                        f"Incompatible data type found in record {n_rec} column {col_idx}. Expected: {col_type}, found: {self.field_types[n_rec][col_idx]}. Record removed")
                    self.raw_data.pop(n_rec)
                    self.field_types.pop(n_rec)
                    self.n_records -= 1

    def cast(self):
        # this methods cast each field in raw_data into the expected data type of each column
        k = 0
        if self.header is not None:
            k = 1
        raw_data = self.raw_data[k:]
        data = [[[] for _ in range(self.n_columns)] for _ in range(self.n_records-k)]
        for col_idx, col_type in enumerate(self.data_types):
            for j in reversed(range(len(data))):
                datum = raw_data[j][col_idx]
                if col_type is int:
                    if datum == "":
                        datum = 0
                        if self.verbose >= 1:
                            print(f"INFO: Empty field in record {j} column {col_idx} of type {col_type}. Default value applied = {datum}")
                    if self.field_types[j][col_idx] is bool:
                        datum = CsvReader.str2bool(datum)
                    data[j][col_idx] = int(datum)
                elif col_type is float:
                    if datum == "":
                        datum = "NaN"
                        if self.verbose >= 1:
                            print(f"INFO: Empty field in record {j} column {col_idx} of type {col_type}. Default value applied = {datum}")
                    if self.field_types[j][col_idx] is bool:
                        datum = CsvReader.str2bool(datum)
                    data[j][col_idx] = float(datum)
                elif col_type is bool:
                    if datum == "":
                        datum = "False"
                        if self.verbose >= 1:
                            print(f"INFO: Empty field in record {j} column {col_idx} of type {col_type}. Default value applied = {datum}")
                    data[j][col_idx] = CsvReader.str2bool(datum)   # to implement
                elif col_type is complex:
                    if datum == "":
                        real = "NaN"
                        imm = "NaN"
                        if self.verbose >= 1:
                            print(f"INFO: Empty field in record {j} column {col_idx} of type {col_type}. Default value applied = {complex(float(real),float(imm))}")
                    else:
                        imm = 0
                        real = 0
                        if self.field_types[j][col_idx] is bool:
                            real = CsvReader.str2bool(datum)
                        else:
                            components = re.split("[-+]",datum)
                            if "" in components:
                                components.remove("")
                            for comp in components:
                                if re.search("(^[j]|[j]$)",comp):
                                    imm = comp.replace("j","")
                                elif re.search("(^[i]|[i]$)",comp):
                                    imm = comp.replace("i", "")
                                else:
                                    real = comp
                    data[j][col_idx] = complex(float(real), float(imm))
                elif col_type is datetime.datetime:
                    if datum == "":
                        datum = "01-01-0001"
                        if self.verbose >= 1:
                            print(f"INFO: Empty field in record {j} column {col_idx} of type {col_type}. Default value applied = {dtu_parse(datum)}")
                    data[j][col_idx] = dtu_parse(datum)
                elif col_type is str:
                    data[j][col_idx] = datum
        self.data = data



    def autodectect_delimiter(self):
    # this methods attempts to estimate the delimited used in the csv file by looking at the most frequently occourring
    # between "," ";" "\t"
        _ , file_content = self.decode_file(self.encoding)
        delimiters = CsvReader.delimiter_options()
        delimters_dict = {}
        for idx, delimiter in enumerate(delimiters):
            delimters_dict[delimiter] = file_content.count(delimiter)
        self.delimiter = max(delimters_dict, key=delimters_dict.get)
        if self.verbose >= 1:
            print(f"Autodetected delimiter = {repr(self.delimiter)}")

    def autodetect_field_types(self):
        # this methods scans each filed in row_data and tries to estimate what is its the data type
        raw_data = self.raw_data
        types = [[[] for _ in range(self.n_columns)] for _ in range(len(raw_data))]
        for row_idx, row in enumerate(raw_data):
            for col_idx, col in enumerate(row):
                if col == "":
                    types[row_idx][col_idx] = None
                # elif re.search("^[01]$", col):
                #     types[row_idx][col_idx] = bool
                elif re.search("(^[Tt][Rr][Uu][Ee]$|^[Ff][Aa][Ll][Ss][Ee]$|^[Yy][Ee][Ss]$|^[Nn][Oo]$)", col):
                    types[row_idx][col_idx] = bool
                elif re.search("^[-+]?[0-9]+([eE][-+]?[0-9]+)?$", col):
                    types[row_idx][col_idx] = int
                elif re.search("^[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?$", col):
                    types[row_idx][col_idx] = float
                elif re.search("[-+]?[ij][0-9]*(\.)?[0-9]+([eE][-+]?[0-9]+)?$", col):
                    types[row_idx][col_idx] = complex
                elif re.search("[-+]?[0-9]*(\.)?[0-9]+([eE][-+]?[0-9]+)?[ij]$", col):
                    types[row_idx][col_idx] = complex
                elif re.search("\d{2,4}[-.:,/ ]\d{2,4}[-.:,/ ]\d{2,4}([ T]{1,2}\d{2}[-.:,/ ]\d{2}[-.:,/ ]\d{2})?", col):
                    types[row_idx][col_idx] = datetime.datetime
                elif re.search("(\d{2,4}[-.:,/ ][JFMASOND][a-z]{2,9}[-.:,/ ]\d{2,4}|[JFMASOND][a-z]{2,9}[-.:,/ ]\d{2,4}[-.:,/ ]\d{2,4}|\d{2,4}[-.:,/ ]\d{2,4}[-.:,/ ][JFMASOND][a-z]{2,9})", col):
                    types[row_idx][col_idx] = datetime.datetime
                else:
                    types[row_idx][col_idx] = str

                # elif re.search("[\d]{0,4}-[\d]{1,4}-[\d]{1,4}\s?[\d]*:?[\d]*:?[\d]*", col):
                #     types[row_idx][col_idx] = "datetime"
                # elif re.search("[\d]{1,4}:[\d]{1,4}:[\d]{1,4}", col):# matches xxxx:xx:xx
                #     types[row_idx][col_idx] = "datetime"
                # elif re.search("([\d]{1,2}[:-][JFMASOND][a-z]{1,10}[:-][\d]{1,4}|[JFMASOND][a-z]{1,10}[:-][\d]{1,2}[:-][\d]{1,4}|[\d]{1,4}[:-][\d]{1,2}[:-][JFMASOND][a-z]{1,10})", col):  # matches June-10-2019 or 10-June-2001 of 2011-10-June
                #     types[row_idx][col_idx] = "datetime"
                # else:
                #     types[row_idx][col_idx] = "string"
        self.field_types = types
        return True

    def autodetect_columns_and_records_number(self):
        # this methods estimates the number of columns of the data contained in the csv file by looking at the number of
        # columns most records have
        rows_len = [len(row) for row in self.raw_data]
        n_records = len(rows_len)
        self.n_records = n_records
        if n_records == 0:
            self.n_columns = 0
            confidence = 1
        else:
            column_count = Counter(rows_len)
            self.n_columns = max(column_count, key=column_count.get)
            n_occ = column_count[self.n_columns]
            confidence = n_occ / n_records
        if self.verbose >=1:
            print(f"Number of detected columns =  {self.n_columns }. Confidence level =  {confidence*100} %")

    def remove_long_short_records(self):
        # this methods removes all those records with more or fewer number of column than expected
        for k in reversed(range(len(self.raw_data))):
            len_rec = len(self.raw_data[k])
            if len_rec > self.n_columns or len_rec < self.n_columns:
                self.raw_data.pop(k)
                print(f"WARNING: Record number {k+1}: Expected {self.n_columns} column(s), found {len_rec}. Record removed")
        self.n_records = len(self.raw_data)

    def fill_short_records(self):
        raw_data = copy.deepcopy(self.raw_data)
        for k in range(len(raw_data)):
            len_rec = len(raw_data[k])
            if len_rec < self.n_columns:
                raw_data[k].append("")
                print(f"WARNING: Record number {k + 1}: Expected {self.n_columns} column(s), found {len_rec}. The record has been auto filled")
        self.raw_data = raw_data

    def remove_short_records(self):
        raw_data = copy.deepcopy(self.raw_data)
        for k in reversed(range(len(raw_data))):
            len_rec = len(raw_data[k])
            if len_rec < self.n_columns:
                raw_data.pop(k)

                print(f"WARNING: Record number {k + 1}: Expected {self.n_columns} column(s), found {len_rec}. Record removed")
        self.raw_data = raw_data
        self.n_records = len(raw_data)

    def autodetect_encoding(self):
        # try to detect what encoding of the file
        byte_list = []
        default_encode = 'utf_8'
        with open(self.file_name, 'rb') as file:
            for i in range(4):
                byte = file.read(1)
                byte_list.append(byte)

        utf8bom_bytes = byte_list[0] + byte_list[1] + byte_list[2]
        utf16bom_bytes = byte_list[0] + byte_list[1]
        utf32bom_bytes = byte_list[0] + byte_list[1] + byte_list[2] + byte_list[3]
        # test for utf-8-bom
        if utf8bom_bytes == codecs.BOM_UTF8:
            encode_type = 'utf_8_sig'
        # test for utf-16-le
        elif utf16bom_bytes == codecs.BOM_UTF16:
            encode_type = 'utf_16_le'
        # test for utf-16-be
        elif utf16bom_bytes == codecs.BOM_UTF16_BE:
            encode_type = 'utf_16_be'
        # test for utf-32-le
        elif utf32bom_bytes == codecs.BOM_UTF32:
            encode_type = 'utf_32_le'
        # test for utf-32-be
        elif utf32bom_bytes == codecs.BOM_UTF32_BE:
            encode_type = 'utf_32_be'
        else:
            encode_type = default_encode

        encoding_list = ['cp1252']
        encoding_list.insert(0, encode_type)
        for encoding in encoding_list:
            bRet,_ = self.decode_file(encoding)
            if bRet:
                if self.verbose >=1:
                    print(f"File encoding = {encoding}")
                self.encoding = encoding
                return encode_type
        raise ValueError(f"Could not decode {self.file_name}")

    def decode_file(self, encoding):
        file_content = []
        try:
            with open(self.file_name, 'r', encoding=encoding) as csv_file:
                file_content = csv_file.read()
            bRet = True
        except UnicodeDecodeError:
            bRet = False
        return bRet, file_content

    @staticmethod
    def delimiter_options():
        return [",", "\t", ";", "|"]

    @staticmethod
    def str2bool(datum):
        val = datum.lower()
        if val[0] in ["y", "t"]:
            return True
        return False

def read_csv(filename,verbose=1):
    cvs_reader = CsvReader(filename, verbose)
    table = cvs_reader.parse()
    return table

if __name__ == "__main__":
    tbl2 = read_csv('CP_PRO_train_data.csv', verbose=1)
    print(tbl2.header)
    print("Done")
    # t1 = read_csv('test2.csv', verbose=1)
    # print(t1)
    # print(t1.weak_outliers(1))
    #
    # t1 = read_csv('rainfall-1617.csv', verbose=1)
    # print(t1.stats())
    # print(t1.weak_outliers(1))
    # print(t1.strong_outliers(1))
    #
    # t1 = read_csv('outside-temperature-1617.csv', verbose=1)
    # print(t1)
    # t1 = read_csv('indoor-temperature-1617.csv', verbose=1)
    # print(t1)
    # t1 = read_csv('barometer-1617.csv', verbose=1)
    # print(t1)