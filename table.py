
import datetime
import re
import copy
import numpy as np

class Table:
    def __init__(self, header=[], data=[], datatypes=[]):
        self.n_cols = len(data)
        self.max_row_print = 50
        self.round_dec = 6
        if self.n_cols == 0:
            self.n_rows = 0
        else:
            self.n_rows = len(data[0])
        self.data = np.array(data)
        self.data_by_row = self.data.T
        #self.data_by_row = list(map(list, zip(*self.data)))
        self.datatypes = datatypes
        self.header = []
        if header is None:
            for k in range(self.n_cols):
                self.header.append(f"Var{k}")
        else:
            for idx, head in enumerate(header):
                if head == "":
                    header[idx] = f"Var{idx}"
            self.header = header
        if self.n_cols > 0:
            Table.add_property(self, self.header, self.data)

    def __getitem__(self, item):
        if not (isinstance(item, int) or isinstance(item, tuple),isinstance(item, slice)):
            raise KeyError("Index must be integer or tuple")
        if isinstance(item, int):
            stop = item + Table.copysign(item, 1)
            row = slice(item, stop, Table.copysign(item, 1))
            col = [k for k in range(self.n_cols)]
        elif isinstance(item, tuple):
            row, col = item
            if isinstance(col, slice):
                start = col.start
                if col.start is None:
                    start = 0
                stop = col.stop
                if col.stop is None:
                    stop = self.n_cols
                col = [n for n in range(start, stop)]
            else:
                col = [col]
            if isinstance(row, int):
                stop = row + Table.copysign(row, 1)
                row = slice(row, stop, Table.copysign(row, 1))
        else:  # must be a slice
            row = item
            col = [k for k in range(self.n_cols)]

        # if row.start < 0 or row.stop <0 or row.start > self.n_rows or row.stop > self.n_rows:
        #     raise KeyError("Index out of range")
        if max(col) > len(self.header)-1 or min(col) < 0:
            raise KeyError("Index out of range")
        sub_data = []
        sub_header = []
        sub_datatypes = []
        if len(col) == 1:
            sub_data = [self.data[col[0]][row]]
            sub_datatypes = [self.datatypes[col[0]]]
            sub_header = [self.header[col[0]]]
        else:
            for c in col:
                sub_data.append(self.data[c][row])
                sub_header.append(self.header[c])
                sub_datatypes.append(self.datatypes[c])
        new_table = Table(sub_header,sub_data,sub_datatypes)
        return new_table

    def __setitem__(self, item, value):
        if not isinstance(item, int):
            raise KeyError("Index must be integer")
        if item < 0:
            raise KeyError("Index out of range")
        if abs(item) > len(self.header)-1:
            raise KeyError("")
        self.data[item] = value

    def __str__(self, row=slice(0, None), column=slice(0, None)):
        if len(self.data) == 0 and len(self.header) == 0:
            return "Table is empty"
        row_max = self.max_row_print
        if row_max is None:
            row_max = 1e8
        data2print = copy.deepcopy(self.data.tolist())
        if len(data2print) == 0:
            data2print = [[] for _ in range(len(self.header))]
        for idx, col in enumerate(data2print):
            head = self.header[idx]
            if "\\n" in repr(head):
                head = repr(head)
            col.insert(0, head)
            #col.insert(0, repr(self.header[idx]))
        s = [[repr(str(e)) if "\\n" in repr(str(e)) else str(e) for idx, e in enumerate(row) if idx <= row_max] for row in data2print]
        lens = [max(map(len, col)) for col in s]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        for idx, col in enumerate(data2print):
            col.insert(1, "-" * lens[idx])
        s = [[repr(str(e)) if "\\n" in repr(str(e)) else str(e) for idx, e in enumerate(row) if idx <= row_max] for row in data2print]
        # lens = [max(map(len, col)) for col in s]
        table = [fmt.format(*row) for row in list(map(list, zip(*s)))]
        if len(self.data) == 0:
            return '\n'.join(table) + "\n"
        elif len(self.data[0]) > row_max:
            return '\n'.join(table) + "\n..."
        return '\n'.join(table) + "\n"

    def mean(self, numeric_only=True):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is datetime.datetime:
                if numeric_only:
                    avg = None
                else:
                    col_values = self[:, idx_c].data[0].copy()
                    col_values.sort()
                    avg = (max(col_values) - min(col_values))/len(col_values)
            elif self.datatypes[idx_c] is int or self.datatypes[idx_c] is float or self.datatypes[idx_c] is complex or self.datatypes[idx_c] is bool:
                #avg = statistics.mean(self[:,idx_c].data[0])
                avg = np.round(np.array(self[:, idx_c].data[0]).mean(), self.round_dec)
            else:  # must be string
                avg = None
            data.append([avg])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def stdev(self, numeric_only):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is datetime.datetime:
                if numeric_only:
                    std = None
                else:
                    col_values = self[:, idx_c].data[0].copy()
                    col_values.sort()
                    delta_val = [0 for _ in range(len(col_values)-1)]
                    for k in range(len(col_values)-1):
                        deltatime = (col_values[k+1]-col_values[k])
                        delta_val[k] = deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.seconds/1e6
                    #std = statistics.stdev(delta_val)
                    std = datetime.timedelta(seconds=np.array(delta_val).std())
                    #std = datetime.timedelta(seconds=std)
            elif self.datatypes[idx_c] is int or self.datatypes[idx_c] is float or self.datatypes[idx_c] is complex or self.datatypes[idx_c] is bool:
                #std = statistics.stdev(self[:,idx_c].data[0])
                std = np.round(np.array(self[:, idx_c].data[0]).std(), self.round_dec)
            else:  # must be string
                std = None
            data.append([std])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def median(self, numeric_only=True):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is datetime.datetime:
                if numeric_only:
                    median = None
                else:
                    col_values = self[:, idx_c].data[0].copy()
                    col_values.sort()
                    delta_val = [0 for _ in range(len(col_values) - 1)]
                    for k in range(len(col_values) - 1):
                        deltatime = (col_values[k + 1] - col_values[k])
                        delta_val[k] = deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.seconds / 1e6
                    median = datetime.timedelta(seconds=np.median(np.array(delta_val)))
            elif self.datatypes[idx_c] is int or self.datatypes[idx_c] is float or self.datatypes[idx_c] is complex:
                #median = statistics.median(self[:, idx_c].data[0])
                median = np.round(np.median(np.array(self[:, idx_c].data[0])), self.round_dec)
            else:  # must be string
                median = None
            data.append([median])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def max(self):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is not str:
                max_val = max(self[:, idx_c].data[0])
            else:  # must be string
                max_val = None
            data.append([max_val])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def min(self):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is not str:
                min_val = min(self[:,idx_c].data[0])
            else:  # must be string
                min_val = None
            data.append([min_val])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def iqr(self, numeric_only=True):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is datetime.datetime:
                if numeric_only:
                    iqr_val = None
                else:
                    col_values = self[:, idx_c].data[0].copy()
                    col_values.sort()
                    delta_val = [0 for _ in range(len(col_values) - 1)]
                    for k in range(len(col_values) - 1):
                        deltatime = (col_values[k + 1] - col_values[k])
                        delta_val[k] = deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.seconds / 1e6
                    val = np.percentile(delta_val,75) - np.percentile(delta_val,25)
                    iqr_val = datetime.timedelta(seconds=val)
            elif self.datatypes[idx_c] is int or self.datatypes[idx_c] is float or self.datatypes[idx_c] is complex:
                # median = statistics.median(self[:, idx_c].data[0])
                iqr_val = np.round(np.percentile(self[:, idx_c].data[0],75)-np.percentile(self[:, idx_c].data[0],25), self.round_dec)
            else:  # must be string or bool
                iqr_val = float("NaN")
            data.append([iqr_val])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def percentile(self,level, numeric_only=True):
        data = []
        for idx_c in range(self.n_cols):
            if self.datatypes[idx_c] is datetime.datetime:
                if numeric_only:
                    iqr_val = None
                else:
                    col_values = self[:, idx_c].data[0].copy()
                    col_values.sort()
                    delta_val = [0 for _ in range(len(col_values) - 1)]
                    for k in range(len(col_values) - 1):
                        deltatime = (col_values[k + 1] - col_values[k])
                        delta_val[k] = deltatime.days * 24 * 3600 + deltatime.seconds + deltatime.seconds / 1e6
                    val = np.percentile(delta_val, level)
                    iqr_val = datetime.timedelta(seconds=val)
            elif self.datatypes[idx_c] is int or self.datatypes[idx_c] is float or self.datatypes[idx_c] is complex:
                # median = statistics.median(self[:, idx_c].data[0])
                iqr_val = np.round(np.percentile(self[:, idx_c].data[0], level), self.round_dec)
            else:  # must be string or Bool
                iqr_val = None
            data.append([iqr_val])
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def outliers(self, col_idx, start = 1.5,end = 3):
        if self.datatypes[col_idx] == datetime.datetime or self.datatypes[col_idx] == str or self.datatypes[col_idx] is bool:
            print("INFO: function not supported for datetime, string or bool data type")
            return Table()
        iqr_val = self.iqr()
        pctl_25 = self.percentile(25).data[col_idx][0]
        pctl_75 = self.percentile(75).data[col_idx][0]
        iqr_col = iqr_val.data[col_idx][0]
        data = []
        for row in self.data_by_row:
            upper1 = pctl_75 + start * iqr_col
            upper2 = pctl_75 + end * iqr_col
            lower1 = pctl_25 - start * iqr_col
            lower2 = pctl_25 - end * iqr_col
            if (upper1 < row[col_idx] <= upper2 or lower2 <= row[col_idx] < lower1):
                data.append(row.copy())
        data = list(map(list,zip(*data)))
        sub_table = Table(self.header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    def weak_outliers(self, col_idx):
        return self.outliers(col_idx, 1.5, 3)

    def strong_outliers(self, col_idx):
        return self.outliers(col_idx, 3, 1e6)

    def stats(self, numeric_only=True):
        stats_tbl_list = []
        stats_tbl_list.append(self.mean(numeric_only))
        stats_tbl_list.append(self.stdev(numeric_only))
        stats_tbl_list.append(self.median(numeric_only))
        stats_tbl_list.append(self.max())
        stats_tbl_list.append(self.min())
        stats_tbl_list.append(self.percentile(25,numeric_only))
        stats_tbl_list.append(self.percentile(75, numeric_only))

        data = [[] for _ in range(self.n_cols)]
        for tbl in stats_tbl_list:
            for idx, col in enumerate(tbl.data):
                data[idx].append(col[0])
        stat_names = ["mean","stdev", "median", "max", "min", "25 percentile", "75 percentile"]
        data.insert(0,stat_names)
        header = ["Statistic"] + self.header
        sub_table = Table(header, data, self.datatypes)
        sub_table.max_row_print = self.max_row_print
        return sub_table

    @staticmethod
    def add_property(cls, header, data):
        for k in range(len(header)):
            header_name = re.sub('\W+', '_', header[k])
            header_name = re.sub('^\d+', '_', header_name)
            setattr(cls, header_name, data[k])

    @staticmethod
    def copysign(x, y):
        y_abs = abs(y)
        if x < 0:
            return -y_abs
        else:
            return y_abs