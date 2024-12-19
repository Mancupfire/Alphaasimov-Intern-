import numpy as np
import os

THRESHOLD = 0.5


class Sequence:
    def __init__(self, sequence):
        """
            sequence: a list of tuples (value, time)
        """
        self.sequence = np.array([value for value, time in sequence])
        self.time = np.array([time for value, time in sequence])
        self.length, self.std_dev, self.mean = self.calculate()
        self.start_time = self.time[0]
        self.end_time = self.time[-1]
        self.max = np.max(self.sequence)
        self.min = np.min(self.sequence)

    def calculate(self):
        length = len(self.sequence)
        std_dev = np.std(self.sequence)
        mean = np.mean(self.sequence)
        return length, std_dev, mean

    def __str__(self):
        return f"Time: {self.start_time} - {self.end_time}, Length: {self.length}, Std Dev: {self.std_dev}, Mean: {self.mean}, Max: {self.max}, Min: {self.min}"


    
class Sequence_Analyser:
    @staticmethod
    def get_sequences(data_frame, column):
        """
            Get list sequences of numbers from a column of a dataframe
        """
        sequences = []
        current_sequence = []

        for i in range(len(data_frame[column])):
            if len(current_sequence) == 0:
                current_sequence.append((data_frame[column][i], data_frame['time'][i]))

            else:
                diff = abs(data_frame[column][i] - current_sequence[-1][0])
                if diff > THRESHOLD:
                    sequences.append(Sequence(current_sequence))
                    current_sequence = []
                current_sequence.append((data_frame[column][i], data_frame['time'][i]))

        # turn the sequence into a Sequence object and add it to the list of sequences
        sequences.append(Sequence(current_sequence))
        return sequences

    @staticmethod
    def unclassified_sequences_to_csv(unclassified_sequences, file_name):
        """
            From the unclassified sequences, write to a .csv file
        """
        with open(file_name, 'w') as f:
            f.write('Length, Start Time, End Time, Std Dev, Mean\n')
            for s in unclassified_sequences:
                f.write(
                        f"{s.length}, {s.start_time}, {s.end_time}, {s.std_dev}, {s.mean}\n"
                        )
        print(f"File {file_name} written successfully")

    @staticmethod
    def classify_sequences(sequences):
        """
            Divide the sequences into classes and then sort based on its length
        """
        classes = {}
        for s in sequences:
            assert type(s) == Sequence, f"{s} is not a Sequence object"
            if s.length in classes:
                classes[s.length].append(s)
            else:
                classes[s.length] = [s]

        # sort the classes by length
        classes = dict(sorted(classes.items()))
        return classes

    @staticmethod
    def classified_sequences_to_csv(classified_sequences: dict, file_name):
        """
            From dict of classified sequences, write to a .csv file
        """
        with open(file_name, 'w') as f:
            f.write('Length, Start Time, End Time, Std Dev, Mean, Max, Min, Number of sequence\n')
            for i in classified_sequences:
                for j in range(len(classified_sequences[i])):
                    if j==0:
                        f.write(
                                f"{i}, {classified_sequences[i][j].start_time}, {classified_sequences[i][j].end_time},"
                                f"{classified_sequences[i][j].std_dev}, {classified_sequences[i][j].mean},"
                                f"{classified_sequences[i][j].max}, {classified_sequences[i][j].min}, {len(classified_sequences[i])}\n"
                                )
                    else:
                        f.write(
                                f"{i}, {classified_sequences[i][j].start_time}, {classified_sequences[i][j].end_time},"
                                f"{classified_sequences[i][j].std_dev}, {classified_sequences[i][j].mean},"
                                f"{classified_sequences[i][j].max}, {classified_sequences[i][j].min},\n"
                                )
        print(f"File {file_name} written successfully")



class Csv_fomart_corrector:
    @staticmethod
    def count_headlines(file):
        """
            Some .csv files have incorrect header lines, this function checks the number of header lines
        """
        num_header_lines = 0
        with open(file) as f:
            while True:
                line = f.readline().strip()
                # print(line[-1])
                if line[-2].isdigit():
                    return num_header_lines
                num_header_lines += 1
        raise ValueError(f"{file} has no data")
        

    # @staticmethod
    # def check_no_headerline(file):
    #     """
    #         Some .csv files have no header line, this function checks if the file has no header line
    #         If it has, return True, else return False
    #     """
    #     with open(file) as f:
    #         line = f.readline().strip()
    #         if line[-1].isdigit():
    #             return True
    #         else:
    #             return False

    # @staticmethod
    # def remove_headerline(file, newfile):
    #     """
    #         Remove the current headerlimes of the file and save it to a new file
    #     """
    #     assert Csv_fomart_corrector.check_2_headerline(file), f"{file} does not have 2 header lines"
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #     with open(newfile, 'w') as f:
    #         f.writelines(lines[2:])
    #     print(f"File {newfile} written successfully")

    # @staticmethod
    # def add_headerline(file, header, newfile):
        """
            Add a header line to the file and save it to a new file
        """
        assert Csv_fomart_corrector.check_no_headerline(file), f"{file} has header line"
        with open(file, 'r') as f:
            lines = f.readlines()
        with open(newfile, 'w') as f:
            f.write(header + '\n')
            f.writelines(lines)


    @staticmethod
    def fix_format(file, newheaders, to_newfile: bool, newfile=None):
        """
            Remove original header lines if exit, and add a new header line
            You can choose to save the new file to a new file or overwrite the original file.
        """
        if not isinstance(to_newfile, bool):
            raise ValueError("to_newfile must be a boolean")    
        if to_newfile and newfile is None:
            raise ValueError("newfile must be provided when to_newfile is True")
        
        with open(file) as f:
            lines = f.readlines()

        i = Csv_fomart_corrector.count_headlines(file)
        lines = lines[i:]
        
        if to_newfile:
            with open(newfile, 'w') as f:
                f.write(newheaders + '\n')
                f.writelines(lines)
            print(f"File {newfile} created successfully")
        else:
            with open(file, 'w') as f:
                f.write(newheaders + '\n')
                f.writelines(lines)
            print(f"File {file} corrected successfully")