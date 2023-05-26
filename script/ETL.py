import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sw
from dateutil import parser


class DataProcessor:
    def __init__(self, dataset_path, top_screens_path):
        self.dataset_path = dataset_path
        self.top_screens_path = top_screens_path
        self.load_data()

    def load_data(self):
        self.dataset = pd.read_csv(self.dataset_path)
        self.top_screens = pd.read_csv(self.top_screens_path)

    def process_data(self):
        self.parse_columns()
        self.classify_customers()
        self.remove_unnecessary_columns()
        self.map_top_screens()
        self.create_funnels()
        self.save_processed_data()

    def parse_columns(self):
        self.dataset['hour'] = self.dataset.hour.str.slice(1, 3).astype(int)
        self.dataset["first_open"] = [parser.parse(
            row_date) for row_date in self.dataset["first_open"]]
        self.dataset["enrolled_date"] = [parser.parse(row_date) if isinstance(
            row_date, str) else row_date for row_date in self.dataset["enrolled_date"]]

    def classify_customers(self):
        self.dataset.loc[self.dataset.difference > 48, 'enrolled'] = 0

    def remove_unnecessary_columns(self):
        self.dataset = self.dataset.drop(
            columns=['enrolled_date', 'difference', 'first_open'])

    def map_top_screens(self):
        self.dataset["screen_list"] = self.dataset.screen_list.astype(
            str) + ','
        for sc in self.top_screens:
            self.dataset[sc] = self.dataset.screen_list.str.contains(
                sc).astype(int)
            self.dataset['screen_list'] = self.dataset.screen_list.str.replace(
                sc+",", "")
        self.dataset['Other'] = self.dataset.screen_list.str.count(",")
        self.dataset = self.dataset.drop(columns=['screen_list'])

    def create_funnels(self):
        savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4",
                           "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
        cm_screens = ["Credit1", "Credit2", "Credit3",
                      "Credit3Container", "Credit3Dashboard"]
        cc_screens = ["CC1", "CC1Category", "CC3"]
        loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
        self.dataset["SavingCount"] = self.dataset[savings_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=savings_screens)
        self.dataset["CMCount"] = self.dataset[cm_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=cm_screens)
        self.dataset["CCCount"] = self.dataset[cc_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=cc_screens)
        self.dataset["LoansCount"] = self.dataset[loan_screens].sum(axis=1)
        self.dataset = self.dataset.drop(columns=loan_screens)

    def save_processed_data(self):
        self.dataset.to_csv(
            '/content/drive/MyDrive/Data_incubator/data_sets/new_appdata10.csv', index=False)

    def generate_report(self):
        analyze_report = sw.analyze(self.dataset)
        analyze_report.show_html(
            '/content/drive/MyDrive/Data_incubator/cb_EDA_output.htm', open_browser=True)


if __name__ == "__main__":
    processor = DataProcessor('/content/drive/MyDrive/Data_incubator/data_sets/appdata10.csv',
                              '/content/drive/MyDrive/Data_incubator/data_sets/top_screens.csv')
    processor.process_data()
    processor.generate_report()
