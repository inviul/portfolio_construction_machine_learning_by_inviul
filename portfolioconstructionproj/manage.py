#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
FILE_DIR = os.path.dirname((os.path.abspath(__file__)))
ROOT_DIR = FILE_DIR[0:FILE_DIR.index('portfolioconstructionproj')].replace("\\","\\\\")
sys.path.append(ROOT_DIR)

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolioconstructionproj.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # data = pd.DataFrame(pd.read_csv("plot.csv"))
    # print(data['Unnamed: 0'])
    # print(data['0'])
    # # plt.plot(data)
    # # plt.savefig("firstTrend.jpg")
    # y= data['Unnamed: 0']
    # x = data['0']
    # plt.barh(y, x)
    # plt.title(f"Feature Ranking for portfolio")
    # plt.figure()
    # plt.savefig("featureranking_1.jpg")

