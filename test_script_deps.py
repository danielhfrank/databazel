import pandas

def main():
  df = pandas.read_csv('data.csv', names=['data'])
  print df['data'].mean()

if __name__ == "__main__":
    main()
