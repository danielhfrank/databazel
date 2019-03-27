def main():
  data = [float(l) for l in open('data.csv')]
  print(sum(data) / len(data))

if __name__ == "__main__":
    main()
